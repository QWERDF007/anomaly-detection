import copy
import os
import pickle
from typing import List, Optional, Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import logging
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def is_faiss_gpu_available(device_id: int = 0) -> bool:
    """Dynamically probe whether the current Python environment supports FAISS GPU."""
    if not torch.cuda.is_available():
        return False
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "get_num_gpus"):
        return False
    try:
        if faiss.get_num_gpus() <= 0:
            return False
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(device_id)
        idx = faiss.GpuIndexFlatL2(res, 8, cfg)
        idx.add(np.zeros((1, 8), dtype=np.float32))
        _, _ = idx.search(np.zeros((1, 8), dtype=np.float32), 1)
        return True
    except Exception:
        return False


class FaissNN(object):
    def __init__(
        self,
        on_gpu: Optional[bool] = None,
        num_workers: int = 4,
        device_id: int = 0,
    ) -> None:
        """FAISS Nearest neighbourhood search with dynamic GPU capability probing.

        Args:
            on_gpu: If True, explicitly requests GPU FAISS. If False, forces CPU FAISS.
                   If None (default), dynamically probes whether faiss-gpu is supported.
            num_workers: Number of workers for CPU FAISS similarity search.
            device_id: Target GPU device ID.
        """
        faiss.omp_set_num_threads(num_workers)
        self.device_id = int(device_id)

        # Dynamic probe
        if on_gpu is None or on_gpu is True:
            gpu_supported = is_faiss_gpu_available(self.device_id)
            if on_gpu is True and not gpu_supported:
                LOGGER.info(
                    "Explicitly requested on_gpu=True, but current Python environment lacks faiss-gpu or GPU resources. Gracefully falling back to faiss-cpu."
                )
            self.on_gpu = gpu_supported
        else:
            self.on_gpu = False

        mode_str = f"GPU (device {self.device_id})" if self.on_gpu else "CPU"
        LOGGER.info("FaissNN initialized: using faiss-%s mode", mode_str.lower())
        self.search_index = None

    def _gpu_cloner_options(self):
        if hasattr(faiss, "GpuClonerOptions"):
            return faiss.GpuClonerOptions()
        return None

    def _index_to_gpu(self, index):
        if self.on_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                dev_id = self.device_id.index if hasattr(self.device_id, 'index') and self.device_id.index is not None else int(self.device_id if isinstance(self.device_id, int) else 0)
                res = faiss.StandardGpuResources()
                res.setTempMemory(32 * 1024 * 1024)
                return faiss.index_cpu_to_gpu(
                    res,
                    dev_id,
                    index,
                    self._gpu_cloner_options(),
                )
            except Exception as exc:
                LOGGER.warning("index_cpu_to_gpu failed (%s), fallback to CPU index", exc)
                self.on_gpu = False
                return index
        return index

    def _index_to_cpu(self, index):
        if index is None:
            return None
        if hasattr(index, "to_cpu"):
            try:
                return index.to_cpu()
            except Exception:
                pass
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                pass
        return index

    def _create_index(self, dimension):
        if self.on_gpu and hasattr(faiss, "GpuIndexFlatL2"):
            try:
                dev_id = self.device_id.index if hasattr(self.device_id, 'index') and self.device_id.index is not None else int(self.device_id if isinstance(self.device_id, int) else 0)
                res = faiss.StandardGpuResources()
                res.setTempMemory(32 * 1024 * 1024)
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = dev_id
                return faiss.GpuIndexFlatL2(res, dimension, cfg)
            except Exception as exc:
                LOGGER.warning("GpuIndexFlatL2 creation failed (%s), falling back to faiss-cpu IndexFlatL2", exc)
                self.on_gpu = False
                return faiss.IndexFlatL2(dimension)
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        cpu_idx = self._index_to_cpu(self.search_index)
        try:
            faiss.write_index(cpu_idx, filename)
        except Exception:
            chunk = faiss.serialize_index(cpu_idx)
            with open(filename, "wb") as f:
                f.write(chunk)

    def load(self, filename: str) -> None:
        try:
            raw_idx = faiss.read_index(filename, faiss.IO_FLAG_MMAP)
        except Exception:
            with open(filename, "rb") as f:
                buf = f.read()
            raw_idx = faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))
        self.search_index = self._index_to_gpu(raw_idx)

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
