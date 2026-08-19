#!/usr/bin/env python3
"""Interactive PySide6 GUI for inspecting per-layer Dinomaly2 anomaly heatmaps.

Allows users to:
1. Load a Dinomaly2 checkpoint and run multi-layer encoder-decoder inference;
2. View the combined score map alongside individual layer heatmaps (e.g. Layer 2 ~ 9);
3. Synchronize zoom/pan across all views to inspect micro-defects;
4. Dynamically adjust layer weights with sliders in real time without re-running the model;
5. Switch on/off Gaussian filtering and Guided Filtering (edge alignment) in real time;
6. Hover to probe pixel-level score values across all layers simultaneously.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Insert Dinomaly2 path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import build_model
from utils import (
    cal_anomaly_maps,
    get_gaussian_kernel,
    guided_filter_2d,
    refine_anomaly_map_guided,
)


def load_dinomaly_model(
    model_path: Path,
    backbone: str = "dinov2reg_vit_small_14",
    device: torch.device = torch.device("cuda:0"),
    dropout: float = 0.4,
    la: int = 1,
    lc: int = 2,
    cr: int = 1,
):
    """Build and load Dinomaly2 model with state_dict, auto-detecting backbone size if needed."""
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Auto-detect embed_dim from state_dict to prevent dimension mismatch
    detected_dim = None
    if "encoder.cls_token" in cleaned_sd:
        detected_dim = cleaned_sd["encoder.cls_token"].shape[-1]
    elif "bottleneck.0.0.weight" in cleaned_sd:
        detected_dim = cleaned_sd["bottleneck.0.0.weight"].shape[1]

    if detected_dim == 768:
        if "base" not in backbone:
            backbone = "dinov2reg_vit_base_14" if "reg" in backbone else "dinov2_vit_base_14"
    elif detected_dim == 384:
        if "small" not in backbone:
            backbone = "dinov2reg_vit_small_14" if "reg" in backbone else "dinov2_vit_small_14"
    elif detected_dim == 1024:
        if "large" not in backbone:
            backbone = "dinov2reg_vit_large_14"

    args = argparse.Namespace(
        backbone=backbone,
        dropout=dropout,
        la=la,
        lc=lc,
        cr=cr,
    )
    model = build_model(args, device)
    model.load_state_dict(cleaned_sd, strict=False)
    model.eval()
    return model, backbone


LOGGER = logging.getLogger("dinomaly_layer_gui")

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".JPG",
    ".PNG",
}

COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Turbo": cv2.COLORMAP_TURBO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Hot": cv2.COLORMAP_HOT,
    "Plasma": cv2.COLORMAP_PLASMA,
}


def load_labelme_shapes(json_path: Path) -> List[Dict[str, Any]]:
    """Load LabelMe annotations as drawable shape dicts."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = []
        for s in data.get("shapes", []):
            pts = s.get("points", [])
            if not pts:
                continue
            shapes.append({
                "label": s.get("label", "anomaly"),
                "shape_type": s.get("shape_type", "polygon"),
                "points": [(float(p[0]), float(p[1])) for p in pts],
            })
        return shapes
    except Exception as e:
        LOGGER.warning("Failed to load LabelMe shapes from %s: %s", json_path, e)
        return []


def load_ground_truth_shapes(
    image_path: Path,
    data_root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """Load Ground Truth annotation, prioritizing ground_truth PNG masks over LabelMe JSON."""
    stem = image_path.stem
    cat = image_path.parent.name
    candidates = []

    if data_root and (data_root / "ground_truth").is_dir():
        candidates.extend([
            data_root / "ground_truth" / cat / f"{stem}.png",
            data_root / "ground_truth" / cat / f"{stem}_mask.png",
            data_root / "ground_truth" / f"{stem}.png",
            data_root / "ground_truth" / f"{stem}_mask.png",
        ])
    candidates.extend([
        image_path.parent.parent / "ground_truth" / cat / f"{stem}.png",
        image_path.parent.parent / "ground_truth" / f"{stem}.png",
        image_path.with_suffix(".png"),
    ])

    mask_file = None
    for p in candidates:
        if p.is_file() and p != image_path:
            mask_file = p
            break

    if mask_file is not None:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask > 0):
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            shapes = []
            for cnt in contours:
                pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
                if len(pts) >= 3:
                    shapes.append({
                        "label": "GT Mask 缺陷",
                        "shape_type": "polygon",
                        "points": pts,
                    })
                elif len(pts) == 2:
                    shapes.append({
                        "label": "GT Mask 缺陷",
                        "shape_type": "line",
                        "points": pts,
                    })
                elif len(pts) == 1:
                    shapes.append({
                        "label": "GT Mask 缺陷",
                        "shape_type": "point",
                        "points": pts,
                    })
            return shapes, mask

    # Fallback to LabelMe JSON if no PNG mask
    json_path = image_path.with_suffix(".json")
    if json_path.is_file():
        return load_labelme_shapes(json_path), None

    return [], None


class SyncImageCanvas(QWidget):
    """Interactive canvas with completely independent pan/zoom and coordinate probing."""

    pixel_hovered = Signal(int, int)  # img_x, img_y

    def __init__(
        self,
        title: str = "",
        show_title: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.title = title
        self.show_title = show_title
        self.image: Optional[QImage] = None
        self.raw_bgr: Optional[np.ndarray] = None
        self.score_map: Optional[np.ndarray] = None
        self.shapes: List[Dict[str, Any]] = []

        self.zoom: float = 1.0
        self.pan: QPointF = QPointF(0, 0)
        self._dragging: bool = False
        self._last_mouse: QPoint = QPoint()

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(180, 160)

    def set_content(
        self,
        bgr_image: Optional[np.ndarray],
        score_map: Optional[np.ndarray] = None,
        shapes: Optional[List[Dict[str, Any]]] = None,
        title_suffix: str = "",
    ) -> None:
        self.raw_bgr = bgr_image
        self.score_map = score_map
        if shapes is not None:
            self.shapes = shapes
        if title_suffix:
            self.title_display = f"{self.title} {title_suffix}".strip()
        else:
            self.title_display = self.title

        if bgr_image is not None:
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            self.image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            self.image = None
        self.update()

    def fit_to_window(self) -> None:
        if self.image is None or self.width() <= 0 or self.height() <= 0:
            return
        scale_x = self.width() / self.image.width()
        scale_y = self.height() / self.image.height()
        self.zoom = min(scale_x, scale_y) * 0.95
        center_x = (self.width() - self.image.width() * self.zoom) / 2
        center_y = (self.height() - self.image.height() * self.zoom) / 2
        self.pan = QPointF(center_x, center_y)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._dragging = True
            self._last_mouse = event.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_mouse
            self._last_mouse = event.pos()
            self.pan += QPointF(delta.x(), delta.y())
            self.update()
        if self.image is not None and self.zoom > 0:
            pos = event.position() if hasattr(event, "position") else event.pos()
            img_x = int((pos.x() - self.pan.x()) / self.zoom)
            img_y = int((pos.y() - self.pan.y()) / self.zoom)
            if 0 <= img_x < self.image.width() and 0 <= img_y < self.image.height():
                self.pixel_hovered.emit(img_x, img_y)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        event.accept()

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        pos = event.position() if hasattr(event, "position") else event.pos()
        old_zoom = self.zoom
        new_zoom = max(0.02, min(50.0, old_zoom * factor))
        self.pan = pos - (pos - self.pan) * (new_zoom / old_zoom)
        self.zoom = new_zoom
        self.update()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))

        if self.image is None:
            painter.setPen(QColor(140, 140, 150))
            painter.setFont(QFont("sans-serif", 10))
            painter.drawText(self.rect(), Qt.AlignCenter, getattr(self, "title_display", self.title) or "无数据")
            return

        painter.save()
        painter.translate(self.pan.x(), self.pan.y())
        painter.scale(self.zoom, self.zoom)
        painter.drawImage(0, 0, self.image)

        # Draw GT annotations if any
        if self.shapes:
            pen = QPen(QColor(255, 23, 68, 235), max(1.5 / self.zoom, 1.5))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for s in self.shapes:
                pts = s.get("points", [])
                stype = s.get("shape_type", "rectangle" if len(pts) == 2 else "polygon")
                label = s.get("label", "")

                if stype == "rectangle" and len(pts) >= 2:
                    x1, y1 = pts[0][0], pts[0][1]
                    x2, y2 = pts[1][0], pts[1][1]
                    rx = min(x1, x2)
                    ry = min(y1, y2)
                    rw = abs(x2 - x1)
                    rh = abs(y2 - y1)
                    rect = QRectF(rx, ry, rw, rh)
                    painter.drawRect(rect)
                    # Draw label badge
                    if label:
                        painter.save()
                        painter.setFont(QFont("sans-serif", int(max(9 / self.zoom, 9))))
                        painter.setPen(QColor(255, 23, 68, 255))
                        painter.drawText(QPointF(rx, ry - 3), label)
                        painter.restore()
                elif stype == "line" and len(pts) >= 2:
                    painter.drawLine(QPointF(pts[0][0], pts[0][1]), QPointF(pts[1][0], pts[1][1]))
                elif len(pts) > 2:
                    poly = QPolygonF([QPointF(x, y) for x, y in pts])
                    painter.drawPolygon(poly)
                    if label and len(pts) > 0:
                        painter.save()
                        painter.setFont(QFont("sans-serif", int(max(9 / self.zoom, 9))))
                        painter.setPen(QColor(255, 23, 68, 255))
                        painter.drawText(QPointF(pts[0][0], pts[0][1] - 3), label)
                        painter.restore()
        painter.restore()

        # Draw Title badge at top-left
        if self.show_title:
            title_text = getattr(self, "title_display", self.title)
            if title_text:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 180))
                font = QFont("sans-serif", 9, QFont.Bold)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                rect_w = metrics.horizontalAdvance(title_text) + 16
                rect_h = metrics.height() + 8
                painter.drawRoundedRect(6, 6, rect_w, rect_h, 4, 4)
                painter.setPen(QColor(240, 240, 245))
                painter.drawText(14, 6 + metrics.ascent() + 4, title_text)


class LayerWeightRow(QWidget):
    """A row containing a layer label, slider, spinbox, and enable checkbox."""

    weight_changed = Signal(int, float)  # layer_idx, weight

    def __init__(self, layer_idx: int, layer_name: str, initial_weight: float = 1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = layer_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.checkbox = QCheckBox(layer_name)
        self.checkbox.setChecked(True)
        self.checkbox.setFixedWidth(75)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 300)  # 0.00 to 3.00
        self.slider.setValue(int(initial_weight * 100))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 3.0)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(initial_weight)
        self.spin.setFixedWidth(65)

        layout.addWidget(self.checkbox)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.checkbox.toggled.connect(self._on_toggle)

    def _on_slider(self, val: int) -> None:
        weight = val / 100.0
        self.spin.blockSignals(True)
        self.spin.setValue(weight)
        self.spin.blockSignals(False)
        self.weight_changed.emit(self.layer_idx, weight if self.checkbox.isChecked() else 0.0)

    def _on_spin(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * 100))
        self.slider.blockSignals(False)
        self.weight_changed.emit(self.layer_idx, val if self.checkbox.isChecked() else 0.0)

    def _on_toggle(self, checked: bool) -> None:
        weight = self.spin.value() if checked else 0.0
        self.slider.setEnabled(checked)
        self.spin.setEnabled(checked)
        self.weight_changed.emit(self.layer_idx, weight)

    def get_weight(self) -> float:
        return self.spin.value() if self.checkbox.isChecked() else 0.0

    def set_weight(self, weight: float) -> None:
        self.checkbox.setChecked(weight > 0)
        self.spin.setValue(weight)
        self.slider.setValue(int(weight * 100))


class LayerInspectionWindow(QMainWindow):
    """Main application window for inspecting multi-layer Dinomaly2 anomaly heatmaps."""

    def __init__(
        self,
        default_model: Optional[str] = None,
        default_data: Optional[str] = None,
        default_backbone: Optional[str] = None,
        default_size: int = 672,
        gpu: int = 0,
    ):
        super().__init__()
        self.setWindowTitle("Dinomaly2 逐层特征热力图与权重调节分析系统")
        self.resize(1720, 980)

        device_str = f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu"
        self.device = torch.device(device_str)
        self.model = None
        self.current_model_path: Optional[Path] = None
        self.target_layers: List[int] = [2, 3, 4, 5, 6, 7, 8, 9]

        self.current_image_path: Optional[Path] = None
        self.current_bgr: Optional[np.ndarray] = None
        self.current_shapes: List[Dict[str, Any]] = []

        # Raw layer anomaly maps before upsampling/smoothing: list of 2D numpy arrays
        self.raw_layer_maps: List[np.ndarray] = []  # shape (H_orig, W_orig) per layer
        self.combined_map: Optional[np.ndarray] = None

        self._init_ui()

        if default_backbone:
            idx = self.backbone_combo.findText(default_backbone)
            if idx >= 0:
                self.backbone_combo.setCurrentIndex(idx)
        if default_size:
            self.size_spin.setValue(default_size)

        if default_model and Path(default_model).is_file():
            self.model_edit.setText(default_model)
            self._load_model()
        if default_data and Path(default_data).is_dir():
            self.data_edit.setText(default_data)
            self._rebuild_file_list()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(6)

        # 1. Top Control Bar (Model, Data, Backbone, Global Buttons)
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Main Content Splitter (Left: Controls & Files, Right: Views)
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter, 1)

        # Left Panel (File list + Layer Sliders + Refinement options)
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)
        content_splitter.setStretchFactor(0, 0)

        # Right Panel (Grid of Canvases)
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([380, 1340])

        # 3. Bottom Status Bar (Pixel Inspector Probe)
        self.probe_label = QLabel("光标探针：移动鼠标至图像上可查看当前像素坐标及各层具体异常分数值")
        self.probe_label.setStyleSheet("background-color: #1e1e24; color: #00e5ff; font-family: monospace; font-size: 13px; padding: 6px 12px; border-radius: 4px;")
        main_layout.addWidget(self.probe_label)

    def _create_top_bar(self) -> QWidget:
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(4)

        # Row 1: Model, Data, Backbone, Image size, Run button
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("模型权重:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("/path/to/model.pth")
        row1.addWidget(self.model_edit, 2)
        btn_browse_model = QPushButton("浏览...")
        btn_browse_model.clicked.connect(self._browse_model)
        row1.addWidget(btn_browse_model)

        row1.addWidget(QLabel("数据根目录:"))
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("/path/to/dataset")
        self.data_edit.returnPressed.connect(self._rebuild_file_list)
        row1.addWidget(self.data_edit, 2)
        btn_browse_data = QPushButton("浏览...")
        btn_browse_data.clicked.connect(self._browse_data)
        row1.addWidget(btn_browse_data)

        row1.addWidget(QLabel("Backbone:"))
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems([
            "dinov2reg_vit_small_14",
            "dinov2reg_vit_base_14",
            "dinov2reg_vit_large_14",
            "dinov2_vit_small_14",
            "dinov2_vit_base_14",
        ])
        row1.addWidget(self.backbone_combo)

        row1.addWidget(QLabel("尺寸:"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(224, 2048)
        self.size_spin.setSingleStep(14)
        self.size_spin.setValue(672)
        self.size_spin.setDecimals(0)
        self.size_spin.setFixedWidth(70)
        row1.addWidget(self.size_spin)

        self.btn_load = QPushButton("⚡ 加载模型 & 推理")
        self.btn_load.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 4px 12px;")
        self.btn_load.clicked.connect(self._on_load_or_run)
        row1.addWidget(self.btn_load)

        v_layout.addLayout(row1)

        # Row 2: Layer Spinbox, GT display checkbox, Fit window button
        row2 = QHBoxLayout()
        row2.setSpacing(10)

        row2.addWidget(QLabel("<b>🔍 查看特征层:</b>"))
        self.btn_prev_layer = QPushButton("◀ 上一层")
        self.btn_prev_layer.setFixedWidth(75)
        self.btn_prev_layer.clicked.connect(self._prev_layer)
        row2.addWidget(self.btn_prev_layer)

        self.layer_spin = QDoubleSpinBox()
        self.layer_spin.setRange(2, 9)
        self.layer_spin.setSingleStep(1)
        self.layer_spin.setDecimals(0)
        self.layer_spin.setValue(2)
        self.layer_spin.setFixedWidth(65)
        self.layer_spin.setStyleSheet("font-size: 13px; font-weight: bold; padding: 2px;")
        self.layer_spin.valueChanged.connect(self._on_layer_spin_changed)
        row2.addWidget(self.layer_spin)

        self.btn_next_layer = QPushButton("下一层 ▶")
        self.btn_next_layer.setFixedWidth(75)
        self.btn_next_layer.clicked.connect(self._next_layer)
        row2.addWidget(self.btn_next_layer)

        self.chk_show_gt = QCheckBox("显示 GT 真实缺陷标注")
        self.chk_show_gt.setChecked(True)
        self.chk_show_gt.toggled.connect(self._refresh_all_heatmaps)
        row2.addWidget(self.chk_show_gt)

        row2.addStretch(1)

        btn_fit = QPushButton("适应窗口 (全部视图)")
        btn_fit.clicked.connect(self._fit_all_views)
        row2.addWidget(btn_fit)

        v_layout.addLayout(row2)
        return container

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Tab 1: 图像列表 / Tab 2: 权重与滤波
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # --- SubTab 1: 图像选择 ---
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_layout.setContentsMargins(4, 4, 4, 4)

        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("过滤文件名/路径...")
        self.search_edit.textChanged.connect(self._filter_files)
        search_layout.addWidget(self.search_edit)
        btn_open_img = QPushButton("打开单图...")
        btn_open_img.clicked.connect(self._open_single_image)
        search_layout.addWidget(btn_open_img)
        file_layout.addLayout(search_layout)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["测试图像文件"])
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        file_layout.addWidget(self.file_tree)
        tab_widget.addTab(file_tab, "图像列表")

        # --- SubTab 2: 逐层权重调节 (Real-time Layer Sliders) ---
        layer_tab = QWidget()
        layer_layout = QVBoxLayout(layer_tab)
        layer_layout.setContentsMargins(6, 6, 6, 6)

        preset_box = QGroupBox("权重预设")
        preset_layout = QHBoxLayout(preset_box)
        preset_layout.setContentsMargins(4, 4, 4, 4)

        btn_eq = QPushButton("等权 (1.0)")
        btn_eq.clicked.connect(lambda: self._apply_weight_preset([1.0] * len(self.target_layers)))
        btn_shallow = QPushButton("强化浅层 (L2~L4)")
        btn_shallow.clicked.connect(lambda: self._apply_weight_preset([2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2]))
        btn_deep = QPushButton("强化深层 (L7~L9)")
        btn_deep.clicked.connect(lambda: self._apply_weight_preset([0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]))

        btn_scratch = QPushButton("聚焦细划痕 (L4/L5/L8)")
        btn_scratch.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        btn_scratch.clicked.connect(lambda: self._apply_weight_preset([0.1, 0.2, 2.5, 2.5, 0.0, 0.5, 2.0, 0.5]))

        preset_layout.addWidget(btn_eq)
        preset_layout.addWidget(btn_scratch)
        preset_layout.addWidget(btn_shallow)
        preset_layout.addWidget(btn_deep)
        layer_layout.addWidget(preset_box)

        # Layer Slider rows
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.slider_layout = QVBoxLayout(scroll_content)
        self.slider_layout.setContentsMargins(2, 2, 2, 2)
        self.slider_layout.setSpacing(4)

        self.layer_rows: List[LayerWeightRow] = []
        for idx, layer_num in enumerate(self.target_layers):
            row = LayerWeightRow(idx, f"Layer {layer_num}", 1.0)
            row.weight_changed.connect(self._on_weight_changed)
            self.slider_layout.addWidget(row)
            self.layer_rows.append(row)
        self.slider_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        layer_layout.addWidget(scroll_area, 1)

        tab_widget.addTab(layer_tab, "层权重调节")

        # --- Bottom Group: 后处理平滑与导向滤波控制 (Refinement Controls) ---
        post_group = QGroupBox("后处理边缘精细化控制 (即时生效)")
        post_layout = QGridLayout(post_group)
        post_layout.setContentsMargins(6, 6, 6, 6)
        post_layout.setSpacing(6)

        # 1. Gaussian Filter Switch & Sigma
        self.chk_gaussian = QCheckBox("高斯平滑 (Gaussian)")
        self.chk_gaussian.setChecked(True)
        self.chk_gaussian.toggled.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.chk_gaussian, 0, 0)

        post_layout.addWidget(QLabel("平滑 Sigma:"), 0, 1)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 8.0)
        self.sigma_spin.setSingleStep(0.2)
        self.sigma_spin.setValue(1.5)  # Default to improved finer sigma
        self.sigma_spin.valueChanged.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.sigma_spin, 0, 2)

        # 2. Guided Filter Switch & Radius/Eps
        self.chk_guided = QCheckBox("导向滤波 (Guided Filter 贴合边缘)")
        self.chk_guided.setChecked(False)
        self.chk_guided.setToolTip("利用原图高频梯度边缘将热力图精准约束在划痕边缘内部")
        self.chk_guided.toggled.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.chk_guided, 1, 0)

        post_layout.addWidget(QLabel("导向半径 R:"), 1, 1)
        self.guided_r_spin = QDoubleSpinBox()
        self.guided_r_spin.setRange(1, 50)
        self.guided_r_spin.setSingleStep(1)
        self.guided_r_spin.setValue(15)
        self.guided_r_spin.setDecimals(0)
        self.guided_r_spin.valueChanged.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.guided_r_spin, 1, 2)

        # 3. Threshold Cutoff Floor (Filter normal background noise)
        post_layout.addWidget(QLabel("背景截断阈值:"), 2, 0)
        cutoff_layout = QHBoxLayout()
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 1.0)
        self.cutoff_spin.setSingleStep(0.005)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setValue(0.0)
        self.cutoff_spin.setToolTip("低于此阈值的正常背景区域不着色/视为无异常，高于此值的区域才映射为热力图")
        self.cutoff_spin.valueChanged.connect(self._refresh_all_heatmaps)
        cutoff_layout.addWidget(self.cutoff_spin)

        btn_auto_base = QPushButton("⚡ 自动基线")
        btn_auto_base.setToolTip("自动将阈值设为背景平均分，立即消除背景杂色")
        btn_auto_base.clicked.connect(self._auto_set_baseline)
        cutoff_layout.addWidget(btn_auto_base)
        post_layout.addLayout(cutoff_layout, 2, 1, 1, 2)

        # 4. Contrast Gamma & Mask Background
        post_layout.addWidget(QLabel("对比度 Gamma:"), 3, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setToolTip("<1.0 提高微弱缺陷的鲜艳度；>1.0 压暗背景")
        self.gamma_spin.valueChanged.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.gamma_spin, 3, 1)

        self.chk_mask_bg = QCheckBox("背景透明 (仅高亮超阈值区域)")
        self.chk_mask_bg.setChecked(True)
        self.chk_mask_bg.toggled.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.chk_mask_bg, 3, 2)

        # 5. Colormap & Alpha
        post_layout.addWidget(QLabel("热力色图:"), 4, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.currentTextChanged.connect(self._refresh_all_heatmaps)
        post_layout.addWidget(self.cmap_combo, 4, 1)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.valueChanged.connect(self._refresh_all_heatmaps)
        alpha_layout.addWidget(self.alpha_slider)
        post_layout.addLayout(alpha_layout, 4, 2)

        layout.addWidget(post_group)

        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 3 Canvases Side-by-Side directly as siblings in Splitter (equal height & equal width)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # 1. Left Canvas: Original Image + GT Annotations
        self.canvas_orig = SyncImageCanvas("原图 (Original + GT)", parent=panel)
        self.canvas_orig.pixel_hovered.connect(self._on_pixel_hovered)
        splitter.addWidget(self.canvas_orig)

        # 2. Middle Canvas: Combined Anomaly Map (All layers weighted)
        self.canvas_combined = SyncImageCanvas("融合总异常图 (Combined Anomaly Map)", parent=panel)
        self.canvas_combined.pixel_hovered.connect(self._on_pixel_hovered)
        splitter.addWidget(self.canvas_combined)

        # 3. Right Canvas: Selected Single Layer Heatmap
        self.canvas_layer = SyncImageCanvas("Layer 2 特征热力图", parent=panel)
        self.canvas_layer.pixel_hovered.connect(self._on_pixel_hovered)
        splitter.addWidget(self.canvas_layer)

        splitter.setSizes([450, 450, 450])
        return panel

    def _prev_layer(self) -> None:
        val = int(self.layer_spin.value())
        if val > int(self.layer_spin.minimum()):
            self.layer_spin.setValue(val - 1)

    def _next_layer(self) -> None:
        val = int(self.layer_spin.value())
        if val < int(self.layer_spin.maximum()):
            self.layer_spin.setValue(val + 1)

    def _on_layer_spin_changed(self, val: float) -> None:
        self._refresh_selected_layer_heatmap()

    def _fit_all_views(self) -> None:
        if self.canvas_orig.image is not None:
            self.canvas_orig.fit_to_window()
        if self.canvas_combined.image is not None:
            self.canvas_combined.fit_to_window()
        if self.canvas_layer.image is not None:
            self.canvas_layer.fit_to_window()

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Dinomaly2 模型权重", str(ROOT), "PyTorch Weights (*.pth *.pt)")
        if path:
            self.model_edit.setText(path)
            self._load_model()

    def _browse_data(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据根目录", str(ROOT))
        if path:
            self.data_edit.setText(path)
            self._rebuild_file_list()

    def _open_single_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开单张图像", str(ROOT), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.webp)")
        if path:
            p = Path(path)
            if not self.data_edit.text().strip():
                cand = p.parent.parent
                if cand.name in ("test", "train", "ground_truth"):
                    cand = cand.parent
                if (cand / "test").is_dir() or (cand / "ground_truth").is_dir():
                    self.data_edit.setText(str(cand))
                    self._rebuild_file_list()
            self._process_and_display_image(p)

    def _filter_files(self, text: str) -> None:
        query = text.strip().lower()
        for idx in range(self.file_tree.topLevelItemCount()):
            top = self.file_tree.topLevelItem(idx)
            for c_idx in range(top.childCount()):
                child = top.child(c_idx)
                path_str = child.data(0, Qt.UserRole) or ""
                match = not query or query in path_str.lower()
                child.setHidden(not match)

    def _rebuild_file_list(self) -> None:
        root_text = self.data_edit.text().strip().strip('"')
        if not root_text:
            return
        root_path = Path(root_text).expanduser()
        if not root_path.is_dir():
            return

        self.file_tree.clear()
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(root_path.rglob(f"*{ext}"))
        image_paths = sorted({p for p in image_paths if p.is_file()}, key=lambda p: str(p).lower())

        groups: Dict[str, List[Path]] = {}
        for p in image_paths:
            try:
                rel = p.relative_to(root_path)
                group = rel.parts[0] if len(rel.parts) > 1 else "root"
            except ValueError:
                group = "default"
            groups.setdefault(group, []).append(p)

        for g_name, paths in sorted(groups.items()):
            top_item = QTreeWidgetItem([f"{g_name} ({len(paths)})"])
            for p in paths:
                item = QTreeWidgetItem([p.name])
                item.setData(0, Qt.UserRole, str(p))
                item.setToolTip(0, str(p))
                top_item.addChild(item)
            top_item.setExpanded(True)
            self.file_tree.addTopLevelItem(top_item)

    def _on_file_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        path_str = item.data(0, Qt.UserRole)
        if path_str:
            self._process_and_display_image(Path(path_str))

    def _load_model(self) -> bool:
        model_path_str = self.model_edit.text().strip().strip('"')
        if not model_path_str:
            return False
        model_path = Path(model_path_str)
        if not model_path.is_file():
            QMessageBox.warning(self, "模型加载失败", f"未找到模型文件: {model_path}")
            return False

        try:
            self.statusBar().showMessage("正在加载模型...", 2000)
            backbone = self.backbone_combo.currentText()
            self.model, actual_backbone = load_dinomaly_model(
                model_path=model_path,
                backbone=backbone,
                device=self.device,
            )
            if actual_backbone != backbone:
                idx = self.backbone_combo.findText(actual_backbone)
                if idx >= 0:
                    self.backbone_combo.blockSignals(True)
                    self.backbone_combo.setCurrentIndex(idx)
                    self.backbone_combo.blockSignals(False)
            self.target_layers = self.model.target_layers
            min_l = min(self.target_layers)
            max_l = max(self.target_layers)
            self.layer_spin.setRange(min_l, max_l)
            if self.layer_spin.value() < min_l or self.layer_spin.value() > max_l:
                self.layer_spin.setValue(min_l)
            self.current_model_path = model_path
            self.statusBar().showMessage(f"模型加载成功: {model_path.name} ({actual_backbone})", 3000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "模型初始化失败", str(e))
            self.model = None
            return False

    def _on_load_or_run(self) -> None:
        if self.model is None or self.current_model_path != Path(self.model_edit.text().strip()):
            if not self._load_model():
                return
        if self.current_image_path:
            self._process_and_display_image(self.current_image_path)
        else:
            # Pick first file from tree if available
            if self.file_tree.topLevelItemCount() > 0:
                first_top = self.file_tree.topLevelItem(0)
                if first_top.childCount() > 0:
                    first_p = first_top.child(0).data(0, Qt.UserRole)
                    if first_p:
                        self._process_and_display_image(Path(first_p))

    def _process_and_display_image(self, image_path: Path) -> None:
        """Run Dinomaly2 forward pass, extract per-layer anomaly maps, and display."""
        if not image_path.is_file():
            return
        if self.model is None:
            if not self._load_model():
                return

        self.current_image_path = image_path
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            QMessageBox.warning(self, "读取图像失败", f"无法读取图像: {image_path}")
            return
        self.current_bgr = bgr

        # Load GT annotations (prioritizing ground_truth PNG masks over LabelMe JSON)
        data_root_text = self.data_edit.text().strip().strip('"')
        data_root_path = Path(data_root_text).expanduser() if data_root_text else None
        self.current_shapes, self.current_gt_mask = load_ground_truth_shapes(image_path, data_root=data_root_path)

        # Prepare PyTorch image tensor
        size = int(self.size_spin.value())
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0).to(self.device)

        orig_shape = bgr.shape[:2]

        # Model Inference for 8 individual target layers
        with torch.no_grad():
            x = tensor
            if hasattr(self.model.encoder, 'prepare_tokens_with_masks'):
                x_tokens = self.model.encoder.prepare_tokens_with_masks(x)
            else:
                x_tokens = self.model.encoder.prepare_tokens(x)
            en_list = []
            for i, blk in enumerate(self.model.encoder.blocks):
                if i <= self.model.target_layers[-1]:
                    x_tokens = blk(x_tokens)
                else:
                    continue
                if i in self.model.target_layers:
                    en_list.append(x_tokens)

            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.model.encoder.num_register_tokens))

            # Bottleneck
            x_bn = self.model.fuse_feature([en_list[idx] for idx in self.model.fuse_layer_bottleneck]).detach()
            for blk in self.model.bottleneck:
                x_bn = blk(x_bn)

            # Decoder
            de_list = []
            curr_de = x_bn
            for blk in self.model.decoder:
                curr_de = blk(curr_de)
                de_list.append(curr_de)
            de_list = de_list[::-1]

            # Compute individual layer anomaly maps upscaled to original resolution
            layer_map_arrays = []
            for i, layer_idx in enumerate(self.model.target_layers):
                en_i = en_list[i]
                de_i = de_list[i]
                de_patch = de_i[:, 1 + self.model.encoder.num_register_tokens:, :]
                if self.model.context_aware_recenter:
                    en_patch = en_i[:, 1 + self.model.encoder.num_register_tokens:, :] - en_i[:, :1, :]
                    en_patch = F.layer_norm(en_patch, (en_patch.shape[-1],), eps=1e-8)
                else:
                    en_patch = en_i[:, 1 + self.model.encoder.num_register_tokens:, :]

                en_2d = en_patch.permute(0, 2, 1).reshape(x.shape[0], -1, side, side).contiguous()
                de_2d = de_patch.permute(0, 2, 1).reshape(x.shape[0], -1, side, side).contiguous()

                a_map = 1 - F.cosine_similarity(en_2d, de_2d)
                a_map = torch.unsqueeze(a_map, dim=1)
                a_map_up = F.interpolate(a_map, size=orig_shape, mode='bilinear', align_corners=False)
                layer_map_arrays.append(a_map_up[0, 0].cpu().numpy().astype(np.float32))

        self.raw_layer_maps = layer_map_arrays

        # Update layer slider layout if target layers changed
        if len(self.raw_layer_maps) != len(self.layer_rows):
            # rebuild rows
            for r in self.layer_rows:
                r.deleteLater()
            self.layer_rows.clear()
            for idx in range(len(self.raw_layer_maps)):
                layer_num = self.target_layers[idx] if idx < len(self.target_layers) else idx
                row = LayerWeightRow(idx, f"Layer {layer_num}", 1.0)
                row.weight_changed.connect(self._on_weight_changed)
                self.slider_layout.addWidget(row)
                self.layer_rows.append(row)

        self._refresh_all_heatmaps()
        self._fit_all_views()

    def _on_weight_changed(self, layer_idx: int, weight: float) -> None:
        """Triggered in real-time when user moves any layer slider."""
        self._refresh_combined_heatmap()

    def _apply_weight_preset(self, weights: List[float]) -> None:
        for idx, w in enumerate(weights):
            if idx < len(self.layer_rows):
                self.layer_rows[idx].set_weight(w)
        self._refresh_combined_heatmap()

    def _filter_score_map(self, raw_score: np.ndarray) -> np.ndarray:
        """Apply Gaussian filtering and optional Guided filtering to a score map."""
        refined = raw_score.copy()

        # 1. Gaussian Filter
        if self.chk_gaussian.isChecked():
            sigma = float(self.sigma_spin.value())
            # Ensure kernel size is odd and at least 3*sigma
            ksize = int(max(3, int(sigma * 3) // 2 * 2 + 1))
            refined = cv2.GaussianBlur(refined, (ksize, ksize), sigma)

        # 2. Guided Filter (using original RGB guidance)
        if self.chk_guided.isChecked() and self.current_bgr is not None:
            radius = int(self.guided_r_spin.value())
            refined = refine_anomaly_map_guided(self.current_bgr, refined, radius=radius, eps=1e-3)

        return np.nan_to_num(refined, nan=0.0)

    def _auto_set_baseline(self) -> None:
        """Automatically estimate background baseline threshold (mean + 0.3*std)."""
        target_map = self.combined_map if self.combined_map is not None else (self.raw_layer_maps[0] if self.raw_layer_maps else None)
        if target_map is None:
            return
        m = float(np.nanmean(target_map))
        s = float(np.nanstd(target_map))
        auto_thresh = round(m + 0.3 * s, 3)
        self.cutoff_spin.setValue(auto_thresh)

    def _blend_heatmap(self, bgr_img: np.ndarray, score_map: np.ndarray) -> np.ndarray:
        """Overlay colormapped score map onto the base BGR image with threshold filtering & gamma."""
        s_min, s_max = float(np.nanmin(score_map)), float(np.nanmax(score_map))
        cutoff = float(self.cutoff_spin.value()) if hasattr(self, "cutoff_spin") else 0.0
        gamma = float(self.gamma_spin.value()) if hasattr(self, "gamma_spin") else 1.0
        mask_bg = self.chk_mask_bg.isChecked() if hasattr(self, "chk_mask_bg") else False

        if s_max - s_min < 1e-12:
            s_norm = np.zeros_like(score_map, dtype=np.uint8)
            mask_active = np.zeros_like(score_map, dtype=bool)
        else:
            if cutoff > 0.0:
                effective_floor = cutoff
                effective_ceil = max(cutoff + 1e-4, s_max)
                rel_score = np.clip((score_map - effective_floor) / (effective_ceil - effective_floor), 0.0, 1.0)
                if gamma != 1.0:
                    rel_score = np.power(rel_score, gamma)
                s_norm = (rel_score * 255.0).astype(np.uint8)
                mask_active = (score_map >= cutoff)
            else:
                rel_score = np.clip((score_map - s_min) / (s_max - s_min), 0.0, 1.0)
                if gamma != 1.0:
                    rel_score = np.power(rel_score, gamma)
                s_norm = (rel_score * 255.0).astype(np.uint8)
                mask_active = np.ones_like(score_map, dtype=bool)

        cmap_name = self.cmap_combo.currentText()
        cmap_code = COLORMAPS.get(cmap_name, cv2.COLORMAP_JET)
        heat = cv2.applyColorMap(s_norm, cmap_code)

        alpha = self.alpha_slider.value() / 100.0
        if cutoff > 0.0 and mask_bg:
            blended = bgr_img.copy()
            if np.any(mask_active):
                blended[mask_active] = cv2.addWeighted(
                    bgr_img[mask_active], 1.0 - alpha, heat[mask_active], alpha, 0
                )
        else:
            blended = cv2.addWeighted(bgr_img, 1.0 - alpha, heat, alpha, 0)
        return blended

    def _refresh_combined_heatmap(self) -> None:
        """Recalculate combined score map using current layer weights and update view."""
        if not self.raw_layer_maps or self.current_bgr is None:
            return

        weights = [row.get_weight() for row in self.layer_rows]
        total_w = sum(weights)
        if total_w <= 1e-8:
            weights = [1.0] * len(weights)
            total_w = sum(weights)

        # Weighted combination of raw layer maps
        combined_raw = np.zeros_like(self.raw_layer_maps[0])
        for w, l_map in zip(weights, self.raw_layer_maps):
            combined_raw += (w / total_w) * l_map

        self.combined_map = self._filter_score_map(combined_raw)
        blended = self._blend_heatmap(self.current_bgr, self.combined_map)

        max_val = float(np.nanmax(self.combined_map))
        mean_val = float(np.nanmean(self.combined_map))
        shapes_to_draw = self.current_shapes if (hasattr(self, "chk_show_gt") and self.chk_show_gt.isChecked()) else []
        self.canvas_combined.set_content(
            blended,
            self.combined_map,
            shapes_to_draw,
            title_suffix=f"[Max: {max_val:.4f}, Mean: {mean_val:.4f}]",
        )

    def _refresh_selected_layer_heatmap(self) -> None:
        """Render the currently selected single layer in the 3rd canvas."""
        if self.current_bgr is None or not self.raw_layer_maps:
            return

        layer_num = int(self.layer_spin.value())
        try:
            layer_idx = self.target_layers.index(layer_num)
        except ValueError:
            layer_idx = 0

        if layer_idx >= len(self.raw_layer_maps):
            layer_idx = 0

        raw_l_map = self.raw_layer_maps[layer_idx]
        filtered_l_map = self._filter_score_map(raw_l_map)
        blended_l = self._blend_heatmap(self.current_bgr, filtered_l_map)

        max_val = float(np.nanmax(filtered_l_map))
        mean_val = float(np.nanmean(filtered_l_map))
        self.canvas_layer.title = f"Layer {layer_num} 特征热力图"
        shapes_to_draw = self.current_shapes if (hasattr(self, "chk_show_gt") and self.chk_show_gt.isChecked()) else []
        self.canvas_layer.set_content(
            blended_l,
            filtered_l_map,
            shapes_to_draw,
            title_suffix=f"[Max: {max_val:.4f}, Mean: {mean_val:.4f}]",
        )

    def _refresh_all_heatmaps(self) -> None:
        """Re-render Original, Combined, and the currently selected Layer canvas."""
        if self.current_bgr is None:
            return

        shapes_to_draw = self.current_shapes if (hasattr(self, "chk_show_gt") and self.chk_show_gt.isChecked()) else []

        # 1. Original Canvas
        self.canvas_orig.set_content(self.current_bgr, None, shapes_to_draw)

        # 2. Combined Canvas
        self._refresh_combined_heatmap()

        # 3. Selected Layer Canvas
        self._refresh_selected_layer_heatmap()

    def _on_pixel_hovered(self, x: int, y: int) -> None:
        """Display real-time probe values across all layers when cursor hovers on any image."""
        if self.current_bgr is None or self.combined_map is None:
            return
        if not (0 <= y < self.current_bgr.shape[0] and 0 <= x < self.current_bgr.shape[1]):
            return

        bgr_val = self.current_bgr[y, x]
        comb_score = self.combined_map[y, x]
        cur_layer_num = int(self.layer_spin.value())

        # Layer probe values
        layer_scores = []
        cur_layer_score = None
        for idx, layer_num in enumerate(self.target_layers):
            if idx < len(self.raw_layer_maps):
                l_score = float(self.raw_layer_maps[idx][y, x])
                layer_scores.append(f"L{layer_num}: {l_score:.4f}")
                if layer_num == cur_layer_num:
                    cur_layer_score = l_score

        layers_text = " | ".join(layer_scores)
        cur_text = f"L{cur_layer_num}={cur_layer_score:.4f}" if cur_layer_score is not None else ""
        probe_msg = (
            f"📍 坐标: (X={x}, Y={y}) | 像素RGB: ({bgr_val[2]}, {bgr_val[1]}, {bgr_val[0]}) | "
            f"⚡ 融合分: {comb_score:.4f} | 🔍 当前查看层({cur_text}) || 各层明细: {layers_text}"
        )
        self.probe_label.setText(probe_msg)


def main():
    parser = argparse.ArgumentParser(description="Dinomaly2 Layer Heatmap Inspection GUI")
    parser.add_argument("--model", type=str, default="/data/wt/trainlogs/leishi_026/Dinomaly/mask_constraint_with_aug/20260728161947/model.pth", help="Path to Dinomaly2 model.pth")
    parser.add_argument("--data_root", type=str, default="/data/wt/ramdisk/leishi_026", help="Path to images directory")
    parser.add_argument("--backbone", type=str, default="dinov2reg_vit_small_14", help="Backbone architecture")
    parser.add_argument("--image_size", type=int, default=672, help="Image resize dimension")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark Theme Palette
    from PySide6.QtGui import QPalette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 36))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 225))
    dark_palette.setColor(QPalette.Base, QColor(20, 20, 24))
    dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 42))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 225))
    dark_palette.setColor(QPalette.Button, QColor(42, 42, 50))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 225))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)

    window = LayerInspectionWindow(
        default_model=args.model if Path(args.model).is_file() else None,
        default_data=args.data_root if Path(args.data_root).is_dir() else None,
        default_backbone=args.backbone,
        default_size=args.image_size,
        gpu=args.gpu,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
