"""PySide6 GUI for querying Dinomaly2 ROI feature libraries.

The three canvases display the direct good-threshold Mask, selectable
candidate/manual query ROIs, and the matched source image with its stored ROI.
A query runs in a separate Python process so the UI stays responsive while
Dinomaly2 and FAISS are loading/searching.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ImageCanvas(QWidget):
    """Image canvas with image-coordinate drawing and optional ROI overlay."""

    shapes_changed = Signal()
    candidate_changed = Signal()

    def __init__(self, editable: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.editable = editable
        self.mode = "rectangle"
        self.image: Optional[QImage] = None
        self.image_path: Optional[Path] = None
        self.candidate_regions: List[Dict[str, Any]] = []
        self.selected_candidate_index: Optional[int] = None
        self.shapes: List[Dict[str, Any]] = []
        self.current_points: List[QPointF] = []
        self.drag_start: Optional[QPointF] = None
        self.drag_end: Optional[QPointF] = None
        self.overlay_bbox: Optional[Tuple[float, float, float, float]] = None
        self.overlay_text = ""
        self.setMinimumSize(420, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self):
        return self.minimumSizeHint()

    def set_image(self, image_path: Path) -> None:
        image = QImage(str(image_path))
        if image.isNull():
            raise OSError(f"无法读取图像：{image_path}")
        self.image = image.convertToFormat(QImage.Format.Format_RGB32)
        self.image_path = Path(image_path)
        self.clear_shapes(emit=False)
        self.clear_candidate_regions(emit=False)
        self.clear_overlay()
        self.update()

    def clear_image(self) -> None:
        self.image = None
        self.image_path = None
        self.clear_shapes(emit=False)
        self.clear_candidate_regions(emit=False)
        self.clear_overlay()
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"candidate", "rectangle", "polygon"}:
            raise ValueError(f"不支持的绘制类型：{mode}")
        self.mode = mode
        self.current_points.clear()
        self.drag_start = None
        self.drag_end = None
        self.update()

    def set_candidate_regions(
        self,
        regions: Sequence[Mapping[str, Any]],
        emit: bool = True,
    ) -> None:
        self.candidate_regions = []
        for region in regions:
            mask = np.asarray(region.get("mask"), dtype=bool)
            bbox = region.get("bbox", ())
            if mask.ndim != 2 or len(bbox) != 4:
                continue
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                points = [
                    QPointF(float(point[0][0]), float(point[0][1]))
                    for point in contour
                ]
            else:
                points = []
            self.candidate_regions.append(
                {
                    "region_id": int(region.get("region_id", len(self.candidate_regions) + 1)),
                    "mask": mask,
                    "bbox": tuple(float(value) for value in bbox),
                    "area": int(region.get("area", int(mask.sum()))),
                    "points": points,
                }
            )
        self.selected_candidate_index = None
        if emit:
            self.candidate_changed.emit()
        self.update()

    def clear_candidate_regions(self, emit: bool = True) -> None:
        self.candidate_regions.clear()
        self.selected_candidate_index = None
        if emit:
            self.candidate_changed.emit()
        self.update()

    def select_candidate(self, index: int) -> None:
        if 0 <= int(index) < len(self.candidate_regions):
            self.selected_candidate_index = int(index)
        else:
            self.selected_candidate_index = None
        self.candidate_changed.emit()
        self.update()

    def selected_candidate_mask(self) -> Optional[np.ndarray]:
        if self.selected_candidate_index is None:
            return None
        if not 0 <= self.selected_candidate_index < len(self.candidate_regions):
            return None
        return np.asarray(
            self.candidate_regions[self.selected_candidate_index]["mask"],
            dtype=np.uint8,
        )

    def candidate_at(self, point: QPointF) -> Optional[int]:
        """Return the smallest candidate containing an image-coordinate point."""

        if self.image is None:
            return None
        x = int(round(point.x()))
        y = int(round(point.y()))
        hits = []
        for index, region in enumerate(self.candidate_regions):
            mask = region["mask"]
            contains_mask = (
                0 <= y < mask.shape[0]
                and 0 <= x < mask.shape[1]
                and bool(mask[y, x])
            )
            if contains_mask:
                hits.append((int(region.get("area", 0)), index))
        if not hits:
            return None
        return min(hits)[1]

    def clear_shapes(self, emit: bool = True) -> None:
        self.shapes.clear()
        self.current_points.clear()
        self.drag_start = None
        self.drag_end = None
        if emit:
            self.shapes_changed.emit()
        self.update()

    def undo(self) -> None:
        if self.current_points:
            self.current_points.pop()
        elif self.shapes:
            self.shapes.pop()
        self.shapes_changed.emit()
        self.update()

    def finish_polygon(self) -> None:
        if len(self.current_points) >= 3:
            self.shapes.append(
                {
                    "type": "polygon",
                    "points": [QPointF(point) for point in self.current_points],
                }
            )
            self.shapes_changed.emit()
        self.current_points.clear()
        self.update()

    def set_overlay_bbox(
        self,
        bbox: Optional[Sequence[float]],
        text: str = "",
    ) -> None:
        self.overlay_bbox = (
            tuple(float(value) for value in bbox)
            if bbox is not None and len(bbox) == 4
            else None
        )
        self.overlay_text = text
        self.update()

    def clear_overlay(self) -> None:
        self.overlay_bbox = None
        self.overlay_text = ""
        self.update()

    def image_rect(self) -> QRectF:
        if self.image is None or self.image.width() < 1 or self.image.height() < 1:
            return QRectF()
        margin = 12.0
        available_width = max(float(self.width()) - 2.0 * margin, 1.0)
        available_height = max(float(self.height()) - 2.0 * margin, 1.0)
        scale = min(
            available_width / float(self.image.width()),
            available_height / float(self.image.height()),
        )
        width = float(self.image.width()) * scale
        height = float(self.image.height()) * scale
        return QRectF(
            (float(self.width()) - width) / 2.0,
            (float(self.height()) - height) / 2.0,
            width,
            height,
        )

    def image_to_widget(self, point: QPointF) -> QPointF:
        image_rect = self.image_rect()
        if image_rect.isNull() or self.image is None:
            return QPointF()
        scale_x = image_rect.width() / float(self.image.width())
        scale_y = image_rect.height() / float(self.image.height())
        return QPointF(
            image_rect.left() + point.x() * scale_x,
            image_rect.top() + point.y() * scale_y,
        )

    def widget_to_image(self, point: QPointF) -> Optional[QPointF]:
        image_rect = self.image_rect()
        if self.image is None or image_rect.isNull() or not image_rect.contains(point):
            return None
        x = (point.x() - image_rect.left()) * self.image.width() / image_rect.width()
        y = (point.y() - image_rect.top()) * self.image.height() / image_rect.height()
        return QPointF(
            max(0.0, min(float(self.image.width() - 1), x)),
            max(0.0, min(float(self.image.height() - 1), y)),
        )

    def _draw_shape(self, painter: QPainter, shape: Dict[str, Any], color: QColor) -> None:
        painter.setPen(QPen(color, 2.0))
        if shape["type"] == "rectangle":
            start, end = shape["points"]
            rect = QRectF(
                self.image_to_widget(start),
                self.image_to_widget(end),
            ).normalized()
            painter.drawRect(rect)
        else:
            polygon = QPolygonF(
                [self.image_to_widget(point) for point in shape["points"]]
            )
            painter.drawPolygon(polygon)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202124"))
        if self.image is None:
            painter.setPen(QColor("#d7d7d7"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "请打开输入图像",
            )
            painter.end()
            return

        image_rect = self.image_rect()
        painter.drawImage(image_rect, self.image)
        for index, candidate in enumerate(self.candidate_regions):
            selected = index == self.selected_candidate_index
            color = QColor("#ffeb3b") if selected else QColor("#00bcd4")
            painter.setPen(QPen(color, 3.0 if selected else 2.0))
            points = candidate.get("points", [])
            polygon = QPolygonF(
                [self.image_to_widget(point) for point in points]
            )
            if len(points) >= 3:
                if selected:
                    painter.setBrush(QColor(255, 235, 59, 55))
                else:
                    painter.setBrush(QColor(0, 188, 212, 30))
                painter.drawPolygon(polygon)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                label_point = polygon.boundingRect().topLeft()
            else:
                x1, y1, x2, y2 = candidate["bbox"]
                top_left = self.image_to_widget(QPointF(x1, y1))
                bottom_right = self.image_to_widget(QPointF(x2, y2))
                candidate_rect = QRectF(top_left, bottom_right).normalized()
                painter.drawRect(candidate_rect)
                label_point = candidate_rect.topLeft()
            label = f"R{index + 1}"
            text_rect = QRectF(
                label_point.x(),
                max(0.0, label_point.y() - 22.0),
                70.0,
                20.0,
            )
            painter.fillRect(text_rect, QColor(0, 0, 0, 180))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, label)
        for shape in self.shapes:
            self._draw_shape(painter, shape, QColor("#00e676"))

        if self.current_points:
            painter.setPen(QPen(QColor("#ffeb3b"), 2.0))
            current_polygon = QPolygonF(
                [self.image_to_widget(point) for point in self.current_points]
            )
            painter.drawPolyline(current_polygon)
            for point in current_polygon:
                painter.drawEllipse(point, 3.0, 3.0)
        if self.drag_start is not None and self.drag_end is not None:
            self._draw_shape(
                painter,
                {"type": "rectangle", "points": [self.drag_start, self.drag_end]},
                QColor("#ffeb3b"),
            )

        if self.overlay_bbox is not None:
            x1, y1, x2, y2 = self.overlay_bbox
            top_left = self.image_to_widget(QPointF(x1, y1))
            bottom_right = self.image_to_widget(QPointF(x2, y2))
            painter.setPen(QPen(QColor("#ff1744"), 3.0))
            painter.drawRect(QRectF(top_left, bottom_right).normalized())
            if self.overlay_text:
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                text_rect = QRectF(top_left.x(), top_left.y() - 24, 420, 22)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                painter.setPen(QColor("#ffffff"))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, self.overlay_text)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if not self.editable:
            return
        if self.mode == "candidate":
            if event.button() == Qt.MouseButton.LeftButton:
                point = self.widget_to_image(event.position())
                if point is not None:
                    candidate_index = self.candidate_at(point)
                    if candidate_index is not None:
                        self.select_candidate(candidate_index)
                    else:
                        self.select_candidate(-1)
            return
        if (
            self.mode == "polygon"
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.finish_polygon()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.widget_to_image(event.position())
        if point is None:
            return
        self.setFocus()
        if self.mode == "rectangle":
            self.drag_start = point
            self.drag_end = point
        else:
            self.current_points.append(point)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self.editable or self.mode != "rectangle" or self.drag_start is None:
            return
        point = self.widget_to_image(event.position())
        if point is not None:
            self.drag_end = point
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if (
            not self.editable
            or self.mode != "rectangle"
            or self.drag_start is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        point = self.widget_to_image(event.position()) or self.drag_end
        if point is not None:
            self.drag_end = point
        if self.drag_end is not None:
            rect = QRectF(self.drag_start, self.drag_end).normalized()
            if rect.width() >= 2.0 and rect.height() >= 2.0:
                self.shapes.append(
                    {
                        "type": "rectangle",
                        "points": [QPointF(rect.topLeft()), QPointF(rect.bottomRight())],
                    }
                )
                self.shapes_changed.emit()
        self.drag_start = None
        self.drag_end = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.editable and self.mode == "polygon" and event.button() == Qt.MouseButton.LeftButton:
            point = self.widget_to_image(event.position())
            if point is not None:
                if not self.current_points or self.current_points[-1] != point:
                    self.current_points.append(point)
                self.finish_polygon()

    def keyPressEvent(self, event) -> None:
        if self.editable and event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.finish_polygon()
        elif self.editable and event.key() == Qt.Key.Key_Escape:
            self.current_points.clear()
            self.update()
        else:
            super().keyPressEvent(event)

    def mask_array(self) -> np.ndarray:
        if self.image is None:
            raise RuntimeError("尚未打开输入图像")
        mask = np.zeros((self.image.height(), self.image.width()), dtype=np.uint8)
        for shape in self.shapes:
            if shape["type"] == "rectangle":
                start, end = shape["points"]
                x1, y1 = int(round(min(start.x(), end.x()))), int(round(min(start.y(), end.y())))
                x2, y2 = int(round(max(start.x(), end.x()))), int(round(max(start.y(), end.y())))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            else:
                points = np.asarray(
                    [[int(round(point.x())), int(round(point.y()))] for point in shape["points"]],
                    dtype=np.int32,
                )
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 1)
        return mask


class MainWindow(QMainWindow):
    def __init__(self, args, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.args = args
        self.process: Optional[QProcess] = None
        self.current_run_dir: Optional[Path] = None
        self.results: List[Dict[str, Any]] = []
        self.setWindowTitle("Dinomaly2 ROI 特征库反查")
        self.resize(1800, 900)

        self.open_button = QPushButton("打开输入图像")
        self.mode_combo = QComboBox()
        if self._artifact_root("candidate_regions") is not None:
            self.mode_combo.addItem("候选区域", "candidate")
        self.mode_combo.addItem("矩形", "rectangle")
        self.mode_combo.addItem("多边形", "polygon")
        self.finish_button = QPushButton("完成多边形")
        self.undo_button = QPushButton("撤销")
        self.clear_button = QPushButton("清空区域")
        self.query_button = QPushButton("查询特征库")
        self.status_label = QLabel("请打开图像，然后在左侧绘制异常区域")
        self.status_label.setWordWrap(True)

        self.left_canvas = ImageCanvas(editable=True)
        self.right_canvas = ImageCanvas(editable=False)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["图像路径", "距离"])
        self.result_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.result_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.result_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.result_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.result_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.result_table.setMinimumHeight(150)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(QLabel("中间区域："))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.finish_button)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        controls.addWidget(self.query_button)

        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.addWidget(QLabel("原始 Dinomaly2 区域（score ≥ good_threshold）"))
        self.raw_canvas = ImageCanvas(editable=False)
        raw_layout.addWidget(self.raw_canvas, 1)

        candidate_panel = QWidget()
        candidate_layout = QVBoxLayout(candidate_panel)
        candidate_layout.addWidget(QLabel("候选区域 / 手动画 ROI（单选后查询）"))
        candidate_layout.addWidget(self.left_canvas, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("最近邻原图 / 对应 ROI"))
        right_layout.addWidget(self.right_canvas, 1)
        right_layout.addWidget(self.result_table)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(raw_panel)
        splitter.addWidget(candidate_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)

        self.open_button.clicked.connect(self.open_image)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        self.finish_button.clicked.connect(self.left_canvas.finish_polygon)
        self.undo_button.clicked.connect(self.left_canvas.undo)
        self.clear_button.clicked.connect(self.clear_query_selection)
        self.query_button.clicked.connect(self.start_query)
        self.left_canvas.shapes_changed.connect(self.update_controls)
        self.left_canvas.candidate_changed.connect(self.candidate_selection_changed)
        self.result_table.currentCellChanged.connect(self._result_row_changed)
        self.query_button.setEnabled(False)

        if args.input:
            self.load_input_image(Path(args.input).expanduser())

    def change_mode(self, _index: int) -> None:
        self.left_canvas.set_mode(self.mode_combo.currentData())
        if self.left_canvas.mode == "candidate":
            self.status_label.setText(
                "候选区域：左键单击一个候选多边形进行选择，然后点击‘查询特征库’。"
            )
        else:
            self.status_label.setText(
                "矩形：按住左键拖拽；多边形：左键依次点击顶点，右键、双击或点击‘完成多边形’结束。"
            )
        self.update_controls()

    def load_input_image(self, image_path: Path) -> None:
        try:
            self.left_canvas.set_image(image_path)
            self.raw_canvas.set_image(image_path)
            self.load_raw_regions(image_path)
            candidate_path = self.load_candidate_regions(image_path)
            self.current_run_dir = None
            self.right_canvas.clear_image()
            self.result_table.setRowCount(0)
            self.results.clear()
            if candidate_path is not None:
                candidate_index = self.mode_combo.findData("candidate")
                if candidate_index >= 0:
                    self.mode_combo.setCurrentIndex(candidate_index)
                self.status_label.setText(
                    f"已打开：{image_path}；原始区域 {len(self.raw_canvas.candidate_regions)} 个，"
                    f"候选区域 {len(self.left_canvas.candidate_regions)} 个；"
                    "请单击一个多边形后查询。"
                )
            else:
                rectangle_index = self.mode_combo.findData("rectangle")
                if rectangle_index >= 0:
                    self.mode_combo.setCurrentIndex(rectangle_index)
                if self._artifact_root("candidate_regions") is not None:
                    self.status_label.setText(
                        f"已打开：{image_path}；原始区域 {len(self.raw_canvas.candidate_regions)} 个，"
                        "未找到对应候选 Mask，可切换为矩形或多边形手动画 ROI。"
                    )
                else:
                    self.status_label.setText(
                        f"已打开：{image_path}；原始区域 "
                        f"{len(self.raw_canvas.candidate_regions)} 个。"
                    )
            self.update_controls()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "打开失败", str(error))

    def _artifact_root(self, artifact_name: str) -> Optional[Path]:
        """Return an explicit artifact root or one under --prediction_dir."""

        configured = getattr(self.args, artifact_name, None)
        if configured:
            return Path(configured).expanduser()
        prediction_dir = getattr(self.args, "prediction_dir", None)
        if prediction_dir:
            return Path(prediction_dir).expanduser() / artifact_name
        return None

    def _mask_path(self, root: Path, image_path: Path) -> Optional[Path]:
        """Resolve one predictor Mask using the input's relative path."""

        if root.is_file():
            return root
        if not root.is_dir():
            raise FileNotFoundError(
                f"Mask 目录不存在或不是文件：{root}"
            )

        candidates: List[Path] = []
        data_root_text = self.args.data_root
        if data_root_text:
            data_root = Path(data_root_text).expanduser().resolve()
            try:
                relative = image_path.resolve().relative_to(data_root)
            except ValueError:
                relative = None
            if relative is not None:
                candidates.extend(
                    [
                        root / relative.with_suffix(".png"),
                        root / relative,
                    ]
                )

        candidates.extend(
            [
                root / image_path.name,
                root / image_path.with_suffix(".png").name,
                root / f"{image_path.stem}.png",
            ]
        )
        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if str(candidate).casefold() in seen:
                continue
            seen.add(str(candidate).casefold())
            if candidate.is_file():
                return candidate

        # This fallback is useful when data_root was omitted and the output
        # directory contains a single matching relative image name.
        matches = [
            path
            for path in root.rglob(f"{image_path.stem}.png")
            if path.is_file()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _mask_components(self, mask_path: Path) -> List[Dict[str, Any]]:
        """Read a predictor Mask and return its connected components."""

        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise OSError(f"无法读取 Mask：{mask_path}")
        if self.left_canvas.image is None:
            raise RuntimeError("尚未打开输入图像")
        expected_shape = (
            self.left_canvas.image.height(),
            self.left_canvas.image.width(),
        )
        if raw_mask.shape != expected_shape:
            raw_mask = cv2.resize(
                raw_mask,
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        binary_mask = (raw_mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
        regions: List[Dict[str, Any]] = []
        for component_id in range(1, count):
            component_mask = labels == component_id
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            ys, xs = np.where(component_mask)
            if not len(xs):
                continue
            regions.append(
                {
                    "region_id": len(regions) + 1,
                    "mask": component_mask,
                    "area": area,
                    "bbox": (
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ),
                }
            )
        regions.sort(key=lambda region: (-region["area"], region["region_id"]))
        for index, region in enumerate(regions, start=1):
            region["region_id"] = index
        return regions

    def load_raw_regions(self, image_path: Path) -> Optional[Path]:
        """Load the direct good-threshold Mask into the left canvas."""

        self.raw_canvas.clear_candidate_regions(emit=False)
        root = self._artifact_root("raw_regions")
        if root is None:
            return None
        mask_path = self._mask_path(root, image_path)
        if mask_path is None:
            return None
        self.raw_canvas.set_candidate_regions(
            self._mask_components(mask_path),
            emit=False,
        )
        return mask_path

    def load_candidate_regions(self, image_path: Path) -> Optional[Path]:
        """Load one mask and split it into selectable connected components."""

        self.left_canvas.clear_candidate_regions(emit=False)
        root = self._artifact_root("candidate_regions")
        if root is None:
            return None
        mask_path = self._mask_path(root, image_path)
        if mask_path is None:
            return None
        self.left_canvas.set_candidate_regions(
            self._mask_components(mask_path),
            emit=False,
        )
        return mask_path

    def candidate_selection_changed(self) -> None:
        self.update_controls()
        index = self.left_canvas.selected_candidate_index
        if index is None or index >= len(self.left_canvas.candidate_regions):
            if self.left_canvas.mode == "candidate":
                self.status_label.setText("当前未选择候选区域，请单击一个候选多边形。")
            return
        region = self.left_canvas.candidate_regions[index]
        self.status_label.setText(
            f"已选择候选区域 R{index + 1}（面积 {region['area']}），请点击‘查询特征库’。"
        )

    def clear_query_selection(self) -> None:
        """Clear the active manual ROI or the single candidate selection."""

        self.left_canvas.clear_shapes(emit=False)
        if self.left_canvas.mode == "candidate":
            self.left_canvas.select_candidate(-1)
        else:
            self.left_canvas.shapes_changed.emit()
        self.status_label.setText("已清空当前查询区域，请重新选择或绘制 ROI。")

    def open_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if image_path:
            self.load_input_image(Path(image_path))

    def update_controls(self) -> None:
        has_image = self.left_canvas.image is not None
        if self.left_canvas.mode == "candidate":
            selected_mask = self.left_canvas.selected_candidate_mask()
            has_roi = selected_mask is not None and bool(np.any(selected_mask))
        else:
            # An unfinished polygon is not a query ROI until it is completed.
            has_roi = bool(self.left_canvas.shapes)
        self.query_button.setEnabled(has_image and has_roi and self.process is None)

    def _query_arguments(self, mask_path: Path, run_dir: Path) -> List[str]:
        query_script = Path(__file__).with_name("query_feature_library.py")
        arguments = [
            "-u",
            str(query_script),
            "--model",
            str(Path(self.args.model).expanduser()),
            "--input",
            str(self.left_canvas.image_path.resolve()),
            "--region_mask",
            str(mask_path),
            "--good_library",
            str(Path(self.args.good_library).expanduser()),
            "--anomaly_library",
            str(Path(self.args.anomaly_library).expanduser()),
            "--top_k",
            str(self.args.top_k),
            "--output_dir",
            str(run_dir),
            "--backbone",
            self.args.backbone,
            "--image_size",
            str(self.args.image_size),
            "--crop_size",
            str(self.args.crop_size),
            "--dropout",
            str(self.args.dropout),
            "--la",
            str(self.args.la),
            "--lc",
            str(self.args.lc),
            "--cr",
            str(self.args.cr),
            "--feature_merge",
            self.args.feature_merge,
            "--roi_size",
            str(self.args.roi_size),
            "--gpu",
            str(self.args.gpu),
        ]
        if self.args.faiss_on_gpu:
            arguments.append("--faiss_on_gpu")
        return arguments

    def start_query(self) -> None:
        if self.process is not None:
            return
        if self.left_canvas.image_path is None:
            return
        if self.left_canvas.mode == "candidate":
            selected_mask = self.left_canvas.selected_candidate_mask()
            if selected_mask is None or not np.any(selected_mask):
                QMessageBox.warning(
                    self,
                    "查询失败",
                    "请先在左侧单击选择一个候选多边形。当前仅支持单选。",
                )
                return
            mask = np.asarray(selected_mask, dtype=np.uint8)
        else:
            try:
                mask = self.left_canvas.mask_array()
            except (RuntimeError, ValueError) as error:
                QMessageBox.warning(self, "查询失败", str(error))
                return
        if not np.any(mask):
            QMessageBox.warning(self, "查询失败", "请先选择或绘制一个有效 ROI。")
            return

        output_root = Path(self.args.output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = output_root / f"query_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        mask_path = run_dir / "query_mask.png"
        if not cv2.imwrite(str(mask_path), mask * 255):
            QMessageBox.critical(self, "查询失败", f"无法保存查询 Mask：{mask_path}")
            return
        self.current_run_dir = run_dir

        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(Path(__file__).parent))
        process.setProgram(sys.executable)
        process.setArguments(self._query_arguments(mask_path, run_dir))
        process.readyReadStandardOutput.connect(self.read_process_output)
        process.readyReadStandardError.connect(self.read_process_error)
        process.finished.connect(self.query_finished)
        self.query_button.setEnabled(False)
        self.status_label.setText("正在提取特征并检索，请稍候……")
        process.start()

    def read_process_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.status_label.setText(lines[-1])

    def read_process_error(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.status_label.setText(lines[-1])

    def query_finished(self, exit_code: int, _exit_status) -> None:
        process = self.process
        self.process = None
        self.update_controls()
        if process is None:
            return
        if exit_code != 0 or self.current_run_dir is None:
            error = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
            QMessageBox.critical(
                self,
                "查询失败",
                error.strip() or f"检索进程退出码：{exit_code}",
            )
            return
        result_path = self.current_run_dir / "lookup_results.json"
        try:
            with result_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.results = list(payload.get("results", []))
        except (OSError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "读取结果失败", str(error))
            return
        self.result_table.setRowCount(0)
        self._print_lookup_results()
        for result in self.results:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            path_item = QTableWidgetItem(str(result.get("image_path", "")))
            path_item.setData(Qt.ItemDataRole.UserRole, result)
            distance_item = QTableWidgetItem(
                f"{float(result.get('distance', 0.0)):.6f}"
            )
            self.result_table.setItem(row, 0, path_item)
            self.result_table.setItem(row, 1, distance_item)
        if self.results:
            self.result_table.selectRow(0)
            self.status_label.setText(f"查询完成：{len(self.results)} 个匹配结果")
        else:
            self.status_label.setText("查询完成，但没有匹配结果")

    def _print_lookup_results(self) -> None:
        """Print IDs and ROI details to the terminal that launched the GUI."""
        print("\n查询结果（image_id / roi_id）:", flush=True)
        for index, result in enumerate(self.results, start=1):
            print(
                f"[{index}] library={result.get('library_type', '')} "
                f"distance={float(result.get('distance', 0.0)):.6f} "
                f"image_path={result.get('image_path', '')} "
                f"image_id={result.get('image_id', '')} "
                f"roi_id={result.get('roi_id', '')} "
                f"bbox_original={result.get('bbox_original', '')}",
                flush=True,
            )

    def _result_row_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        self.show_result(current_row)

    def show_result(self, row: int) -> None:
        if row < 0 or row >= len(self.results):
            return
        result = self.results[row]
        source_path = Path(result.get("image_path", ""))
        try:
            if source_path.is_file():
                self.right_canvas.set_image(source_path)
                self.right_canvas.set_overlay_bbox(result.get("bbox_original"))
                self.status_label.setText(
                    f"匹配：{source_path}，"
                    f"distance={float(result.get('distance', 0.0)):.6f}"
                )
            else:
                self.right_canvas.clear_image()
                self.status_label.setText(
                    f"匹配成功，但原图不存在：{source_path}；"
                    "路径已显示在结果表中。"
                )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "显示匹配图像失败", str(error))

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.waitForFinished(1000)
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PySide6 GUI for viewing Dinomaly2 threshold/candidate regions and "
            "querying ROI feature libraries"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--good_library", required=True)
    parser.add_argument("--anomaly_library", required=True)
    parser.add_argument("--backbone", default="dinov2reg_vit_small_14")
    parser.add_argument("--image_size", type=int, default=672)
    parser.add_argument("--crop_size", type=int, default=672)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--la", type=int, default=1)
    parser.add_argument("--lc", type=int, default=2)
    parser.add_argument("--cr", type=int, default=1)
    parser.add_argument("--feature_merge", choices=("mean", "concat"), default="mean")
    parser.add_argument("--roi_size", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--input", default=None, help="Optional initial input image")
    parser.add_argument(
        "--prediction_dir",
        default=None,
        help=(
            "Output directory of dinomaly_two_threshold_predict.py; reads "
            "raw_regions/ and candidate_regions/ from it."
        ),
    )
    parser.add_argument(
        "--raw_regions",
        default=None,
        help="Optional raw good-threshold Mask file or directory",
    )
    parser.add_argument(
        "--candidate_regions",
        default=None,
        help=(
            "Optional candidate-region mask file or directory. A directory is "
            "matched by input's data_root-relative path and .png suffix."
        ),
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional input root used to map input images to candidate_regions/<relative>.png",
    )
    parser.add_argument("--output_dir", default="./gui_lookup_results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = MainWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
