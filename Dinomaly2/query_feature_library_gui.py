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
from dinomaly_two_stage import (
    calculate_distance_offset,
    load_labelme_library_mask,
    load_mask,
    mask_bbox,
)
from dinomaly_two_threshold_predict import final_score_label
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

TWO_STAGE_FORMULA_HTML = """<h4 style="margin:2px;">两阶段分数调整公式</h4>
<pre style="font-family:Consolas,'Courier New',monospace; font-size:9pt; white-space:pre-wrap;">
d_good      = ‖v − p_good‖₂            良品库最近邻 L2 距离
d_anomaly   = ‖v − p_anomaly‖₂         异常库最近邻 L2 距离
confidence  = |d_good − d_anomaly| / (d_good + d_anomaly + ε)
offset      = min(confidence × offset_scale, max_offset)
signed_offset = −offset（近良品库） / +offset（近异常库）
adjusted_score = region_score + signed_offset
region_score = score_map 在 ROI 内的最大值
</pre>
<b>双阈值判定</b>（good_threshold &lt; anomaly_threshold）：
<ul style="margin:2px; padding-left:20px;">
<li>adjusted_score &lt; good_threshold → <span style="color:#00c853;"><b>正常</b></span></li>
<li>adjusted_score &gt; anomaly_threshold → <span style="color:#ff1744;"><b>异常</b></span></li>
<li>介于两阈值之间 → 取更近库类型；平局取阈值中点</li>
</ul>"""


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
        self.overlay_color = QColor("#ff1744")
        self.overlay_dashed = False
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.panning = False
        self.pan_last: Optional[QPointF] = None
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
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.panning = False
        self.pan_last = None
        self.clear_shapes(emit=False)
        self.clear_candidate_regions(emit=False)
        self.clear_overlay()
        self.update()

    def clear_image(self) -> None:
        self.image = None
        self.image_path = None
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.panning = False
        self.pan_last = None
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
            stored_region = {
                "region_id": int(
                    region.get("region_id", len(self.candidate_regions) + 1)
                ),
                "mask": mask,
                "bbox": tuple(float(value) for value in bbox),
                "area": int(region.get("area", int(mask.sum()))),
                "points": points,
            }
            score = region.get("score", region.get("region_score"))
            if score is not None:
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = None
                if score is not None and np.isfinite(score):
                    stored_region["score"] = score
            color = region.get("color")
            if color is not None:
                stored_region["color"] = str(color)
            label = region.get("label")
            if label is not None:
                stored_region["label"] = str(label)
            self.candidate_regions.append(stored_region)
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
        color: Optional[QColor] = None,
        dashed: bool = False,
    ) -> None:
        self.overlay_bbox = (
            tuple(float(value) for value in bbox)
            if bbox is not None and len(bbox) == 4
            else None
        )
        self.overlay_text = text
        self.overlay_dashed = bool(dashed)
        if color is not None:
            self.overlay_color = QColor(color)
        self.update()

    def clear_overlay(self) -> None:
        self.overlay_bbox = None
        self.overlay_text = ""
        self.overlay_dashed = False
        self.update()

    def _image_rect(self, pan: QPointF) -> QRectF:
        if self.image is None or self.image.width() < 1 or self.image.height() < 1:
            return QRectF()
        margin = 12.0
        available_width = max(float(self.width()) - 2.0 * margin, 1.0)
        available_height = max(float(self.height()) - 2.0 * margin, 1.0)
        scale = (
            min(
                available_width / float(self.image.width()),
                available_height / float(self.image.height()),
            )
            * self.zoom
        )
        width = float(self.image.width()) * scale
        height = float(self.image.height()) * scale
        return QRectF(
            (float(self.width()) - width) / 2.0 + pan.x(),
            (float(self.height()) - height) / 2.0 + pan.y(),
            width,
            height,
        )

    def image_rect(self) -> QRectF:
        return self._image_rect(self.pan)

    def _clamp_pan(self) -> None:
        if self.image is None:
            self.pan = QPointF(0.0, 0.0)
            return
        rect = self._image_rect(QPointF(0.0, 0.0))
        width, height = rect.width(), rect.height()
        pan_x, pan_y = self.pan.x(), self.pan.y()
        if width <= float(self.width()):
            pan_x = 0.0
        else:
            pan_x = min(max(pan_x, -(width - 40.0)), float(self.width()) - 40.0)
        if height <= float(self.height()):
            pan_y = 0.0
        else:
            pan_y = min(max(pan_y, -(height - 40.0)), float(self.height()) - 40.0)
        self.pan = QPointF(pan_x, pan_y)

    def fit_to_window(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.update()

    def wheelEvent(self, event) -> None:
        if self.image is None:
            return
        position = event.position()
        anchor = self.widget_to_image(position)
        if anchor is None:
            return
        factor = 1.25 ** (event.angleDelta().y() / 120.0)
        new_zoom = min(max(self.zoom * factor, 0.1), 32.0)
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self.zoom = new_zoom
        base = self._image_rect(QPointF(0.0, 0.0))
        self.pan = QPointF(
            position.x() - (base.left() + anchor.x() * base.width() / float(self.image.width())),
            position.y() - (base.top() + anchor.y() * base.height() / float(self.image.height())),
        )
        self._clamp_pan()
        self.update()

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
            if selected:
                color = QColor("#ffeb3b")
            else:
                color = QColor(candidate.get("color", "#00bcd4"))
            painter.setPen(QPen(color, 3.0 if selected else 2.0))
            points = candidate.get("points", [])
            polygon = QPolygonF(
                [self.image_to_widget(point) for point in points]
            )
            if len(points) >= 3:
                fill_color = QColor(color)
                fill_color.setAlpha(55 if selected else 30)
                painter.setBrush(fill_color)
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
            label = candidate.get("label")
            if label is not None:
                text = str(label)
            else:
                score = candidate.get("score")
                text = f"{float(score):.4f}" if score is not None else ""
            if text:
                text_rect = QRectF(
                    label_point.x(),
                    max(0.0, label_point.y() - 22.0),
                    240.0,
                    20.0,
                )
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.setPen(QColor("#ff1744"))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, text)
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
            pen = QPen(self.overlay_color, 3.0)
            if self.overlay_dashed:
                pen.setWidth(2.0)
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(top_left, bottom_right).normalized())
            if self.overlay_text:
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                text_rect = QRectF(top_left.x(), top_left.y() - 24, 420, 22)
                painter.setPen(QColor("#ff1744"))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, self.overlay_text)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_last = event.position()
            return
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
        if self.panning and self.pan_last is not None:
            delta = event.position() - self.pan_last
            self.pan_last = event.position()
            self.pan = QPointF(self.pan.x() + delta.x(), self.pan.y() + delta.y())
            self._clamp_pan()
            self.update()
            return
        if not self.editable or self.mode != "rectangle" or self.drag_start is None:
            return
        point = self.widget_to_image(event.position())
        if point is not None:
            self.drag_end = point
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.pan_last = None
            return
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
        self.score_map: Optional[np.ndarray] = None
        self.query_result_image: Optional[Path] = None
        self.unmatched_region_count = 0
        self.images_root: Optional[Path] = None
        self._file_rows: List[Dict[str, Any]] = []
        self.setWindowTitle(
            f"Dinomaly2 ROI 特征库反查 — preds: {Path(self.args.preds).expanduser()}"
        )
        self.resize(2300, 950)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "搜索路径/文件名；回车输入含 good/bad 子目录的图像根目录"
        )
        self.open_button = QPushButton("选择文件")
        self.mode_combo = QComboBox()
        if self._artifact_root("candidate_regions") is not None:
            self.mode_combo.addItem("候选区域", "candidate")
        self.mode_combo.addItem("矩形", "rectangle")
        self.mode_combo.addItem("多边形", "polygon")
        self.finish_button = QPushButton("完成多边形")
        self.undo_button = QPushButton("撤销")
        self.clear_button = QPushButton("清空区域")
        self.query_button = QPushButton("查询特征库")
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.setToolTip("将所有图像视图的缩放还原到适应窗口大小")
        self.threshold_label = QLabel()
        self.threshold_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        self._update_threshold_label()
        self.status_label = QLabel("请打开图像，然后在中间选择或绘制查询区域")
        self.status_label.setWordWrap(True)

        self.formula_label = QLabel()
        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
        self.formula_label.setWordWrap(True)
        self.formula_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.calculation_label = QLabel()
        self.calculation_label.setTextFormat(Qt.TextFormat.RichText)
        self.calculation_label.setWordWrap(True)
        self.calculation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.formula_label.setText(TWO_STAGE_FORMULA_HTML)

        self.left_canvas = ImageCanvas(editable=True)
        self.right_canvas = ImageCanvas(editable=False)
        self.adjust_canvas = ImageCanvas(editable=False)
        self._reset_calculation_panel()
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["图像路径", "距离", "库类型"])
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
        self.result_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.result_table.setMinimumHeight(150)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("图像根目录/搜索："))
        controls.addWidget(self.search_edit, 2)
        controls.addWidget(self.open_button)
        controls.addWidget(QLabel("中间区域："))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.finish_button)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        controls.addWidget(self.fit_button)
        controls.addWidget(self.threshold_label)
        controls.addWidget(self.query_button)

        file_panel = QWidget()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(QLabel("图像列表（good/bad 分组，按调整后分数升序，点击加载）"))
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        file_layout.addWidget(self.file_tree, 1)

        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.addWidget(QLabel("原始 Dinomaly2 区域（青色）/ 标注异常区域（红色）"))
        self.raw_canvas = ImageCanvas(editable=False)
        raw_layout.addWidget(self.raw_canvas, 1)

        candidate_panel = QWidget()
        candidate_layout = QVBoxLayout(candidate_panel)
        candidate_layout.addWidget(QLabel("候选区域 / 手动画 ROI（单选后查询）"))
        candidate_layout.addWidget(self.left_canvas, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("最近邻原图 / 对应 ROI（良品库绿色，异常库红色）"))
        right_layout.addWidget(self.right_canvas, 1)
        right_layout.addWidget(self.result_table)

        adjust_panel = QWidget()
        adjust_layout = QVBoxLayout(adjust_panel)
        adjust_layout.addWidget(QLabel("两阶段调整结果（完整：全图分数与全部区域）"))
        adjust_layout.addWidget(self.adjust_canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(file_panel)
        splitter.addWidget(raw_panel)
        splitter.addWidget(candidate_panel)
        splitter.addWidget(right_panel)
        splitter.addWidget(adjust_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 1)
        splitter.setStretchFactor(4, 1)
        splitter.setSizes([320, 430, 430, 430, 430])

        formula_panel = QWidget()
        formula_layout = QVBoxLayout(formula_panel)
        formula_layout.setContentsMargins(0, 0, 0, 0)
        formula_layout.addWidget(QLabel("公式"))
        formula_layout.addWidget(self._scrollable_label(self.formula_label), 1)
        calculation_panel = QWidget()
        calculation_layout = QVBoxLayout(calculation_panel)
        calculation_layout.setContentsMargins(0, 0, 0, 0)
        calculation_layout.addWidget(QLabel("实际计算与结果"))
        calculation_layout.addWidget(self._scrollable_label(self.calculation_label), 1)
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.addWidget(formula_panel, 1)
        info_layout.addWidget(calculation_panel, 1)

        bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        bottom_splitter.addWidget(splitter)
        bottom_splitter.addWidget(info_widget)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([600, 220])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(bottom_splitter, 1)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)

        self.open_button.clicked.connect(self.open_image)
        self.search_edit.textChanged.connect(self._apply_search)
        self.search_edit.returnPressed.connect(self._search_box_entered)
        self.file_tree.itemClicked.connect(self.file_item_clicked)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        self.finish_button.clicked.connect(self.left_canvas.finish_polygon)
        self.undo_button.clicked.connect(self.left_canvas.undo)
        self.clear_button.clicked.connect(self.clear_query_selection)
        self.query_button.clicked.connect(self.start_query)
        self.fit_button.clicked.connect(self.fit_all_canvases)
        self.left_canvas.shapes_changed.connect(self.update_controls)
        self.left_canvas.shapes_changed.connect(self._update_selected_region_calculation)
        self.left_canvas.candidate_changed.connect(self.candidate_selection_changed)
        self.result_table.currentCellChanged.connect(self._result_row_changed)
        self.query_button.setEnabled(False)

        self.change_mode(self.mode_combo.currentIndex())

        data_root = getattr(args, "data_root", None)
        if data_root:
            self._set_images_root(Path(data_root).expanduser())
        else:
            self._rebuild_file_list()

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
            score_map = self.load_score_map(image_path)
            self.score_map = score_map
            self.query_result_image = None
            self.load_raw_regions(image_path, score_map)
            self.load_annotation_regions(image_path, score_map)
            candidate_path = self.load_candidate_regions(image_path, score_map)
            self.current_run_dir = None
            self.right_canvas.clear_image()
            self.result_table.setRowCount(0)
            self.results.clear()
            self._update_two_stage_panel()
            self._update_selected_region_calculation()
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
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "打开失败", str(error))

    def fit_all_canvases(self) -> None:
        """Reset zoom/pan on every image view to fit the window."""

        for canvas in (
            self.left_canvas,
            self.raw_canvas,
            self.right_canvas,
            self.adjust_canvas,
        ):
            canvas.fit_to_window()

    @staticmethod
    def _scrollable_label(label: QLabel) -> QScrollArea:
        """Wrap a rich-text label so it scrolls instead of being clipped.

        A plain QLabel reports a large minimum height for its content, which
        blocks the vertical splitter from resizing it; a scroll area has a
        small minimum and shows scrollbars when the content overflows.
        """

        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(80)
        return scroll

    def _update_threshold_label(self) -> None:
        good_threshold = float(self.args.good_threshold)
        anomaly_threshold = float(self.args.anomaly_threshold)
        offset_scale = float(self.args.offset_scale)
        if self.args.max_offset is None:
            max_offset_text = "∞"
        else:
            max_offset_text = f"{float(self.args.max_offset):.4f}"
        source = getattr(self.args, "config_source", "CLI")
        self.threshold_label.setText(
            f"good_threshold={good_threshold:.4f}   "
            f"anomaly_threshold={anomaly_threshold:.4f}   "
            f"offset_scale={offset_scale:.4f}   "
            f"max_offset={max_offset_text}   "
            f"({source})"
        )

    def _reset_calculation_panel(self) -> None:
        self.calculation_label.setText(
            "打开图像后，选择候选区域或绘制 ROI，即可查看该区域的两阶段计算。"
        )

    @staticmethod
    def _find_detail_region(
        regions: Sequence[Mapping[str, Any]],
        bbox: Optional[Sequence[float]] = None,
        area: Optional[int] = None,
        process_size: int = 0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Mapping[str, Any]]:
        """Match a canvas region to a prediction-detail region.

        Details written with ``--process_size`` store bboxes/areas in the
        downsampled space; they are scaled to the image space before the
        exact (2-decimal) bbox comparison and the area fallback.
        """

        scale_x = scale_y = 1.0
        if int(process_size) > 0 and image_shape:
            height, width = image_shape
            scale_x = float(width) / float(process_size)
            scale_y = float(height) / float(process_size)
        if bbox is not None and len(bbox) == 4:
            target = [round(float(value), 2) for value in bbox]
            for region in regions:
                region_bbox = region.get("bbox_original")
                if (
                    isinstance(region_bbox, (list, tuple))
                    and len(region_bbox) == 4
                ):
                    scaled = [
                        round(float(region_bbox[0]) * scale_x, 2),
                        round(float(region_bbox[1]) * scale_y, 2),
                        round(float(region_bbox[2]) * scale_x, 2),
                        round(float(region_bbox[3]) * scale_y, 2),
                    ]
                    if scaled == target:
                        return region
        if area is not None and int(area) > 0:
            for region in regions:
                detail_area = int(round(int(region.get("area", -1)) * scale_x * scale_y))
                if detail_area == int(area):
                    return region
        return None

    @staticmethod
    def _result_library_type(result: Mapping[str, Any]) -> Optional[str]:
        library_type = str(result.get("library_type", "")).strip().casefold()
        if library_type in {"good", "good_library", "良品", "良品库"}:
            return "good"
        if library_type in {"anomaly", "anomaly_library", "异常", "异常库"}:
            return "anomaly"
        return None

    def _prediction_details_path(self, image_path: Path) -> Optional[Path]:
        preds = self._preds_dir()
        details_root = preds / "details"
        if not details_root.is_dir():
            return None
        return self._artifact_path(details_root, image_path, ".json")

    @staticmethod
    def _similar_library_cn(similar_library: Any) -> str:
        text = str(similar_library).strip().casefold()
        if text in {"good", "good_library", "良品", "良品库"}:
            return "良品库"
        if text in {"anomaly", "anomaly_library", "异常", "异常库"}:
            return "异常库"
        if text == "tie":
            return "平局"
        if text == "invalid":
            return "无效"
        return str(similar_library) or "无"

    def _update_two_stage_panel(self) -> None:
        """Show the complete two-stage prediction result for the current image.

        The result comes from ``--root/preds/details/<image>.json`` written
        by dinomaly_two_threshold_predict.py, so it appears as soon as the
        input image is loaded and does not depend on a feature-library query.
        """

        image_path = self.left_canvas.image_path
        if image_path is None:
            self.adjust_canvas.clear_image()
            return
        details_path = self._prediction_details_path(image_path)
        if details_path is None:
            self.adjust_canvas.clear_image()
            return
        try:
            with details_path.open("r", encoding="utf-8") as file:
                detail = json.load(file)
        except (OSError, json.JSONDecodeError):
            self.adjust_canvas.clear_image()
            return

        raw_score = float(detail.get("raw_score", 0.0))
        final_label = str(detail.get("final_label", ""))
        final_cn = "正常" if final_label == "good" else "异常"
        good_threshold = float(
            detail.get("good_threshold", float(self.args.good_threshold))
        )
        anomaly_threshold = float(
            detail.get("anomaly_threshold", float(self.args.anomaly_threshold))
        )

        self.adjust_canvas.set_image(image_path)
        regions = detail.get("regions", [])
        judged = []
        unmatched_count = 0
        for region_data in self.left_canvas.candidate_regions:
            candidate = dict(region_data)
            candidate["color"] = "#00bcd4"
            candidate["label"] = None
            candidate["is_judged"] = False
            match = self._find_detail_region(
                regions,
                candidate.get("bbox"),
                candidate.get("area"),
                process_size=int(detail.get("process_size") or 0),
                image_shape=(
                    self.left_canvas.image.height(),
                    self.left_canvas.image.width(),
                ),
            )
            if match is not None:
                region_score = float(match.get("region_score", 0.0))
                signed_offset = float(match.get("signed_offset", 0.0))
                adjusted = region_score + signed_offset
                label, _reason = final_score_label(
                    adjusted,
                    good_threshold,
                    anomaly_threshold,
                    str(match.get("similar_library", "")),
                )
                candidate["color"] = (
                    "#00c853" if label == "good" else "#ff1744"
                )
                candidate["score"] = adjusted
                candidate["is_judged"] = True
            else:
                unmatched_count += 1
            judged.append(candidate)
        self.unmatched_region_count = unmatched_count
        self.adjust_canvas.set_candidate_regions(judged, emit=False)
        judged_with_score = [
            candidate
            for candidate in judged
            if candidate.get("is_judged") and candidate.get("score") is not None
        ]
        if judged_with_score:
            strongest = max(
                judged_with_score,
                key=lambda candidate: (
                    float(candidate.get("score", 0.0)),
                    float(candidate.get("area", 0)),
                ),
            )
            strongest_bbox = strongest.get("bbox")
            if isinstance(strongest_bbox, (list, tuple)) and len(strongest_bbox) == 4:
                self.adjust_canvas.set_overlay_bbox(
                    strongest_bbox,
                    color=QColor("#ff1744"),
                    dashed=True,
                )

    def _update_selected_region_calculation(self) -> None:
        """Show the whole-image final score and the selected ROI's calculation.

        The first line reproduces the image-level adjusted_score written to
        results.csv (raw_score + selected-region signed_offset); the rest
        details the currently selected ROI, reusing the distances stored in
        the prediction details or falling back to the last feature-library
        query on the same image.
        """

        image_path = self.left_canvas.image_path
        if image_path is None:
            self._reset_calculation_panel()
            return
        lines: List[str] = ['<h4 style="margin:2px;">实际计算与结果</h4>']

        detail = None
        details_path = self._prediction_details_path(image_path)
        if details_path is not None:
            try:
                with details_path.open("r", encoding="utf-8") as file:
                    detail = json.load(file)
            except (OSError, json.JSONDecodeError):
                detail = None
        if detail:
            raw_score = float(detail.get("raw_score", 0.0))
            good_threshold = float(
                detail.get("good_threshold", float(self.args.good_threshold))
            )
            anomaly_threshold = float(
                detail.get("anomaly_threshold", float(self.args.anomaly_threshold))
            )
            regions = detail.get("regions", [])
            if regions:
                strongest = max(
                    regions,
                    key=lambda region: (
                        float(region.get("region_score", 0.0))
                        + float(region.get("signed_offset", 0.0)),
                        float(region.get("region_score", 0.0)),
                    ),
                )
                signed_offset = float(strongest.get("signed_offset", 0.0))
                similar_library = str(strongest.get("similar_library", ""))
            else:
                signed_offset = 0.0
                similar_library = ""
            adjusted_score = raw_score + signed_offset
            label, _reason = final_score_label(
                adjusted_score,
                good_threshold,
                anomaly_threshold,
                similar_library,
            )
            final_cn = "正常" if label == "good" else "异常"
            lines.append(
                f"整图最终调整（按调整后分数最大区域）：adjusted_score = "
                f"{raw_score:.4f} + ({signed_offset:+.4f}) = "
                f"<b>{adjusted_score:.4f}</b>（{final_cn}）"
            )
        else:
            lines.append("整图最终调整：未找到预测详情（--root/preds/details/）。")
            good_threshold = float(self.args.good_threshold)
            anomaly_threshold = float(self.args.anomaly_threshold)
        if self.unmatched_region_count > 0:
            lines.append(
                f"<span style=\"color:#ff9800;\">提示："
                f"{self.unmatched_region_count} 个候选区域未匹配到预测详情"
                f"（details 与候选掩码可能不是同一次预测/同一阈值生成的），"
                f"这些区域以青色显示原始分数、未参与两阶段调整。</span>"
            )
        lines.append("<hr>")

        if self.left_canvas.mode == "candidate":
            index = self.left_canvas.selected_candidate_index
            if index is None or not 0 <= index < len(
                self.left_canvas.candidate_regions
            ):
                lines.append("未选择候选区域：请单击左侧一个候选多边形。")
                self.calculation_label.setText("<br>".join(lines))
                return
            region_data = self.left_canvas.candidate_regions[index]
            roi_bbox = region_data.get("bbox")
            roi_area = int(region_data.get("area", 0))
            roi_mask = np.asarray(region_data.get("mask"), dtype=bool)
        else:
            if not self.left_canvas.shapes:
                lines.append("未绘制 ROI：请绘制矩形或多边形。")
                self.calculation_label.setText("<br>".join(lines))
                return
            try:
                roi_mask = np.asarray(self.left_canvas.mask_array(), dtype=bool)
            except (RuntimeError, ValueError) as error:
                lines.append(f"无法生成 ROI 掩码：{error}")
                self.calculation_label.setText("<br>".join(lines))
                return
            roi_bbox = mask_bbox(roi_mask)
            roi_area = int(np.count_nonzero(roi_mask))

        if self.score_map is not None and np.any(roi_mask):
            region_score = float(np.nanmax(self.score_map[roi_mask]))
            lines.append(f"region_score（ROI 内 max） = <b>{region_score:.4f}</b>")
        else:
            region_score = None
            lines.append("region_score = 未提供 score_map")
        lines.append(
            f"ROI 面积 = {roi_area} 像素（占图像 {self._area_ratio_text(roi_area)}）"
        )

        if detail:
            matched = self._find_detail_region(
                detail.get("regions", []),
                roi_bbox,
                roi_area,
                process_size=int(detail.get("process_size") or 0),
                image_shape=(
                    self.left_canvas.image.height(),
                    self.left_canvas.image.width(),
                ),
            )
            if matched is not None:
                self._append_region_decision(
                    lines,
                    matched,
                    region_score,
                    good_threshold,
                    anomaly_threshold,
                )
                self.calculation_label.setText("<br>".join(lines))
                return

        if (
            self.query_result_image is not None
            and Path(self.query_result_image).resolve() == image_path.resolve()
            and self.results
        ):
            good_distances = [
                float(result.get("distance"))
                for result in self.results
                if self._result_library_type(result) == "good"
            ]
            anomaly_distances = [
                float(result.get("distance"))
                for result in self.results
                if self._result_library_type(result) == "anomaly"
            ]
            if good_distances and anomaly_distances:
                decision = calculate_distance_offset(
                    min(good_distances),
                    min(anomaly_distances),
                    float(self.args.offset_scale),
                    self.args.max_offset,
                    float(self.args.offset_eps),
                )
                query_region = {
                    "good_distance": min(good_distances),
                    "anomaly_distance": min(anomaly_distances),
                    "confidence": decision["confidence"],
                    "offset": decision["offset"],
                    "signed_offset": decision["signed_offset"],
                    "similar_library": decision["similar_library"],
                }
                self._append_region_decision(
                    lines,
                    query_region,
                    region_score,
                    good_threshold,
                    anomaly_threshold,
                )
                self.calculation_label.setText("<br>".join(lines))
                return
        lines.append("该 ROI 不在预测详情中；点击‘查询特征库’获取两库最近距离后显示。")
        self.calculation_label.setText("<br>".join(lines))

    def _append_region_decision(
        self,
        lines: List[str],
        region: Mapping[str, Any],
        region_score: Optional[float],
        good_threshold: float,
        anomaly_threshold: float,
    ) -> None:
        """Append one region's distance/offset calculation and final judgment."""

        good_distance = float(region.get("good_distance", 0.0))
        anomaly_distance = float(region.get("anomaly_distance", 0.0))
        similar_library = str(region.get("similar_library", ""))
        similar_cn = self._similar_library_cn(similar_library)
        lines.append(f"d_good（良品库第 1 近） = <b>{good_distance:.6f}</b>")
        lines.append(f"d_anomaly（异常库第 1 近） = <b>{anomaly_distance:.6f}</b>")
        lines.append(
            f"confidence = {float(region.get('confidence', 0.0)):.6f}，"
            f"offset = {float(region.get('offset', 0.0)):.6f}，"
            f"signed_offset = {float(region.get('signed_offset', 0.0)):+.6f}（近{similar_cn}）"
        )
        if region_score is None:
            return
        signed_offset = float(region.get("signed_offset", 0.0))
        adjusted = region_score + signed_offset
        lines.append(
            f"adjusted_score = {region_score:.4f} + ({signed_offset:+.4f}) = "
            f"<b>{adjusted:.4f}</b>"
        )
        lines.append(
            f"双阈值：good_threshold = <b>{good_threshold:.4f}</b>，"
            f"anomaly_threshold = <b>{anomaly_threshold:.4f}</b>"
        )
        label, reason = final_score_label(
            adjusted,
            good_threshold,
            anomaly_threshold,
            similar_library,
        )
        label_cn = "正常" if label == "good" else "异常"
        color = "#00c853" if label == "good" else "#ff1744"
        reason_cn = {
            "adjusted_below_good_threshold": "调整后低于良品阈值",
            "adjusted_above_anomaly_threshold": "调整后高于异常阈值",
            "feature_library_good": "介于两阈值之间，取更近库（良品库）",
            "feature_library_anomaly": "介于两阈值之间，取更近库（异常库）",
            "threshold_midpoint_fallback": "两库平局，取阈值中点判定",
        }.get(reason, reason)
        lines.append(
            f"最终判定：<span style=\"color:{color}; font-weight:bold;\">"
            f"{label_cn}</span>（{reason_cn}）"
        )

    def _preds_dir(self) -> Path:
        return Path(self.args.preds).expanduser()

    def _artifact_root(self, artifact_name: str) -> Optional[Path]:
        """Return ``preds/<artifact_name>`` or None when preds is missing."""

        preds = self._preds_dir()
        if not preds.is_dir():
            return None
        return preds / artifact_name

    def _artifact_path(
        self,
        root: Path,
        image_path: Path,
        suffix: str,
    ) -> Optional[Path]:
        """Resolve one predictor artifact using the input's relative path."""

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
                        root / relative.with_suffix(suffix),
                        root / relative,
                    ]
                )

        candidates.extend(
            [
                root / image_path.name,
                root / image_path.with_suffix(suffix).name,
                root / f"{image_path.stem}{suffix}",
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
            for path in root.rglob(f"{image_path.stem}{suffix}")
            if path.is_file()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _mask_components(
        self,
        mask_path: Path,
        score_map: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
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
            region = {
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
            if score_map is not None:
                region_score = np.asarray(score_map)[component_mask]
                if region_score.size:
                    score = float(np.nanmax(region_score))
                    if np.isfinite(score):
                        region["score"] = score
            regions.append(region)
        regions.sort(key=lambda region: (-region["area"], region["region_id"]))
        for index, region in enumerate(regions, start=1):
            region["region_id"] = index
        return regions

    def load_annotation_regions(
        self,
        image_path: Path,
        score_map: Optional[np.ndarray] = None,
    ) -> None:
        """Overlay annotation-mask anomaly regions on the raw-region canvas.

        ``--mask_dir`` mirrors the ``--data_root`` layout, so an image maps to
        ``mask_dir/<relative>.json`` (LabelMe, label 'good'/'ignore' are
        skipped) or to a binary mask with another extension.  Anomaly regions
        are drawn as red polygons on the first canvas, labelled with the
        maximum of the original Dinomaly2 score map inside each region.
        """

        mask_dir = getattr(self.args, "mask_dir", None)
        if not mask_dir:
            return
        mask_root = Path(mask_dir).expanduser()
        if not mask_root.is_dir():
            return
        if self.left_canvas.image is None:
            return
        image_shape = (
            self.left_canvas.image.height(),
            self.left_canvas.image.width(),
        )
        mask_path = None
        for suffix in (".json", ".png", ".npy", ".tif", ".tiff", ".bmp"):
            mask_path = self._artifact_path(mask_root, image_path, suffix)
            if mask_path is not None:
                break
        if mask_path is None:
            return
        try:
            if mask_path.suffix.lower() == ".json":
                mask = load_labelme_library_mask(
                    mask_path,
                    image_shape,
                    "anomaly",
                    good_labels=("good",),
                    ignore_labels=("ignore",),
                )
            else:
                mask = load_mask(mask_path, image_shape)
            mask_regions = self._mask_components_from_mask(mask)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "读取标注 Mask 失败", str(error))
            return
        for region in mask_regions:
            region["color"] = "#ff1744"
            if score_map is not None and np.any(region["mask"]):
                region_values = np.asarray(score_map)[region["mask"]]
                if region_values.size:
                    score = float(np.nanmax(region_values))
                    if np.isfinite(score):
                        region["score"] = score
        combined = list(self.raw_canvas.candidate_regions) + mask_regions
        self.raw_canvas.set_candidate_regions(combined, emit=False)

    def _mask_components_from_mask(
        self,
        mask: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Split a binary mask into connected components for display."""

        binary = (np.asarray(mask) > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
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

    def load_score_map(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and resize the predictor score map to the input image."""

        root = self._artifact_root("score_maps")
        if root is None:
            return None
        score_path = self._artifact_path(root, image_path, ".npy")
        if score_path is None:
            return None
        score_map = np.asarray(np.load(score_path), dtype=np.float32)
        score_map = np.squeeze(score_map)
        if score_map.ndim != 2:
            raise ValueError(
                f"Score map must be 2D: {score_path}; got {score_map.shape}"
            )
        if self.left_canvas.image is None:
            raise RuntimeError("尚未打开输入图像")
        expected_shape = (
            self.left_canvas.image.height(),
            self.left_canvas.image.width(),
        )
        if score_map.shape != expected_shape:
            score_map = cv2.resize(
                score_map,
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)

    def load_raw_regions(
        self,
        image_path: Path,
        score_map: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Load the direct good-threshold Mask into the left canvas."""

        self.raw_canvas.clear_candidate_regions(emit=False)
        root = self._artifact_root("raw_regions")
        if root is None:
            return None
        mask_path = self._artifact_path(root, image_path, ".png")
        if mask_path is None:
            return None
        self.raw_canvas.set_candidate_regions(
            self._mask_components(mask_path, score_map),
            emit=False,
        )
        return mask_path

    def load_candidate_regions(
        self,
        image_path: Path,
        score_map: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Load one mask and split it into selectable connected components."""

        self.left_canvas.clear_candidate_regions(emit=False)
        root = self._artifact_root("candidate_regions")
        if root is None:
            return None
        mask_path = self._artifact_path(root, image_path, ".png")
        if mask_path is None:
            return None
        self.left_canvas.set_candidate_regions(
            self._mask_components(mask_path, score_map),
            emit=False,
        )
        return mask_path

    def _area_ratio_text(self, area: int) -> str:
        """Format an area in pixels as a percentage of the input image."""

        if self.left_canvas.image is None:
            return ""
        total = self.left_canvas.image.width() * self.left_canvas.image.height()
        if total <= 0:
            return ""
        return f"{float(area) / total * 100.0:.3f}%"

    def candidate_selection_changed(self) -> None:
        self.update_controls()
        self._update_selected_region_calculation()
        index = self.left_canvas.selected_candidate_index
        if index is None or index >= len(self.left_canvas.candidate_regions):
            if self.left_canvas.mode == "candidate":
                self.status_label.setText("当前未选择候选区域，请单击一个候选多边形。")
            return
        region = self.left_canvas.candidate_regions[index]
        self.status_label.setText(
            f"已选择候选区域 R{index + 1}（面积 {region['area']}，"
            f"占图像 {self._area_ratio_text(int(region['area']))}），"
            "请点击‘查询特征库’。"
        )

    def clear_query_selection(self) -> None:
        """Clear the active manual ROI or the single candidate selection."""

        self.left_canvas.clear_shapes(emit=False)
        if self.left_canvas.mode == "candidate":
            self.left_canvas.select_candidate(-1)
        else:
            self.left_canvas.shapes_changed.emit()
        self.status_label.setText("已清空当前查询区域，请重新选择或绘制 ROI。")

    def _set_images_root(self, root: Path) -> None:
        """Set the good/bad image root and rebuild the file list."""

        self.images_root = Path(root)
        self.args.data_root = str(self.images_root)
        self._rebuild_file_list()
        self.status_label.setText(f"图像根目录：{self.images_root}")

    def _rebuild_file_list(self) -> None:
        """Build the grouped, adjusted-score-sorted image file list."""

        self.file_tree.clear()
        self._file_rows = []
        if self.images_root is None or not self.images_root.is_dir():
            item = QTreeWidgetItem(["(未设置图像根目录；在搜索框输入含 good/bad 的目录后回车)"])
            self.file_tree.addTopLevelItem(item)
            return

        image_paths: List[Path] = []
        for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            image_paths.extend(self.images_root.rglob(f"*{extension}"))
        image_paths.extend(self.images_root.rglob("*.JPG"))
        image_paths = sorted(
            {path for path in image_paths if path.is_file()},
            key=lambda path: str(path).casefold(),
        )

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for image_path in image_paths:
            try:
                relative = image_path.relative_to(self.images_root)
            except ValueError:
                relative = Path(image_path.name)
            group = relative.parts[0] if relative.parts else ""
            raw_score = None
            adjusted_score = None
            details_path = self._prediction_details_path(image_path)
            if details_path is not None:
                try:
                    with details_path.open("r", encoding="utf-8") as file:
                        detail = json.load(file)
                    raw_score = float(detail.get("raw_score", np.nan))
                    adjusted_score = float(detail.get("adjusted_score", np.nan))
                    if not np.isfinite(raw_score):
                        raw_score = None
                    if not np.isfinite(adjusted_score):
                        adjusted_score = None
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            groups.setdefault(group, []).append(
                {
                    "path": image_path,
                    "group": group,
                    "raw": raw_score,
                    "adjusted": adjusted_score,
                }
            )

        for group, rows in sorted(groups.items()):
            rows.sort(
                key=lambda row: (
                    row["adjusted"]
                    if row["adjusted"] is not None
                    else float("inf"),
                )
            )
            group_item = QTreeWidgetItem([f"{group}（{len(rows)}）"])
            self.file_tree.addTopLevelItem(group_item)
            for row in rows:
                raw_text = f"{row['raw']:.4f}" if row["raw"] is not None else "—"
                adjusted_text = (
                    f"{row['adjusted']:.4f}"
                    if row["adjusted"] is not None
                    else "—"
                )
                item = QTreeWidgetItem(
                    [f"{row['path'].name}  raw={raw_text}  adj={adjusted_text}"]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, str(row["path"]))
                item.setToolTip(0, str(row["path"]))
                group_item.addChild(item)
                row["item"] = item
                self._file_rows.append(row)
            group_item.setExpanded(True)

    def _apply_search(self, text: str) -> None:
        """Filter the file list by path/filename substring (empty = all)."""

        query = str(text).strip().casefold()
        for row in self._file_rows:
            item = row.get("item")
            if item is None:
                continue
            relative = str(row["path"]).casefold()
            if not query or query in relative or query in str(row["path"].name).casefold():
                item.setHidden(False)
            else:
                item.setHidden(True)
        for index in range(self.file_tree.topLevelItemCount()):
            group_item = self.file_tree.topLevelItem(index)
            if group_item.childCount() == 0:
                continue
            visible = any(
                not group_item.child(child).isHidden()
                for child in range(group_item.childCount())
            )
            group_item.setHidden(not visible)

    def _search_box_entered(self) -> None:
        """Enter in the search box: load a root dir, a single image, or filter."""

        text = self.search_edit.text().strip().strip('"')
        if not text:
            self._apply_search("")
            return
        path = Path(text).expanduser()
        if path.is_dir():
            self._set_images_root(path)
            return
        if path.is_file():
            self.load_input_image(path)
            return
        self._apply_search(text)

    def file_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path_text = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_text:
            return
        self.load_input_image(Path(path_text))

    def open_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            str(self.images_root) if self.images_root is not None else "",
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
        root = self._preds_dir().parent
        arguments = [
            "-u",
            str(query_script),
            "--input",
            str(self.left_canvas.image_path.resolve()),
            "--region_mask",
            str(mask_path),
            "--good_library",
            str(root / "good"),
            "--anomaly_library",
            str(root / "anomaly"),
            "--top_k",
            str(self.args.top_k),
            "--output_dir",
            str(run_dir),
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
        self.query_result_image = self.left_canvas.image_path

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
            self.query_result_image = None
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
            library_name = self._library_display_name(result)
            library_item = QTableWidgetItem(library_name)
            library_item.setForeground(
                QColor("#00a844")
                if library_name == "良品库"
                else QColor("#d50000")
            )
            self.result_table.setItem(row, 0, path_item)
            self.result_table.setItem(row, 1, distance_item)
            self.result_table.setItem(row, 2, library_item)
        if self.results:
            self.result_table.selectRow(0)
            self.status_label.setText(f"查询完成：{len(self.results)} 个匹配结果")
        else:
            self.status_label.setText("查询完成，但没有匹配结果")
        self._update_selected_region_calculation()

    @staticmethod
    def _library_display_name(result: Mapping[str, Any]) -> str:
        library_type = str(result.get("library_type", "")).strip().casefold()
        if library_type in {"good", "good_library", "良品", "良品库"}:
            return "良品库"
        if library_type in {"anomaly", "anomaly_library", "异常", "异常库"}:
            return "异常库"
        return str(result.get("library_type", "未知库")) or "未知库"

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
                library_name = self._library_display_name(result)
                is_good_library = library_name == "良品库"
                overlay_color = (
                    QColor("#00c853")
                    if is_good_library
                    else QColor("#ff1744")
                )
                self.right_canvas.set_image(source_path)
                self.right_canvas.set_overlay_bbox(
                    result.get("bbox_original"),
                    text=f"distance={float(result.get('distance', 0.0)):.6f}",
                    color=overlay_color,
                )
                self.status_label.setText(
                    f"匹配：{source_path}，"
                    f"distance={float(result.get('distance', 0.0)):.6f}，"
                    f"库类型={library_name}"
                )
            else:
                self.right_canvas.clear_image()
                self.status_label.setText(
                    f"匹配成功，但原图不存在：{source_path}；"
                    f"库类型={self._library_display_name(result)}；"
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
    parser.add_argument(
        "--preds",
        required=True,
        help=(
            "Output directory of dinomaly_two_threshold_predict.py "
            "(<root>/preds/); the good/ anomaly/ libraries are read from its "
            "parent and all artifacts (score_maps/, raw_regions/, "
            "candidate_regions/, details/, run.json) from this directory"
        ),
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Optional image root containing good/ and bad/ subdirectories; "
            "drives the left file list (grouped, sorted by adjusted score)"
        ),
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help=(
            "Optional annotation-mask directory mirroring --data_root's "
            "layout; LabelMe JSON (label 'good'/'ignore' skipped) or binary "
            "masks. Anomaly regions are drawn as red polygons on the first "
            "canvas."
        ),
    )
    parser.add_argument("--input", default=None, help="Optional initial input image")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--output_dir", default="./gui_lookup_results")
    return parser


def load_prediction_config(args) -> Dict[str, Any]:
    """Overlay thresholds/offset settings from ``preds/run.json``.

    The two-threshold predictor records the exact parameters it used in
    ``run.json``, so the on-screen calculation matches the actual prediction.
    Defaults are applied first, then the run.json values.
    """

    for key, default in (
        ("good_threshold", 0.5),
        ("anomaly_threshold", 0.7),
        ("offset_scale", 1.0),
        ("max_offset", None),
        ("offset_eps", 1e-8),
    ):
        setattr(args, key, default)
    run_path = Path(args.preds).expanduser() / "run.json"
    if not run_path.is_file():
        return {}
    try:
        with run_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"WARNING: cannot read {run_path}: {error}", flush=True)
        return {}
    overridden: Dict[str, Any] = {}
    for key in (
        "good_threshold",
        "anomaly_threshold",
        "offset_scale",
        "max_offset",
        "offset_eps",
    ):
        value = config.get(key)
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        setattr(args, key, value)
        overridden[key] = value
    if overridden:
        args.config_source = "run.json"
        print(f"从 run.json 读取参数：{overridden}", flush=True)
    return overridden


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preds = Path(args.preds).expanduser()
    if not preds.is_dir():
        raise SystemExit(f"--preds does not exist: {preds}")
    root = preds.parent
    for subdir in ("good", "anomaly"):
        if not (root / subdir).is_dir():
            raise SystemExit(
                f"--preds parent must contain a {subdir}/ directory: {root}"
            )
    load_prediction_config(args)
    if args.good_threshold >= args.anomaly_threshold:
        raise SystemExit(
            f"good_threshold ({args.good_threshold}) must be smaller than "
            f"anomaly_threshold ({args.anomaly_threshold})"
        )
    if args.offset_scale < 0:
        raise SystemExit("offset_scale cannot be negative")
    if args.max_offset is not None and args.max_offset < 0:
        raise SystemExit("max_offset cannot be negative")
    if args.offset_eps < 0:
        raise SystemExit("offset_eps cannot be negative")
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = MainWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
