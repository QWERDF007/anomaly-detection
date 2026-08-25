"""PySide6 GUI for querying Dinomaly2 ROI feature libraries.

The query tab displays the direct good-threshold Mask, selectable
candidate/manual query ROIs, and the matched source image with its stored ROI.
A query runs in a separate Python process so the UI stays responsive while
Dinomaly2 and FAISS are loading/searching.  The library-patch tab inspects
the images used to build the good/anomaly libraries (from their metadata):
original image, mask (background 0 / foreground 255) and the stored
top-ratio patches drawn as blue dashed boxes.
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
    dilate_mask,
    feature_patch_geometry,
    load_labelme_library_mask,
    load_mask,
    mask_bbox,
)
from dinomaly_two_threshold_predict import final_score_label
from utils import refine_anomaly_map_guided
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

TWO_STAGE_FORMULA_HTML = """<h4 style="margin:2px;">两阶段分数自适应调整公式（无量纲 Relative Margin）</h4>
<pre style="font-family:Consolas,'Courier New',monospace; font-size:9pt; white-space:pre-wrap;">
d_good      = ‖v − p_good‖₂            良品库最近邻 L2 距离
d_anomaly   = ‖v − p_anomaly‖₂         异常库最近邻 L2 距离
margin      = (d_anomaly − d_good) / (d_anomaly + d_good) ∈ [-1, 1]
bandwidth   = anomaly_threshold − good_threshold (双阈值带宽)
offset      = |margin| × (bandwidth / 2) × offset_scale（上限 max_offset）
signed_offset = −margin × (bandwidth / 2) × offset_scale
adjusted_score = region_score + signed_offset
region_score = score_map 在 ROI 内 top x% 均值
</pre>
<b>双阈值判定</b>（good_threshold &lt; anomaly_threshold）：
<ul style="margin:2px; padding-left:20px;">
<li>adjusted_score &lt; good_threshold → <span style="color:#00c853;"><b>正常</b></span></li>
<li>adjusted_score ≥ good_threshold（含中间带与超过 anomaly_threshold）→ <span style="color:#ff1744;"><b>异常</b></span></li>
</ul>"""

TWO_STAGE_FORMULA_IP_HTML = """<h4 style="margin:2px;">两阶段分数调整公式（IP 索引）</h4>
<pre style="font-family:Consolas,'Courier New',monospace; font-size:9pt; white-space:pre-wrap;">
d_good    = 1 − 内积（良品库最近邻）
d_anomaly = 1 − 内积（异常库最近邻）
内积距离无界，不做 offset 连续修正，按近库取固定档位：
  d_anomaly &lt; d_good → region_score = 1.5 × anomaly_threshold（异常）
  否则              → region_score = 0.5 × good_threshold（正常）
</pre>
<b>双阈值判定</b>（good_threshold &lt; anomaly_threshold）：
<ul style="margin:2px; padding-left:20px;">
<li>region_score &lt; good_threshold → <span style="color:#00c853;"><b>正常</b></span></li>
<li>region_score ≥ good_threshold（含中间带与超过 anomaly_threshold）→ <span style="color:#ff1744;"><b>异常</b></span></li>
</ul>"""


def two_stage_formula_html(index_type: str) -> str:
    """Return the formula panel HTML for the library index type."""

    if str(index_type).casefold() in {"indexflatip", "ip"}:
        return TWO_STAGE_FORMULA_IP_HTML
    return TWO_STAGE_FORMULA_HTML


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
        # The GUI has four image columns plus the file list.  Keep the
        # horizontal minimum small enough that the splitter can fit on a
        # normal laptop display; fit_to_window scales the image itself.
        self.setMinimumSize(240, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show_scores = True
        self.patch_boxes: List[Tuple[QPointF, float]] = []
        self.patch_rects: List[QRectF] = []

    def set_patch_boxes(self, boxes: Sequence[Tuple[QPointF, float]]) -> None:
        """Show blue dashed boxes for the queried patch positions.

        Each entry is ``(center image point, half patch size in image px)``.
        """

        self.patch_boxes = [
            (QPointF(center), float(half_size)) for center, half_size in boxes
        ]
        self.patch_rects = [
            QRectF(
                center.x() - float(half_size),
                center.y() - float(half_size),
                float(half_size) * 2.0,
                float(half_size) * 2.0,
            )
            for center, half_size in self.patch_boxes
        ]
        self.update()

    def set_patch_rects(
        self,
        rects: Sequence[Sequence[float]],
    ) -> None:
        """Show blue dashed Patch rectangles in image coordinates."""

        self.patch_rects = []
        for rect in rects:
            if len(rect) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in rect]
            if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            self.patch_rects.append(QRectF(x1, y1, x2 - x1, y2 - y1).normalized())
        self.patch_boxes = []
        self.update()

    def clear_patch_boxes(self) -> None:
        self.patch_boxes.clear()
        self.patch_rects.clear()
        self.update()

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
        self.clear_patch_boxes()
        self.update()

    def set_numpy_image(self, array: np.ndarray) -> None:
        """Display a numpy image: 2D uint8 grayscale or HxWx3 uint8 RGB."""

        array = np.asarray(array)
        if array.ndim == 2:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            array = np.ascontiguousarray(array)
            height, width = array.shape
            image = QImage(
                array.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
            image = image.convertToFormat(QImage.Format.Format_RGB32)
        elif array.ndim == 3 and array.shape[2] == 3:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            array = np.ascontiguousarray(array)
            height, width, _channels = array.shape
            image = QImage(
                array.data,
                width,
                height,
                array.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            image = image.convertToFormat(QImage.Format.Format_RGB32)
        else:
            raise ValueError(
                "无法显示该图像形状；需要 2D 灰度或 HxWx3 RGB："
                f"{array.shape}"
            )
        self.image = image
        self.image_path = None
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.panning = False
        self.pan_last = None
        self.clear_shapes(emit=False)
        self.clear_candidate_regions(emit=False)
        self.clear_overlay()
        self.clear_patch_boxes()
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
        self.clear_patch_boxes()
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
            if bool(region.get("is_annotation", False)):
                stored_region["is_annotation"] = True
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
        widget_width = float(self.width())
        widget_height = float(self.height())
        pan_x, pan_y = self.pan.x(), self.pan.y()
        if width <= widget_width:
            pan_x = 0.0
        else:
            # rect.left = (W - w)/2 + pan_x must stay within
            # [W - w - 40, 40] so every image edge is reachable.
            center_x = (widget_width - width) / 2.0
            min_pan_x = center_x - 40.0
            max_pan_x = 40.0 - center_x
            pan_x = min(max(pan_x, min_pan_x), max_pan_x)
        if height <= widget_height:
            pan_y = 0.0
        else:
            center_y = (widget_height - height) / 2.0
            min_pan_y = center_y - 40.0
            max_pan_y = 40.0 - center_y
            pan_y = min(max(pan_y, min_pan_y), max_pan_y)
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
                # 多边形只绘制边线，不填充。
                painter.drawPolygon(polygon)
                label_point = polygon.boundingRect().topLeft()
            else:
                x1, y1, x2, y2 = candidate["bbox"]
                top_left = self.image_to_widget(QPointF(x1, y1))
                bottom_right = self.image_to_widget(QPointF(x2, y2))
                candidate_rect = QRectF(top_left, bottom_right).normalized()
                painter.drawRect(candidate_rect)
                label_point = candidate_rect.topLeft()
            label = candidate.get("label")
            if label in {"GOOD", "Anomaly"}:
                # Keep the color-coded ROI, but do not clutter the image with
                # classification text labels.
                text = ""
            elif label is not None:
                text = str(label)
            elif self.show_scores:
                score = candidate.get("score")
                text = f"{float(score):.4f}" if score is not None else ""
            else:
                text = ""
            if text:
                text_rect = QRectF(
                    label_point.x(),
                    max(0.0, label_point.y() - 22.0),
                    240.0,
                    20.0,
                )
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.setPen(QColor(candidate.get("color", "#ff1744")))
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
        for image_rect_patch in self.patch_rects:
            top_left = self.image_to_widget(image_rect_patch.topLeft())
            bottom_right = self.image_to_widget(image_rect_patch.bottomRight())
            pen = QPen(QColor("#2196f3"), 2.0)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(top_left, bottom_right).normalized())
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
        self._queried_candidate_index: Optional[int] = None
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
        if (
            self._artifact_root("candidate_regions") is not None
            or getattr(args, "mask_dir", None)
        ):
            self.mode_combo.addItem("候选区域", "candidate")
        self.mode_combo.addItem("矩形", "rectangle")
        self.mode_combo.addItem("多边形", "polygon")
        self.finish_button = QPushButton("完成多边形")
        self.undo_button = QPushButton("撤销")
        self.clear_button = QPushButton("清空区域")
        self.query_button = QPushButton("查询特征库")
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.setToolTip("将所有图像视图的缩放还原到适应窗口大小")
        self.region_top_spin = QDoubleSpinBox()
        self.region_top_spin.setRange(0.1, 100.0)
        self.region_top_spin.setDecimals(1)
        self.region_top_spin.setSingleStep(1.0)
        self.region_top_spin.setSuffix("%")
        self.region_top_spin.setValue(
            float(getattr(args, "region_top_ratio", 0.10)) * 100.0
        )
        self.region_top_spin.setToolTip(
            "区域分数 top x% 均值；初始值来自 run.json 元数据，可在此调整"
        )
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
        self.formula_label.setText(
            two_stage_formula_html(
                str(self._library_metadata().get("index_type", "IndexFlatL2"))
            )
        )

        self.left_canvas = ImageCanvas(editable=True)
        self.left_canvas.show_scores = False
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
        self.result_table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.result_table.setMinimumWidth(0)
        self.result_table.setMinimumHeight(120)
        self.result_table.setMaximumHeight(320)
        self.result_table_scroll = QScrollArea()
        self.result_table_scroll.setWidgetResizable(True)
        self.result_table_scroll.setWidget(self.result_table)
        self.result_table_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.result_table_scroll.setMinimumWidth(0)
        self.result_table_scroll.setMinimumHeight(120)
        self.result_table_scroll.setMaximumHeight(320)

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
        controls.addWidget(QLabel("区域 top%："))
        controls.addWidget(self.region_top_spin)
        self.heatmap_checkbox = QCheckBox("热力图叠加")
        self.heatmap_checkbox.setToolTip(
            "候选区域图与最近邻原图：勾选后显示原图叠加热力图的混合图，"
            "取消则显示原图；切换时已绘制的多边形/虚线框保持显示"
        )
        controls.addWidget(self.heatmap_checkbox)
        self.guided_checkbox = QCheckBox("边缘贴合(导向滤波)")
        self.guided_checkbox.setToolTip(
            "利用原图边缘高频信息引导热力图，使热力图边缘紧密贴合划痕物理轮廓"
        )
        controls.addWidget(self.guided_checkbox)
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
        self.file_tree.setColumnWidth(0, 300)
        self.file_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.file_tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.file_tree_scroll = QScrollArea()
        self.file_tree_scroll.setWidgetResizable(True)
        self.file_tree_scroll.setWidget(self.file_tree)
        self.file_tree_scroll.setFrameShape(QFrame.Shape.NoFrame)
        file_layout.addWidget(self.file_tree_scroll, 1)
        file_panel.setMaximumWidth(560)
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(
            self._file_tree_context_menu
        )

        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        self.raw_panel_label = QLabel("原始区域")
        raw_layout.addWidget(self.raw_panel_label)
        self.raw_canvas = ImageCanvas(editable=False)
        raw_layout.addWidget(self.raw_canvas, 1)

        candidate_panel = QWidget()
        candidate_layout = QVBoxLayout(candidate_panel)
        self.candidate_panel_label = QLabel("候选区域")
        candidate_layout.addWidget(self.candidate_panel_label)
        candidate_layout.addWidget(self.left_canvas, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("最近邻原图 / 对应 ROI（良品库绿色，异常库红色）"))
        right_layout.addWidget(self.right_canvas, 1)
        right_layout.addWidget(self.result_table_scroll)

        adjust_panel = QWidget()
        adjust_layout = QVBoxLayout(adjust_panel)
        self.adjust_panel_label = QLabel("两阶段调整结果")
        adjust_layout.addWidget(self.adjust_panel_label)
        adjust_layout.addWidget(self.adjust_canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(file_panel)
        splitter.addWidget(raw_panel)
        splitter.addWidget(candidate_panel)
        splitter.addWidget(right_panel)
        splitter.addWidget(adjust_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 1)
        splitter.setStretchFactor(4, 1)
        splitter.setSizes([320, 430, 430, 430, 430])
        self.image_splitter = splitter

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

        self.library_view_tab = LibraryPatchTab(args)
        self.tabs = QTabWidget()
        query_tab = QWidget()
        query_layout = QVBoxLayout(query_tab)
        query_layout.addLayout(controls)
        query_layout.addWidget(bottom_splitter, 1)
        query_layout.addWidget(self.status_label)
        self.tabs.addTab(query_tab, "ROI 反查")
        self.tabs.addTab(self.library_view_tab, "建库 Patch 查看")
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.tabs)
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
        self.region_top_spin.valueChanged.connect(self._region_top_ratio_changed)
        self.heatmap_checkbox.toggled.connect(self._apply_display_mode)
        self.guided_checkbox.toggled.connect(self._apply_display_mode)
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

    def _predictor_process_size(self) -> int:
        """Read ``process_size`` from run.json, as recorded by the predictor."""

        run_path = self._preds_dir() / "run.json"
        if not run_path.is_file():
            return 0
        try:
            with run_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            return int(config.get("process_size", 0) or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0

    def _predictor_region_score(self, roi_mask: np.ndarray) -> float:
        """Compute the region score exactly like dinomaly_two_threshold_predict.py.

        The predictor scores the ROI in ``process_size`` space (when set):
        its score map is downsampled there and the top-ratio mean is taken
        over the mask pixels.  The GUI repeats the same downsampling on both
        the score map and the image-space mask so the displayed value matches
        the prediction.
        """

        if self.score_map is None or not np.any(roi_mask):
            return 0.0
        region_ratio = float(self.region_top_spin.value()) / 100.0
        process_size = self._predictor_process_size()
        if process_size > 0:
            score = cv2.resize(
                self.score_map,
                (process_size, process_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = (
                cv2.resize(
                    np.asarray(roi_mask, dtype=np.uint8),
                    (process_size, process_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        else:
            score = self.score_map
            mask = roi_mask
        values = np.asarray(score, dtype=np.float32)[mask]
        top_count = max(1, int(values.size * region_ratio))
        return float(np.sort(values)[-top_count:].mean())

    def _library_metadata(self) -> Dict[str, Any]:
        root = self._preds_dir().parent
        metadata_path = root / "good" / "metadata.json"
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _index_is_ip(self) -> bool:
        """True when the feature libraries use ``IndexFlatIP``."""

        metadata = self._library_metadata()
        return (
            str(metadata.get("index_type", "")).casefold()
            in {"indexflatip", "ip"}
        )

    @staticmethod
    def _backbone_patch_size(backbone: str) -> int:
        name = str(backbone).casefold()
        for token, size in (("_8", 8), ("_16", 16), ("_14", 14)):
            if token in name:
                return size
        return 14

    def _max_patch_boxes(self) -> List[List[float]]:
        """Return the query Patch bbox recorded by ``query_feature_library``.

        The query process is the source of truth for the selected feature
        cell.  Re-selecting a cell in the GUI can use a different score-map
        geometry and was the reason for visibly misplaced blue boxes.
        """

        metadata = self._library_metadata()
        if str(metadata.get("library_mode", "roi")) != "patch":
            return []
        if self.left_canvas.image is None:
            return []
        if not self.results:
            return []
        index = self._queried_candidate_index
        if index is None or self.left_canvas.selected_candidate_index != index:
            return []
        if not 0 <= index < len(self.left_canvas.candidate_regions):
            return []
        result = next(
            (
                item
                for item in self.results
                if isinstance(
                    item.get("query_patch_bbox_original"),
                    (list, tuple),
                )
                and len(item.get("query_patch_bbox_original")) == 4
            ),
            None,
        )
        if result is None:
            return []
        return [
            [float(value) for value in result["query_patch_bbox_original"]]
        ]

    def _update_patch_boxes(self) -> None:
        self.left_canvas.set_patch_rects(self._max_patch_boxes())

    def _update_right_patch_box(self, result: Mapping[str, Any]) -> None:
        """Mark the matched library patch on the nearest-neighbour canvas.

        In patch-mode libraries every stored vector is one patch, whose
        feature-space bbox (``bbox_feature``) is converted back to the
        library image's original coordinates and drawn as a blue dashed box.
        """

        self.right_canvas.clear_patch_boxes()
        metadata = self._library_metadata()
        if str(metadata.get("library_mode", "roi")) != "patch":
            return
        if self.right_canvas.image is None:
            return
        bbox_original = result.get("patch_bbox_original")
        if not isinstance(bbox_original, (list, tuple)) or len(bbox_original) != 4:
            return
        self.right_canvas.set_patch_rects(
            [[float(value) for value in bbox_original]]
        )

    def _update_raw_score_label(self, image_path: Path) -> None:
        raw_text = "—"
        details_path = self._prediction_details_path(image_path)
        if details_path is not None:
            try:
                with details_path.open("r", encoding="utf-8") as file:
                    detail = json.load(file)
                raw_score = float(detail.get("raw_score", float("nan")))
                if np.isfinite(raw_score):
                    raw_text = f"{raw_score:.4f}"
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        self.raw_panel_label.setText(
            f"原始区域　原始图像分数={raw_text}"
        )

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
            self._queried_candidate_index = None
            self.left_canvas.clear_patch_boxes()
            self._update_raw_score_label(image_path)
            self._update_two_stage_panel()
            self._update_selected_region_calculation()
            if candidate_path is not None or self.left_canvas.candidate_regions:
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
            self._apply_display_mode()
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "打开失败", str(error))

    def fit_all_canvases(self) -> None:
        """Reset zoom/pan on every image view to fit the window."""

        self._fit_image_splitter()
        for canvas in (
            self.left_canvas,
            self.raw_canvas,
            self.right_canvas,
            self.adjust_canvas,
        ):
            canvas.fit_to_window()
        self.library_view_tab.fit_all()

    @staticmethod
    def _heatmap_blend(
        image_bgr: np.ndarray,
        score_map: np.ndarray,
        guided: bool = False,
    ) -> np.ndarray:
        """Overlay a jet-coloured score map on an image copy."""

        score_map = np.asarray(score_map, dtype=np.float32)
        if score_map.shape[:2] != image_bgr.shape[:2]:
            score_map = cv2.resize(
                score_map,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if guided:
            score_map = refine_anomaly_map_guided(image_bgr, score_map, radius=6, eps=1e-3)
        score_min = float(score_map.min())
        score_max = float(score_map.max())
        if score_max - score_min < 1e-12:
            score_norm = np.zeros_like(score_map, dtype=np.uint8)
        else:
            score_norm = (
                (score_map - score_min) / (score_max - score_min) * 255.0
            ).astype(np.uint8)
        heat = cv2.applyColorMap(score_norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 0.6, heat, 0.4, 0)

    def _apply_canvas_display(
        self,
        canvas: ImageCanvas,
        image_path: Optional[Path],
        score_map: Optional[np.ndarray],
    ) -> None:
        """Re-render one canvas in the current heatmap mode, keeping overlays.

        The candidate regions, patch rects and overlay bbox are captured,
        the base image is swapped (original or heatmap blend), then every
        overlay is restored so toggling the switch never drops annotations.
        """

        if canvas.image is None or image_path is None:
            return
        regions = list(canvas.candidate_regions)
        selected_index = canvas.selected_candidate_index
        patch_rects = [
            [
                float(rect.left()),
                float(rect.top()),
                float(rect.right()),
                float(rect.bottom()),
            ]
            for rect in canvas.patch_rects
        ]
        overlay_bbox = canvas.overlay_bbox
        overlay_text = canvas.overlay_text
        overlay_color = QColor(canvas.overlay_color)
        overlay_dashed = canvas.overlay_dashed
        zoom = canvas.zoom
        pan = canvas.pan
        if (
            self.heatmap_checkbox.isChecked()
            and score_map is not None
            and bool(np.any(score_map))
        ):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is not None:
                canvas.set_numpy_image(
                    cv2.cvtColor(
                        self._heatmap_blend(
                            image_bgr,
                            score_map,
                            guided=self.guided_checkbox.isChecked(),
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            else:
                canvas.set_image(image_path)
        else:
            canvas.set_image(image_path)
        if regions:
            canvas.set_candidate_regions(regions, emit=False)
        canvas.selected_candidate_index = selected_index
        if patch_rects:
            canvas.set_patch_rects(patch_rects)
        if overlay_bbox is not None:
            canvas.set_overlay_bbox(
                overlay_bbox,
                overlay_text,
                overlay_color,
                overlay_dashed,
            )
        # set_numpy_image 会把 image_path 置空，这里恢复，否则下次
        # 切换显示模式时 _apply_display_mode 会因 image_path 为 None 而跳过。
        canvas.image_path = image_path
        # set_image/set_numpy_image 会重置视图缩放与平移；
        # 模式切换不应丢失当前视图，这里恢复。
        canvas.zoom = zoom
        canvas.pan = pan
        canvas.update()

    def _apply_display_mode(self) -> None:
        """Re-render the candidate and nearest-neighbour canvases.

        The nearest-neighbour image uses its own cached score map when one
        exists (test-set images); library build images have none and stay as
        the original image.
        """

        if self.left_canvas.image_path is not None:
            self._apply_canvas_display(
                self.left_canvas,
                self.left_canvas.image_path,
                self.score_map,
            )
        if self.right_canvas.image_path is not None:
            right_score = None
            try:
                right_score = self.load_score_map(self.right_canvas.image_path)
            except (OSError, ValueError, RuntimeError):
                right_score = None
            self._apply_canvas_display(
                self.right_canvas,
                self.right_canvas.image_path,
                right_score,
            )

    def _fit_image_splitter(self) -> None:
        """Redistribute the file list and four image panels to the window."""

        splitter = getattr(self, "image_splitter", None)
        if splitter is None or splitter.width() <= 0:
            return
        count = splitter.count()
        if count != 5:
            return
        handle_space = splitter.handleWidth() * (count - 1)
        available = max(splitter.width() - handle_space, 0)
        file_width = min(320, max(240, int(available * 0.15)))
        image_width = max(240, (available - file_width) // 4)
        sizes = [file_width] + [image_width] * 4
        sizes[-1] += max(0, available - sum(sizes))
        splitter.setSizes(sizes)

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

    def _region_top_ratio_changed(self, value: float) -> None:
        ratio = float(value) / 100.0
        if not 0.0 < ratio <= 1.0:
            return
        setattr(self.args, "region_top_ratio", ratio)
        self._update_selected_region_calculation()

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
        candidate_judged = []
        final_judged = []
        unmatched_count = 0
        for region_data in self.left_canvas.candidate_regions:
            candidate = dict(region_data)
            candidate["color"] = candidate.get("color", "#00bcd4")
            candidate["label"] = candidate.get("label")
            candidate["is_judged"] = False
            match = None
            if not candidate.get("is_annotation"):
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
                candidate["label"] = "GOOD" if label == "good" else "Anomaly"
                candidate["score"] = adjusted
                candidate["is_judged"] = True
            else:
                if not candidate.get("is_annotation"):
                    unmatched_count += 1
            candidate_judged.append(candidate)
            # The adjustment canvas remains an anomaly-only final view;
            # the candidate canvas above intentionally retains GOOD ROIs for
            # manual GOOD/Anomaly library retrieval.
            if (
                not candidate.get("is_annotation")
                and candidate.get("label") != "GOOD"
            ):
                final_judged.append(candidate)
        self.unmatched_region_count = unmatched_count
        # 候选面板保持候选区域青色/标注红色的着色（load_candidate_regions），
        # 不把二阶段判定结果写回；二阶段着色只用于下方调整结果面板。
        self.adjust_canvas.set_candidate_regions(final_judged, emit=False)
        judged_with_score = [
            candidate
            for candidate in final_judged
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

        # 整图最终结果：优先读取 details 记录的 final_label
        adjusted_score = float(detail.get("adjusted_score", raw_score))
        label = str(detail.get("final_label", ""))
        if not label:
            label, _reason = final_score_label(
                adjusted_score,
                good_threshold,
                anomaly_threshold,
            )
        label_cn = "正常" if label == "good" else "异常"
        color = "#00c853" if label == "good" else "#ff1744"
        self.adjust_panel_label.setText(
            f"两阶段调整结果（整图最终：{label_cn}，"
            f"adjusted={adjusted_score:.4f}）"
        )
        self.adjust_panel_label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
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
            adjusted_score = float(detail.get("adjusted_score", raw_score))
            label, _reason = final_score_label(
                adjusted_score,
                good_threshold,
                anomaly_threshold,
            )
            final_cn = "正常" if label == "good" else "异常"
            region_count = len(regions)
            lines.append(
                f"整图最终调整（{region_count} 个 ROI 结果覆写后 "
                f"score map 的 top 1% 均值）：adjusted_score = "
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
            region_ratio = float(self.region_top_spin.value()) / 100.0
            region_score = self._predictor_region_score(roi_mask)
            if self._index_is_ip():
                lines.append(
                    f"region_score（ROI 内 top {region_ratio * 100.0:g}% 均值，"
                    "仅参考，IP 库实际按近库固定档位） = "
                    f"<b>{region_score:.4f}</b>"
                )
            else:
                lines.append(
                    f"region_score（ROI 内 top {region_ratio * 100.0:g}% 均值） = "
                    f"<b>{region_score:.4f}</b>"
                )
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
                    good_threshold=good_threshold,
                    anomaly_threshold=anomaly_threshold,
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
        if self._index_is_ip():
            # IP 索引：内积距离无界，无 offset 修正，按近库取固定档位。
            # region_score 直接采用预测写入的固定档位值。
            stored_score = float(region.get("region_score", 0.0))
            if anomaly_distance < good_distance:
                band_text = (
                    f"d_anomaly &lt; d_good（近异常库）→ "
                    f"region_score = 1.5 × anomaly_threshold = "
                    f"<b>{1.5 * anomaly_threshold:.4f}</b>"
                )
            else:
                band_text = (
                    f"d_anomaly ≥ d_good（近良品库）→ "
                    f"region_score = 0.5 × good_threshold = "
                    f"<b>{0.5 * good_threshold:.4f}</b>"
                )
            lines.append(band_text)
            if region_score is None:
                return
            adjusted = stored_score
            lines.append(
                f"区域分数（预测写入）= <b>{stored_score:.4f}</b>"
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
                "adjusted_below_good_threshold": "低于良品阈值",
                "adjusted_above_anomaly_threshold": "高于异常阈值",
                "adjusted_in_middle_band": "介于两阈值之间",
            }.get(reason, reason)
            lines.append(
                f"最终判定：<span style=\"color:{color}; font-weight:bold;\">"
                f"{label_cn}</span>（{reason_cn}）"
            )
            return
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
        data_root_text = getattr(self.args, "data_root", None)
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

        The mask root is derived from the prediction products: it defaults to
        ``<data_root>/ground_truth`` (auto-reconstructed from the details),
        mirroring the ``--data_root`` layout, so an image maps to
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
            # 兜底：图像同目录下的同名 labelme 标注（test/bad/x.jpg 旁常有
            # x.json），即使 GT 目录推导失败也能显示标注异常多边形。
            sibling = image_path.with_suffix(".json")
            if sibling.is_file():
                mask_path = sibling
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
            region["label"] = "Anomaly"
            region["is_annotation"] = True
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
        components = self._mask_components(mask_path, score_map)
        image_area = (
            self.left_canvas.image.width() * self.left_canvas.image.height()
            if self.left_canvas.image is not None
            else 1
        )
        components = [
            component
            for component in components
            if int(component.get("area", 0)) <= 0.9 * image_area
        ]
        self.raw_canvas.set_candidate_regions(components, emit=False)
        return mask_path

    def _load_detail_for(self, image_path: Path) -> Optional[Dict[str, Any]]:
        details_path = self._prediction_details_path(image_path)
        if details_path is None:
            return None
        try:
            with details_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def load_candidate_regions(
        self,
        image_path: Path,
        score_map: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Load one mask and split it into selectable connected components.

        Directly-classified anomaly images have no middle-band candidates; in
        that case the raw threshold regions are shown instead so ROIs can
        still be queried.  The panel title states the image band.
        """

        self.left_canvas.clear_candidate_regions(emit=False)
        detail = self._load_detail_for(image_path) or {}
        band = str(detail.get("initial_label", ""))
        fallback_used = False
        result_path = None
        components = []
        # The candidate panel is the ROI retrieval panel.  Keep middle-band
        # candidates and also expose first-stage regions that never entered
        # stage two, so every useful GOOD/Anomaly ROI can be queried.
        artifact = "candidate_regions" if band == "middle" else "raw_regions"
        root = self._artifact_root(artifact)
        if root is not None:
            result_path = self._artifact_path(root, image_path, ".png")
        if result_path is not None:
            components = self._mask_components(result_path, score_map)
        if not components and band == "middle":
            raw_root = self._artifact_root("raw_regions")
            if raw_root is not None:
                raw_mask_path = self._artifact_path(raw_root, image_path, ".png")
                if raw_mask_path is not None:
                    raw_components = self._mask_components(raw_mask_path, score_map)
                    if raw_components:
                        components = raw_components
                        result_path = raw_mask_path
                        fallback_used = True
        good_threshold = float(self.args.good_threshold)
        anomaly_threshold = float(self.args.anomaly_threshold)
        image_area = (
            self.left_canvas.image.width() * self.left_canvas.image.height()
            if self.left_canvas.image is not None
            else 1
        )
        filtered: List[Dict[str, Any]] = []
        for component in components:
            if component.get("is_annotation"):
                # 标注区域固定红色，与 Dinomaly2 预测候选的青色区分。
                component.setdefault("color", "#ff1744")
                filtered.append(component)
                continue
            # 背景高分连片会把整图变成一个连通域；仅过滤接近整图的区域。
            if int(component.get("area", 0)) > 0.9 * image_area:
                continue
            # Dinomaly2 预测的候选区域统一青色，不再按分数分档着色，
            # 避免与标注（红色）混淆。
            component["label"] = None
            component["color"] = "#00bcd4"
            filtered.append(component)
        components = filtered

        # Reuse the annotation regions already loaded on the raw canvas.
        # Do not read --mask_dir again here or create a second annotation
        # loading path for the candidate panel.
        for region in self.raw_canvas.candidate_regions:
            if region.get("is_annotation"):
                components.append(dict(region))

        self.left_canvas.set_candidate_regions(components, emit=False)
        self._update_candidate_band_label(band, fallback_used)
        return result_path

    def _update_candidate_band_label(self, band: str, fallback_used: bool) -> None:
        self.candidate_panel_label.setText("候选区域")

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
        self._update_patch_boxes()
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

        groups: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
        for image_path in image_paths:
            try:
                relative = image_path.relative_to(self.images_root)
            except ValueError:
                relative = Path(image_path.name)
            group = relative.parts[0] if relative.parts else ""
            raw_score = None
            adjusted_score = None
            initial_label = ""
            final_label = ""
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
                    initial_label = str(detail.get("initial_label", ""))
                    final_label = str(detail.get("final_label", ""))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            groups.setdefault(group, {}).setdefault(initial_label, {}).setdefault(
                final_label, []
            ).append(
                {
                    "path": image_path,
                    "group": group,
                    "raw": raw_score,
                    "adjusted": adjusted_score,
                    "initial_label": initial_label,
                    "final_label": final_label,
                }
            )

        label_order = {"good": 0, "middle": 1, "anomaly": 2, "": 3}
        for group, initial_groups in sorted(groups.items()):
            group_item = QTreeWidgetItem(
                [
                    f"{group}（{sum(len(r) for rows in initial_groups.values() for r in rows.values())}）"
                ]
            )
            self.file_tree.addTopLevelItem(group_item)
            for initial_label, final_groups in sorted(
                initial_groups.items(),
                key=lambda item: label_order.get(item[0], 3),
            ):
                initial_cn = {
                    "good": "正常",
                    "middle": "中间带",
                    "anomaly": "异常",
                }.get(initial_label, "无详情")
                initial_item = QTreeWidgetItem(
                    [
                        f"{initial_cn}（{sum(len(r) for r in final_groups.values())}）"
                    ]
                )
                group_item.addChild(initial_item)
                for final_label, rows in sorted(
                    final_groups.items(),
                    key=lambda item: label_order.get(item[0], 3),
                ):
                    rows.sort(
                        key=lambda row: (
                            row["adjusted"]
                            if row["adjusted"] is not None
                            else float("inf"),
                        )
                    )
                    final_cn = {
                        "good": "正常",
                        "anomaly": "异常",
                    }.get(final_label, "无详情")
                    final_item = QTreeWidgetItem([f"{final_cn}（{len(rows)}）"])
                    initial_item.addChild(final_item)
                    for row in rows:
                        raw_text = (
                            f"{row['raw']:.4f}" if row["raw"] is not None else "—"
                        )
                        adjusted_text = (
                            f"{row['adjusted']:.4f}"
                            if row["adjusted"] is not None
                            else "—"
                        )
                        item = QTreeWidgetItem(
                            [
                                f"{row['path'].name}  raw={raw_text}  "
                                f"adj={adjusted_text}"
                            ]
                        )
                        item.setData(0, Qt.ItemDataRole.UserRole, str(row["path"]))
                        item.setToolTip(0, str(row["path"]))
                        final_item.addChild(item)
                        row["item"] = item
                        self._file_rows.append(row)
                    final_item.setExpanded(True)
                initial_item.setExpanded(True)
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
            for child_index in range(group_item.childCount()):
                initial_item = group_item.child(child_index)
                if initial_item.childCount() == 0:
                    continue
                for final_index in range(initial_item.childCount()):
                    final_item = initial_item.child(final_index)
                    if final_item.childCount() == 0:
                        continue
                    visible = any(
                        not final_item.child(leaf).isHidden()
                        for leaf in range(final_item.childCount())
                    )
                    final_item.setHidden(not visible)
                initial_visible = any(
                    not initial_item.child(fi).isHidden()
                    for fi in range(initial_item.childCount())
                )
                initial_item.setHidden(not initial_visible)
            if group_item.childCount() == 0:
                continue
            group_visible = any(
                not group_item.child(ci).isHidden()
                for ci in range(group_item.childCount())
            )
            group_item.setHidden(not group_visible)

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

    def _file_tree_context_menu(self, position) -> None:
        item = self.file_tree.itemAt(position)
        if item is None:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        menu = QMenu(self.file_tree)
        copy_action = menu.addAction("复制路径")
        chosen = menu.exec(self.file_tree.viewport().mapToGlobal(position))
        if chosen == copy_action:
            QApplication.clipboard().setText(str(path))
            self.status_label.setText(f"已复制：{path}")

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
            self._queried_candidate_index = (
                self.left_canvas.selected_candidate_index
            )
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
        self._update_patch_boxes()
        self._update_selected_region_calculation()
        self._fit_image_splitter()

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
                self._update_right_patch_box(result)
                # 只重渲染最近邻画布（按热力图开关），不动候选区域图，
                # 避免其视图/多边形被重置。
                right_score = None
                try:
                    right_score = self.load_score_map(source_path)
                except (OSError, ValueError, RuntimeError):
                    right_score = None
                self._apply_canvas_display(
                    self.right_canvas,
                    source_path,
                    right_score,
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


class LibraryPatchTab(QWidget):
    """Inspect library build images and the patches stored in the libraries.

    The good/ and anomaly/ libraries written by dinomaly_two_stage.py record,
    for every stored vector, the source image, the annotation mask and (in
    patch mode) the feature-grid patch position.  For each build image this
    tab renders three views: the original image, the mask (background 0 /
    foreground 255) with the stored patches drawn as blue dashed boxes, and
    the original image with the mask's foreground polygons and the same
    patch boxes.
    """

    def __init__(self, args, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.args = args
        self.entries: List[Dict[str, Any]] = []
        self._build_ui()
        self._reload_entries()

    def _library_root(self) -> Path:
        return Path(self.args.preds).expanduser().parent

    @staticmethod
    def _read_library_records(library_dir: Path) -> List[Dict[str, Any]]:
        """Read build records from ``metadata.json`` (fallback: id_mapping)."""

        metadata_path = library_dir / "metadata.json"
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                records = json.load(file).get("records", [])
            if records:
                return records
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        mapping_path = library_dir / "id_mapping.json"
        try:
            with mapping_path.open("r", encoding="utf-8") as file:
                return json.load(file).get("records", [])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    @staticmethod
    def _entry_key(image_path: Any) -> str:
        return str(Path(str(image_path)).expanduser())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.info_label = QLabel("正在读取建库记录……")
        self.info_label.setWordWrap(True)
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self.fit_all)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.info_label, 1)
        toolbar.addWidget(fit_button)

        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderHidden(True)
        self.library_tree.setColumnWidth(0, 300)
        self.library_tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.library_tree.itemClicked.connect(self._tree_item_clicked)
        self.library_tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.library_tree.customContextMenuRequested.connect(
            self._library_tree_context_menu
        )
        tree_scroll = QScrollArea()
        tree_scroll.setWidgetResizable(True)
        tree_scroll.setWidget(self.library_tree)
        tree_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.raw_canvas = ImageCanvas(editable=False)
        self.blend_canvas = ImageCanvas(editable=False)

        def panel(title: str, canvas: ImageCanvas) -> QWidget:
            panel_widget = QWidget()
            panel_layout = QVBoxLayout(panel_widget)
            panel_layout.addWidget(QLabel(title))
            panel_layout.addWidget(canvas, 1)
            return panel_widget

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tree_scroll)
        splitter.addWidget(panel("原始图像", self.raw_canvas))
        splitter.addWidget(
            panel("混合图", self.blend_canvas)
        )
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([300, 700, 700])
        layout.addLayout(toolbar)
        layout.addWidget(splitter, 1)

    def _reload_entries(self) -> None:
        self.entries = []
        self.library_tree.clear()
        for library_type in ("good", "anomaly"):
            for record in self._read_library_records(
                self._library_root() / library_type
            ):
                image_path = record.get("image_path")
                if not image_path:
                    continue
                key = self._entry_key(image_path)
                entry = next(
                    (entry for entry in self.entries if entry["key"] == key),
                    None,
                )
                if entry is None:
                    entry = {
                        "key": key,
                        "path": Path(str(image_path)),
                        "libraries": [],
                        "records": [],
                    }
                    self.entries.append(entry)
                if library_type not in entry["libraries"]:
                    entry["libraries"].append(library_type)
                entry["records"].append((library_type, record))
        for entry in sorted(
            self.entries,
            key=lambda item: str(item["path"]).casefold(),
        ):
            item = QTreeWidgetItem([entry["path"].name])
            item.setData(0, Qt.ItemDataRole.UserRole, entry["key"])
            item.setToolTip(0, str(entry["path"]))
            self.library_tree.addTopLevelItem(item)
        if self.entries:
            self.info_label.setText(
                f"共 {len(self.entries)} 张建库图像（来自 good/anomaly 库 metadata）"
            )
        else:
            self.info_label.setText(
                "未找到建库记录：请检查 --preds 上级目录的 good/anomaly/metadata.json"
            )

    def _library_metadata(self, library_type: str = "good") -> Dict[str, Any]:
        metadata_path = self._library_root() / library_type / "metadata.json"
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        entry = next(
            (entry for entry in self.entries if entry["key"] == key),
            None,
        )
        if entry is not None:
            self._show_entry(entry)

    def _library_tree_context_menu(self, position) -> None:
        item = self.library_tree.itemAt(position)
        if item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        entry = next(
            (entry for entry in self.entries if entry["key"] == key),
            None,
        )
        if entry is None:
            return
        menu = QMenu(self.library_tree)
        copy_image = menu.addAction("复制图像路径")
        copy_mask = menu.addAction("复制 Mask 路径")
        chosen = menu.exec(self.library_tree.viewport().mapToGlobal(position))
        if chosen == copy_image:
            QApplication.clipboard().setText(str(entry["path"]))
            self.info_label.setText(f"已复制图像路径：{entry['path']}")
        elif chosen == copy_mask:
            mask_path = next(
                (
                    str(record["mask_path"])
                    for _library_type, record in entry["records"]
                    if record.get("mask_path")
                ),
                None,
            )
            if mask_path:
                QApplication.clipboard().setText(mask_path)
                self.info_label.setText(f"已复制 Mask 路径：{mask_path}")

    @staticmethod
    def _draw_mask_outline(
        blend_bgr: np.ndarray,
        mask: np.ndarray,
        color_bgr: Tuple[int, int, int],
    ) -> np.ndarray:
        """Draw one mask's foreground polygon outlines onto an image copy."""

        contours, _ = cv2.findContours(
            (np.asarray(mask) > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            cv2.drawContours(blend_bgr, contours, -1, color_bgr, 2)
        return blend_bgr

    def _load_record_mask(
        self,
        mask_path: Any,
        image_shape: Tuple[int, int],
        library_type: str,
    ) -> Optional[np.ndarray]:
        """Load one library's mask; Labelme JSONs are routed by label."""

        if not mask_path or not Path(str(mask_path)).is_file():
            return None
        mask_path = Path(str(mask_path))
        if mask_path.suffix.lower() == ".json":
            return load_labelme_library_mask(
                mask_path,
                image_shape,
                library_type,
                good_labels=("good",),
                ignore_labels=("ignore",),
            )
        return load_mask(mask_path, image_shape)

    def _show_entry(self, entry: Mapping[str, Any]) -> None:
        try:
            original = cv2.imread(str(entry["path"]))
            if original is None:
                raise OSError(f"无法读取图像：{entry['path']}")
            height, width = original.shape[:2]
            image_shape = (height, width)
            metadata = self._library_metadata()
            library_mode = str(metadata.get("library_mode", "roi"))
            image_size = int(metadata.get("image_size", 672))
            crop_size = int(metadata.get("crop_size", 672))
            metadata_feature_shape = metadata.get("feature_shape")
            if (
                isinstance(metadata_feature_shape, (list, tuple))
                and len(metadata_feature_shape) == 2
            ):
                metadata_feature_shape = tuple(
                    int(value) for value in metadata_feature_shape
                )
            else:
                metadata_feature_shape = None

            good_mask_path = None
            anomaly_mask_path = None
            for library_type, record in entry["records"]:
                mask_path = record.get("mask_path")
                if not mask_path:
                    continue
                if library_type == "good" and good_mask_path is None:
                    good_mask_path = mask_path
                elif library_type == "anomaly" and anomaly_mask_path is None:
                    anomaly_mask_path = mask_path
            good_mask = self._load_record_mask(
                good_mask_path,
                image_shape,
                "good",
            )
            anomaly_mask = self._load_record_mask(
                anomaly_mask_path,
                image_shape,
                "anomaly",
            )

            patch_rects: List[List[float]] = []
            good_patch_count = 0
            anomaly_patch_count = 0
            if library_mode == "patch":
                for library_type, record in entry["records"]:
                    row = record.get("patch_row")
                    col = record.get("patch_col")
                    if row is None or col is None:
                        continue
                    if library_type == "good":
                        good_patch_count += 1
                    else:
                        anomaly_patch_count += 1
                    patch_bbox = record.get("patch_bbox_original")
                    if not (
                        isinstance(patch_bbox, (list, tuple))
                        and len(patch_bbox) == 4
                    ):
                        record_shape = record.get(
                            "feature_shape",
                            metadata_feature_shape,
                        )
                        if (
                            metadata_feature_shape is None
                            or not isinstance(record_shape, (list, tuple))
                            or len(record_shape) != 2
                        ):
                            continue
                        geometry = feature_patch_geometry(
                            int(row),
                            int(col),
                            tuple(int(value) for value in record_shape),
                            image_shape,
                            image_size,
                            crop_size,
                        )
                        patch_bbox = geometry["bbox_original"]
                    patch_rects.append([float(value) for value in patch_bbox])

            combined = None
            if good_mask is not None or anomaly_mask is not None:
                combined = np.zeros(image_shape, dtype=bool)
                if good_mask is not None:
                    combined |= good_mask
                if anomaly_mask is not None:
                    combined |= anomaly_mask

            outlined = original.copy()
            if good_mask is not None:
                # 建库时若对该库区域做了膨胀，绘制时按同样的膨胀圈数
                # 绘制膨胀后的区域，使多边形范围与建库 patch 选取一致。
                good_dilation = int(
                    self._library_metadata("good").get("region_dilation", 0)
                )
                good_outline_mask = (
                    dilate_mask(good_mask, good_dilation)
                    if good_dilation > 0
                    else good_mask
                )
                outlined = self._draw_mask_outline(
                    outlined,
                    good_outline_mask,
                    (83, 200, 0),
                )
            if anomaly_mask is not None:
                anomaly_dilation = int(
                    self._library_metadata("anomaly").get("region_dilation", 0)
                )
                anomaly_outline_mask = (
                    dilate_mask(anomaly_mask, anomaly_dilation)
                    if anomaly_dilation > 0
                    else anomaly_mask
                )
                outlined = self._draw_mask_outline(
                    outlined,
                    anomaly_outline_mask,
                    (68, 23, 255),
                )

            self.raw_canvas.clear_image()
            self.blend_canvas.clear_image()
            if combined is not None:
                outlined_rgb = cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB)
                self.raw_canvas.set_numpy_image(outlined_rgb)
                self.blend_canvas.set_numpy_image(outlined_rgb)
                if patch_rects:
                    self.blend_canvas.set_patch_rects(patch_rects)
            else:
                self.raw_canvas.set_image(entry["path"])
                self.blend_canvas.set_numpy_image(
                    cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                )
            mode_text = "patch" if library_mode == "patch" else "roi"
            if good_patch_count or anomaly_patch_count:
                patch_hint = (
                    f"；正常 patch {good_patch_count} 个，"
                    f"异常 patch {anomaly_patch_count} 个"
                )
            else:
                patch_hint = "（roi 模式或无 patch 记录，无 patch 框）"
            self.info_label.setText(
                f"{entry['path']}　库模式={mode_text}　"
                f"patch_top_ratio="
                f"{float(metadata.get('patch_top_ratio', 0.5)) * 100.0:.0f}%　"
                f"image_size={image_size}　crop_size={crop_size}　"
                f"backbone={metadata.get('backbone', '')}{patch_hint}"
            )
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "查看失败", str(error))

    def fit_all(self) -> None:
        for canvas in (self.raw_canvas, self.blend_canvas):
            canvas.fit_to_window()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PySide6 GUI for viewing Dinomaly2 threshold/candidate regions, "
            "querying ROI feature libraries, and inspecting library build "
            "patches"
        )
    )
    parser.add_argument(
        "--preds",
        required=True,
        help=(
            "Output directory of dinomaly_two_threshold_predict.py "
            "(<root>/preds/); the good/ anomaly/ libraries are read from its "
            "parent and all artifacts (score_maps/, raw_regions/, "
            "candidate_regions/, adjusted_candidate_regions/, details/, "
            "run.json) from this directory. The image root is derived from "
            "preds/details/*.json; ground-truth masks default to "
            "<data_root>/ground_truth or <data_root>/../ground_truth."
        ),
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--output_dir", default="./gui_lookup_results")
    return parser


def resolve_ground_truth(data_root: Path) -> Optional[Path]:
    """Locate the ground-truth mask directory for a derived image root.

    Tries ``<data_root>/ground_truth`` first, then the sibling directory
    ``<data_root>/../ground_truth`` (e.g. ``.../leishi_026/ground_truth``
    next to ``.../leishi_026/test``).
    """

    for candidate in (
        data_root / "ground_truth",
        data_root.parent / "ground_truth",
    ):
        if candidate.is_dir():
            return candidate
    return None


def derive_data_root(preds: Path) -> Optional[Path]:
    """Reconstruct the image root used by dinomaly_two_threshold_predict.py.

    Every ``preds/details/*.json`` stores the absolute ``image_path`` and its
    path relative to the data root; subtracting the relative parts yields the
    root again, so the GUI needs no separate ``--data_root`` argument.
    """

    details_root = Path(preds).expanduser() / "details"
    if not details_root.is_dir():
        return None
    first_candidate: Optional[Path] = None
    for detail_path in details_root.rglob("*.json"):
        try:
            with detail_path.open("r", encoding="utf-8") as file:
                detail = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        image_path = detail.get("image_path")
        image_relative = detail.get("image_relative")
        if not image_path or not image_relative:
            continue
        candidate = Path(str(image_path)).expanduser()
        for _part in Path(str(image_relative)).parts:
            candidate = candidate.parent
        if first_candidate is None:
            first_candidate = candidate
        if candidate.is_dir():
            return candidate
    return first_candidate


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
        ("region_top_ratio", 0.10),
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
        "region_top_ratio",
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
    data_root = derive_data_root(preds)
    if data_root is not None:
        args.data_root = str(data_root)
        ground_truth = resolve_ground_truth(data_root)
        if ground_truth is not None:
            args.mask_dir = str(ground_truth)
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
