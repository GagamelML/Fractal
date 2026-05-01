"""Video playback with queue, monitor selection and corner-pin (homography).

Opened from the Video tab via the "Video Playback…" button.

Components
----------
* :class:`PlaybackDialog`   – queue / monitor / corner-pin / start UI
* :class:`CornerPinWidget`  – interactive 4-corner editor
* :class:`PlaybackWindow`   – fullscreen, frameless playback surface
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # OpenCV: video decode + warpPerspective
except ImportError as e:  # pragma: no cover
    cv2 = None
    _CV2_ERR = e
else:
    _CV2_ERR = None

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QGuiApplication,
    QKeyEvent, QMouseEvent, QPaintEvent,
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem, QComboBox, QCheckBox,
    QFileDialog, QLabel, QWidget, QMessageBox, QSizePolicy,
)

from .common import PROJECT_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Corner-pin editor
# ─────────────────────────────────────────────────────────────────────────────
class CornerPinWidget(QWidget):
    """Editor for 4 destination corners stored as normalised (0..1) coords.

    Order of corners: TL, TR, BR, BL.  Drag any corner with the mouse.
    A "Reset" call returns to a full-screen rectangle.
    """

    HANDLE_R = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Normalised corner positions, TL TR BR BL
        self.corners: List[QPointF] = [
            QPointF(0.0, 0.0), QPointF(1.0, 0.0),
            QPointF(1.0, 1.0), QPointF(0.0, 1.0),
        ]
        self._screen_size = (1920, 1080)  # informational
        self._dragging: Optional[int] = None

    # ── public API ──────────────────────────────────────────────────────────
    def set_screen_size(self, w: int, h: int):
        self._screen_size = (w, h)
        self.update()

    def reset_corners(self):
        self.corners = [
            QPointF(0.0, 0.0), QPointF(1.0, 0.0),
            QPointF(1.0, 1.0), QPointF(0.0, 1.0),
        ]
        self.update()

    def get_destination_points(self) -> np.ndarray:
        """Return 4×2 array of dest. corners in *screen* pixel coords (TL,TR,BR,BL)."""
        sw, sh = self._screen_size
        return np.array([[c.x() * sw, c.y() * sh] for c in self.corners],
                        dtype=np.float32)

    # ── geometry helpers ────────────────────────────────────────────────────
    def _canvas_rect(self) -> QRectF:
        """Letterboxed rectangle (preserving screen aspect) inside the widget."""
        sw, sh = self._screen_size
        margin = 16
        avail_w = self.width() - 2 * margin
        avail_h = self.height() - 2 * margin
        if avail_w <= 0 or avail_h <= 0 or sw <= 0 or sh <= 0:
            return QRectF(0, 0, max(1, self.width()), max(1, self.height()))
        scale = min(avail_w / sw, avail_h / sh)
        w = sw * scale
        h = sh * scale
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        return QRectF(x, y, w, h)

    def _norm_to_widget(self, p: QPointF) -> QPointF:
        r = self._canvas_rect()
        return QPointF(r.x() + p.x() * r.width(), r.y() + p.y() * r.height())

    def _widget_to_norm(self, p: QPointF) -> QPointF:
        r = self._canvas_rect()
        if r.width() <= 0 or r.height() <= 0:
            return QPointF(0, 0)
        nx = (p.x() - r.x()) / r.width()
        ny = (p.y() - r.y()) / r.height()
        return QPointF(max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    # ── painting ────────────────────────────────────────────────────────────
    def paintEvent(self, _e: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(40, 40, 40))

        rect = self._canvas_rect()
        # Screen background (black, so contents look like the playback surface)
        p.fillRect(rect, QColor(0, 0, 0))
        p.setPen(QPen(QColor(90, 90, 90), 1))
        p.drawRect(rect)

        # Quad polygon
        pts = [self._norm_to_widget(c) for c in self.corners]
        p.setPen(QPen(QColor(80, 200, 255), 2))
        p.setBrush(QBrush(QColor(80, 200, 255, 50)))
        from PyQt6.QtGui import QPolygonF
        p.drawPolygon(QPolygonF(pts))

        # Diagonals as a guide
        p.setPen(QPen(QColor(80, 200, 255, 90), 1, Qt.PenStyle.DashLine))
        p.drawLine(pts[0], pts[2])
        p.drawLine(pts[1], pts[3])

        # Handles + labels
        labels = ["TL", "TR", "BR", "BL"]
        for i, wp in enumerate(pts):
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QBrush(QColor(255, 180, 60)))
            p.drawEllipse(wp, self.HANDLE_R, self.HANDLE_R)
            p.setPen(QPen(QColor(230, 230, 230), 1))
            p.drawText(QPointF(wp.x() + 10, wp.y() - 6), labels[i])

        # Info text
        p.setPen(QPen(QColor(200, 200, 200), 1))
        sw, sh = self._screen_size
        p.drawText(8, 16, f"Screen {sw}×{sh} – drag corners to corner-pin")

    # ── mouse interaction ───────────────────────────────────────────────────
    def _hit_test(self, pos: QPointF) -> Optional[int]:
        for i, c in enumerate(self.corners):
            wp = self._norm_to_widget(c)
            if (wp - pos).manhattanLength() <= self.HANDLE_R * 2.5:
                return i
        return None

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = self._hit_test(QPointF(e.position()))

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging is None:
            return
        self.corners[self._dragging] = self._widget_to_norm(QPointF(e.position()))
        self.update()

    def mouseReleaseEvent(self, _e: QMouseEvent):
        self._dragging = None


# ─────────────────────────────────────────────────────────────────────────────
#  Fullscreen playback window
# ─────────────────────────────────────────────────────────────────────────────
class PlaybackWindow(QWidget):
    """Frameless fullscreen window that plays a queue of videos with optional
    homography (corner-pin) and looping.
    """

    def __init__(self, video_paths: List[str], screen, dest_pts: np.ndarray,
                 loop: bool, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet("background-color: black;")

        self._videos = list(video_paths)
        self._loop = loop
        self._idx = 0
        self._dest_pts = dest_pts.astype(np.float32)
        self._cap: Optional["cv2.VideoCapture"] = None
        self._H: Optional[np.ndarray] = None
        self._screen = screen

        geom = screen.geometry()
        self._sw, self._sh = geom.width(), geom.height()

        self._label = QLabel(self)
        self._label.setStyleSheet("background-color: black;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setGeometry(0, 0, self._sw, self._sh)

        # Move to target screen + go fullscreen
        self.setGeometry(geom)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # Pre-allocate canvas
        self._canvas = np.zeros((self._sh, self._sw, 3), dtype=np.uint8)

        if not self._open_current():
            QMessageBox.warning(self, "Playback",
                                "Could not open any of the selected videos.")
            QTimer.singleShot(0, self.close)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def showEvent(self, e):
        super().showEvent(e)
        self.windowHandle().setScreen(self._screen)
        self.showFullScreen()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(e)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _open_current(self) -> bool:
        while self._idx < len(self._videos):
            path = self._videos[self._idx]
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                if self._cap is not None:
                    self._cap.release()
                self._cap = cap
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if fps <= 1 or fps > 240:
                    fps = 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                self._H = cv2.getPerspectiveTransform(src, self._dest_pts)
                interval = max(1, int(round(1000.0 / fps)))
                self._timer.start(interval)
                return True
            self._idx += 1
        return False

    def _advance(self) -> bool:
        self._idx += 1
        if self._idx >= len(self._videos):
            if self._loop:
                self._idx = 0
            else:
                return False
        return self._open_current()

    def _tick(self):
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if not self._advance():
                self.close()
            return

        # Warp into screen-sized black canvas. cv2 expects BGR; we keep it BGR
        # then convert once at the end → Format_BGR888 in QImage.
        warped = cv2.warpPerspective(
            frame, self._H, (self._sw, self._sh),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Convert to QImage. cv2 frames are BGR.
        qimg = QImage(warped.data, self._sw, self._sh,
                      warped.strides[0], QImage.Format.Format_BGR888).copy()
        self._label.setPixmap(QPixmap.fromImage(qimg))


# ─────────────────────────────────────────────────────────────────────────────
#  Setup dialog
# ─────────────────────────────────────────────────────────────────────────────
class PlaybackDialog(QDialog):
    """Configure queue, monitor, corner-pin, then launch fullscreen playback."""

    VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Playback")
        self.resize(900, 600)

        self._playback: Optional[PlaybackWindow] = None

        root = QHBoxLayout(self)

        # ── left column: queue + controls ──────────────────────────────────
        left = QVBoxLayout()
        root.addLayout(left, 1)

        qbox = QGroupBox("Video queue")
        qv = QVBoxLayout(qbox)
        self.list = QListWidget()
        self.list.setSelectionMode(self.list.SelectionMode.ExtendedSelection)
        qv.addWidget(self.list, 1)
        row = QHBoxLayout()
        add_btn = QPushButton("Add…")
        rem_btn = QPushButton("Remove")
        up_btn = QPushButton("↑")
        down_btn = QPushButton("↓")
        clear_btn = QPushButton("Clear")
        for b in (add_btn, rem_btn, up_btn, down_btn, clear_btn):
            row.addWidget(b)
        qv.addLayout(row)
        left.addWidget(qbox, 1)

        add_btn.clicked.connect(self._add_videos)
        rem_btn.clicked.connect(self._remove_selected)
        clear_btn.clicked.connect(self.list.clear)
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        down_btn.clicked.connect(lambda: self._move_selected(+1))

        mbox = QGroupBox("Output")
        mform = QFormLayout(mbox)
        self.monitor = QComboBox()
        self._populate_monitors()
        mform.addRow("Monitor:", self.monitor)
        self.loop_cb = QCheckBox("Loop queue")
        mform.addRow(self.loop_cb)
        left.addWidget(mbox)

        btns = QHBoxLayout()
        reset_btn = QPushButton("Reset Corners")
        self.start_btn = QPushButton("▶  Start Playback")
        self.start_btn.setStyleSheet("padding: 8px; font-weight: bold;")
        btns.addWidget(reset_btn)
        btns.addStretch(1)
        btns.addWidget(self.start_btn)
        left.addLayout(btns)

        # ── right column: corner pin ───────────────────────────────────────
        right = QVBoxLayout()
        root.addLayout(right, 2)
        right.addWidget(QLabel("Corner-pin (drag handles):"))
        self.pin = CornerPinWidget()
        right.addWidget(self.pin, 1)

        # Wire monitor change → update screen size in pin widget
        self.monitor.currentIndexChanged.connect(self._sync_screen_size)
        self._sync_screen_size()

        reset_btn.clicked.connect(self.pin.reset_corners)
        self.start_btn.clicked.connect(self._start)

        if cv2 is None:
            QMessageBox.critical(self, "OpenCV missing",
                                 f"OpenCV (cv2) is required for playback.\n\n{_CV2_ERR}")
            self.start_btn.setEnabled(False)

    # ── monitor handling ───────────────────────────────────────────────────
    def _populate_monitors(self):
        self.monitor.clear()
        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
        for i, s in enumerate(screens):
            geom = s.geometry()
            tag = " (primary)" if s is primary else ""
            self.monitor.addItem(
                f"[{i}] {s.name()} {geom.width()}×{geom.height()}{tag}", s)

    def _selected_screen(self):
        return self.monitor.currentData()

    def _sync_screen_size(self):
        s = self._selected_screen()
        if s is None:
            return
        g = s.geometry()
        self.pin.set_screen_size(g.width(), g.height())

    # ── queue handling ─────────────────────────────────────────────────────
    def _add_videos(self):
        start_dir = os.path.join(PROJECT_DIR, "results")
        if not os.path.isdir(start_dir):
            start_dir = PROJECT_DIR
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", start_dir,
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*)")
        for f in files:
            it = QListWidgetItem(os.path.relpath(f, PROJECT_DIR)
                                 if f.startswith(PROJECT_DIR) else f)
            it.setData(Qt.ItemDataRole.UserRole, f)
            self.list.addItem(it)

    def _remove_selected(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _move_selected(self, delta: int):
        rows = sorted([self.list.row(it) for it in self.list.selectedItems()],
                      reverse=(delta > 0))
        for r in rows:
            new_r = r + delta
            if 0 <= new_r < self.list.count():
                it = self.list.takeItem(r)
                self.list.insertItem(new_r, it)
                it.setSelected(True)

    def _all_paths(self) -> List[str]:
        out = []
        for i in range(self.list.count()):
            out.append(self.list.item(i).data(Qt.ItemDataRole.UserRole))
        return out

    # ── start playback ─────────────────────────────────────────────────────
    def _start(self):
        paths = self._all_paths()
        if not paths:
            QMessageBox.warning(self, "No videos", "Please add at least one video.")
            return
        screen = self._selected_screen()
        if screen is None:
            QMessageBox.warning(self, "No monitor", "No monitor selected.")
            return
        dest = self.pin.get_destination_points()
        try:
            self._playback = PlaybackWindow(
                paths, screen, dest, self.loop_cb.isChecked(), parent=None)
            self._playback.show()
        except Exception as e:
            QMessageBox.critical(self, "Playback error", str(e))

