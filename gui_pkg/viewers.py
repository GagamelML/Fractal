"""Right-hand pane image viewers and the bottom output console."""
import os
import sys
import glob
import re

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QFileDialog, QPlainTextEdit, QComboBox, QLineEdit, QSizePolicy,
    QTabWidget,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators import GENERATORS

from .common import (
    PROJECT_DIR, RESULTS_DIR, PREVIEW_DIR,
    save_preview_params, load_preview_params, next_preview_folder,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Set Selection Viewer – click-to-zoom + preview quick-select
# ═══════════════════════════════════════════════════════════════════════════════
class SetSelectionViewer(QWidget):
    """Image viewer for the Set Selection tab: preview browsing + click-to-zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_panel = None  # set by MainWindow
        layout = QVBoxLayout(self)

        qs_row = QHBoxLayout()
        qs_row.addWidget(QLabel("Previews:"))
        self._qs_container = QHBoxLayout()
        qs_row.addLayout(self._qs_container)
        qs_row.addStretch()
        self.clear_btn = QPushButton("\U0001f5d1 Clear All")
        self.clear_btn.setMaximumWidth(100)
        self.clear_btn.clicked.connect(self._clear_previews)
        qs_row.addWidget(self.clear_btn)
        layout.addLayout(qs_row)

        self.image_label = QLabel("No frames loaded \u2013 use Preview buttons to generate")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background: #1a1a1a; color: #888;")
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label, 1)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.coord_label)

        bot = QHBoxLayout()
        self.prev_btn = QPushButton("\u25c0")
        self.prev_btn.setMaximumWidth(28)
        self.prev_btn.setToolTip("Previous frame")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(lambda: self._step_frame(-1))
        self.next_btn = QPushButton("\u25b6")
        self.next_btn.setMaximumWidth(28)
        self.next_btn.setToolTip("Next frame")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self._step_frame(+1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        bot.addWidget(self.prev_btn)
        bot.addWidget(self.slider, 1)
        bot.addWidget(self.next_btn)
        bot.addWidget(self.frame_label)
        layout.addLayout(bot)

        self.load_params_btn = QPushButton("\U0001f4e5 Load Set Parameters")
        self.load_params_btn.setToolTip(
            "Load the parameters that generated the currently displayed preview\n"
            "back into the Set Selection panel.")
        self.load_params_btn.clicked.connect(self._load_params)
        layout.addWidget(self.load_params_btn)

        self.frame_paths = []
        self._current_pixmap = None
        self._qs_buttons = []
        self._current_scale = None

        self.refresh_quick_select()

    # ── Load saved parameters ──────────────────────────────────────────────
    def _load_params(self):
        if not self.frame_paths or self._set_panel is None:
            return
        folder = os.path.dirname(self.frame_paths[0])
        params = load_preview_params(folder)
        if params is None:
            main_win = self.window()
            main_win.console.append_text(
                f"\u26a0 No params.json found in {folder}\n")
            return

        sp = self._set_panel
        if "generator" in params:
            idx = sp.generator.findText(params["generator"])
            if idx >= 0:
                sp.generator.setCurrentIndex(idx)
        if "kernel" in params:
            idx = sp.kernel.findText(params["kernel"])
            if idx >= 0:
                sp.kernel.setCurrentIndex(idx)
        if "power" in params:
            sp.power.setValue(int(params["power"]))
        if "c_real" in params:
            sp.c_real.setValue(float(params["c_real"]))
        if "c_imag" in params:
            sp.c_imag.setValue(float(params["c_imag"]))
        if "center_real" in params:
            sp.center_real.setValue(float(params["center_real"]))
        if "center_imag" in params:
            sp.center_imag.setValue(float(params["center_imag"]))
        if "scale" in params:
            sp.scale_start.setValue(float(params["scale"]))
        elif "scale_start" in params:
            sp.scale_start.setValue(float(params["scale_start"]))
        if "max_iter" in params:
            sp.max_iter.setValue(int(params["max_iter"]))
        if "bailout" in params:
            sp.bailout.setValue(float(params["bailout"]))
        if "recenter" in params:
            sp.recenter.setChecked(bool(params["recenter"]))
        if params.get("generator") == "flower":
            if "petals" in params and params["petals"] is not None:
                sp.petals.setValue(int(params["petals"]))
            if "mirror_segments" in params and params["mirror_segments"] is not None:
                sp.mirror_segments.setChecked(bool(params["mirror_segments"]))
            if "interest_angle" in params and params["interest_angle"] is not None:
                sp.interest_angle.setValue(float(params["interest_angle"]))
            if "align_north" in params and params["align_north"] is not None:
                idx = sp.align_north.findText(params["align_north"])
                if idx >= 0:
                    sp.align_north.setCurrentIndex(idx)
        if "zoom_factor" in params and sp._image_panel:
            sp._image_panel.zoom_factor.setValue(float(params["zoom_factor"]))
        if hasattr(sp, "apply_extra_constants"):
            sp.apply_extra_constants(params.get("extra_constants") or [])

        main_win = self.window()
        main_win.console.append_text(
            f"\u2713 Loaded parameters from {os.path.basename(folder)}/params.json\n")

    # ── Coordinate mapping ────────────────────────────────────────────────
    def _pixel_to_fractal(self, px, py):
        if self._current_pixmap is None or self._set_panel is None:
            return None
        label_w, label_h = self.image_label.width(), self.image_label.height()
        pm = self._current_pixmap
        scaled = pm.scaled(self.image_label.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        sw, sh = scaled.width(), scaled.height()
        ox = (label_w - sw) / 2
        oy = (label_h - sh) / 2
        ix = px - ox
        iy = py - oy
        if ix < 0 or iy < 0 or ix >= sw or iy >= sh:
            return None
        orig_w, orig_h = pm.width(), pm.height()
        img_x = ix * orig_w / sw
        img_y = iy * orig_h / sh

        from math import sqrt
        w, h = orig_w, orig_h
        factor = sqrt((w / 2.) ** 2 + (h / 2.) ** 2)
        grid_real = (img_x - w / 2) / factor
        grid_imag = (h / 2 - img_y) / factor

        sp = self._set_panel
        scale = self._current_scale if self._current_scale else sp.scale_start.value()
        center = complex(sp.center_real.value(), sp.center_imag.value())

        if sp.recenter.isChecked():
            z = scale * complex(grid_real, grid_imag) + center
        else:
            z = scale * (complex(grid_real, grid_imag) + center)
        return z

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        pos = self.image_label.mapFrom(self, event.pos())
        z = self._pixel_to_fractal(pos.x(), pos.y())
        if z is None or self._set_panel is None:
            return
        sp = self._set_panel
        zoom_factor = sp.manual_zoom_factor.value()
        steps = sp.manual_zoom_steps.value()

        sp.center_real.setValue(z.real)
        sp.center_imag.setValue(z.imag)
        sp.recenter.setChecked(True)

        old_scale = self._current_scale if self._current_scale else sp.scale_start.value()
        new_scale = old_scale * (zoom_factor ** steps)
        sp.scale_start.setValue(round(new_scale, 6))

        if self.frame_paths:
            target_folder = os.path.dirname(self.frame_paths[0])
        else:
            target_folder = None

        main_win = self.window()
        main_win.console.append_text(
            f"\n\U0001f50d Manual zoom: center={z:.7f}, scale={new_scale:.6f} "
            f"(zoom={zoom_factor}^{steps})\n")
        self._run_click_zoom(sp, new_scale, target_folder)

    def _run_click_zoom(self, sp, scale, target_folder):
        import threading
        from time import perf_counter as clock
        size = 500
        main_win = self.window()

        def _log(msg):
            main_win._pending_logs.append(msg)

        gen_name = sp.generator.currentText()
        gen_cls = GENERATORS[gen_name]
        kwargs = dict(
            c=complex(sp.c_real.value(), sp.c_imag.value()),
            center=complex(sp.center_real.value(), sp.center_imag.value()),
            width=size, height=size,
            max_iter=sp.max_iter.value(),
            power=sp.power.value(),
            kernel=sp.kernel.currentText(),
            bailout=sp.bailout.value(),
            recenter=sp.recenter.isChecked(),
        )
        if gen_name == "flower":
            kwargs.update(
                petals=sp.petals.value(),
                mirror_segments=sp.mirror_segments.isChecked(),
                interest_angle=sp.interest_angle.value(),
                align_north=sp.align_north.currentText(),
            )
        mask_path = sp._image_panel.mask_svg.text().strip() if sp._image_panel else ""

        def _do_render():
            try:
                import numpy as np
                from PIL import Image as PILImage

                _log(f"  Rendering {size}\u00d7{size} at scale={scale:.6f}\u2026\n")
                t0 = clock()
                gen = gen_cls(**kwargs)
                mask = None
                if mask_path:
                    from mask import load_svg_mask, radial_fill
                    display_mask, _ = load_svg_mask(mask_path, size, size)
                    mask = radial_fill(display_mask)
                counts = gen.render(scale, mask=mask)
                _log(f"  Render done ({clock()-t0:.2f}s)\n")

                esc = counts < kwargs["max_iter"]
                grey = np.zeros(counts.shape, dtype=np.uint8)
                if esc.any():
                    vals = counts[esc].astype(np.float64)
                    lo, hi = np.percentile(vals, [1, 99])
                    if hi <= lo:
                        hi = lo + 1.0
                    grey[esc] = (np.clip((vals - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

                img = PILImage.fromarray(grey, mode='L')

                if target_folder and os.path.isdir(target_folder):
                    out_dir = target_folder
                else:
                    out_dir, _ = next_preview_folder()

                existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
                existing += sorted(glob.glob(os.path.join(out_dir, "preview_*.png")))
                if existing:
                    nums = []
                    for p in existing:
                        m = re.search(r'(\d+)\.png$', os.path.basename(p))
                        if m:
                            nums.append(int(m.group(1)))
                    next_num = max(nums, default=-1) + 1
                else:
                    next_num = 0

                path = os.path.join(out_dir, f"frame_{next_num:04d}.png")
                img.save(path)
                save_preview_params(out_dir, {
                    "generator": gen_name,
                    "kernel": kwargs["kernel"],
                    "power": kwargs["power"],
                    "c_real": kwargs["c"].real,
                    "c_imag": kwargs["c"].imag,
                    "center_real": kwargs["center"].real,
                    "center_imag": kwargs["center"].imag,
                    "scale": scale,
                    "max_iter": kwargs["max_iter"],
                    "bailout": kwargs["bailout"],
                    "recenter": kwargs["recenter"],
                    "extra_constants": (sp.extra_constants_list()
                                        if hasattr(sp, "extra_constants_list") else []),
                    **({"petals": kwargs.get("petals"),
                        "mirror_segments": kwargs.get("mirror_segments"),
                        "interest_angle": kwargs.get("interest_angle"),
                        "align_north": kwargs.get("align_north")}
                       if gen_name == "flower" else {}),
                })
                _log(f"  Saved {path}\n")
                return path, out_dir
            except Exception as e:
                import traceback
                return f"{e}\n{traceback.format_exc()}", None

        self._click_result = None

        def _thread():
            self._click_result = _do_render()

        t = threading.Thread(target=_thread, daemon=True)
        t.start()

        def _poll():
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(100, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            result = self._click_result
            if isinstance(result[0] if isinstance(result, tuple) else result, str) and result[1] is None:
                main_win.console.append_text(f"\n\u26a0 Click-zoom error: {result[0]}\n")
            elif isinstance(result, tuple):
                path, out_dir = result
                self._load_frames(out_dir)
                self.slider.setValue(self.slider.maximum())
                self.refresh_quick_select()
                main_win.console.append_text(f"\u2713 Frame appended.\n")

        QTimer.singleShot(100, _poll)

    def mouseMoveEvent(self, event):
        pos = self.image_label.mapFrom(self, event.pos())
        z = self._pixel_to_fractal(pos.x(), pos.y())
        if z is not None:
            self.coord_label.setText(f"z = {z.real:+.7f} {z.imag:+.7f}i")
        else:
            self.coord_label.setText("")
        super().mouseMoveEvent(event)

    # ── Frame loading ────────────────────────────────────────────────────
    def refresh_quick_select(self):
        for btn in self._qs_buttons:
            self._qs_container.removeWidget(btn)
            btn.deleteLater()
        self._qs_buttons = []
        if not os.path.isdir(PREVIEW_DIR):
            return
        dirs = sorted(d for d in os.listdir(PREVIEW_DIR)
                      if os.path.isdir(os.path.join(PREVIEW_DIR, d)) and d.isdigit())
        for d in dirs:
            btn = QPushButton(d.lstrip("0") or "0")
            btn.setFixedWidth(36)
            folder = os.path.join(PREVIEW_DIR, d)
            btn.clicked.connect(lambda checked, f=folder: self.load_folder(f))
            self._qs_container.addWidget(btn)
            self._qs_buttons.append(btn)

    def _clear_previews(self):
        import shutil
        if os.path.isdir(PREVIEW_DIR):
            shutil.rmtree(PREVIEW_DIR)
        self.refresh_quick_select()
        self.frame_paths = []
        self.slider.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.frame_label.setText("0 / 0")
        self.image_label.setText("Previews cleared")
        self.image_label.setPixmap(QPixmap())
        self._current_pixmap = None

    def load_folder(self, folder):
        self._load_frames(folder)

    def _load_frames(self, folder=None):
        if folder is None:
            if not self.frame_paths:
                return
            folder = os.path.dirname(self.frame_paths[0])
        if not os.path.isabs(folder):
            folder = os.path.join(PROJECT_DIR, folder)
        frames = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
        previews = sorted(glob.glob(os.path.join(folder, "preview_*.png")))
        self.frame_paths = frames if frames else previews
        n = len(self.frame_paths)
        if n == 0:
            self.image_label.setText(f"No frame_*.png in\n{folder}")
            self.slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.frame_label.setText("0 / 0")
            return
        self.slider.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self._show_frame(0)

    def _step_frame(self, delta):
        if not self.frame_paths:
            return
        new_idx = max(0, min(self.slider.value() + delta, len(self.frame_paths) - 1))
        self.slider.setValue(new_idx)

    def _show_frame(self, idx):
        if not self.frame_paths or idx >= len(self.frame_paths):
            return
        pixmap = QPixmap(self.frame_paths[idx])
        if pixmap.isNull():
            return
        self._current_pixmap = pixmap
        if self._set_panel:
            sp = self._set_panel
            ip = sp._image_panel
            zf = ip.zoom_factor.value() if ip else 0.92
            self._current_scale = sp.scale_start.value() * (zf ** idx)
        self._fit_pixmap()
        self.frame_label.setText(f"{idx + 1} / {len(self.frame_paths)}")

    def _fit_pixmap(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Rendering Viewer – Results Browser
# ═══════════════════════════════════════════════════════════════════════════════
class RenderingViewer(QWidget):
    """Image viewer for the Image Rendering tab: browse results series + colour."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Results:"))
        self.results_folder = QLineEdit(RESULTS_DIR)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._pick_results_folder)
        top.addWidget(self.results_folder, 1)
        top.addWidget(browse_btn)
        layout.addLayout(top)

        series_row = QHBoxLayout()
        series_row.addWidget(QLabel("Series:"))
        self.series_combo = QComboBox()
        self.series_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.series_combo.currentTextChanged.connect(self._on_series_changed)
        series_row.addWidget(self.series_combo, 1)
        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self._refresh_series)
        series_row.addWidget(refresh_btn)
        layout.addLayout(series_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("View:"))
        self._color_btn_container = QHBoxLayout()
        color_row.addLayout(self._color_btn_container)
        color_row.addStretch()
        layout.addLayout(color_row)
        self._color_buttons = []

        self.image_label = QLabel("No series selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background: #1a1a1a; color: #888;")
        layout.addWidget(self.image_label, 1)

        bot = QHBoxLayout()
        self.prev_btn = QPushButton("\u25c0")
        self.prev_btn.setMaximumWidth(28)
        self.prev_btn.setToolTip("Previous frame")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(lambda: self._step_frame(-1))
        self.next_btn = QPushButton("\u25b6")
        self.next_btn.setMaximumWidth(28)
        self.next_btn.setToolTip("Next frame")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self._step_frame(+1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        bot.addWidget(self.prev_btn)
        bot.addWidget(self.slider, 1)
        bot.addWidget(self.next_btn)
        bot.addWidget(self.frame_label)
        layout.addLayout(bot)

        self.load_params_btn = QPushButton("\U0001f4e5 Load Set Parameters")
        self.load_params_btn.setToolTip(
            "Load the parameters that generated the currently selected series\n"
            "back into the Set Selection and Image Rendering panels.")
        self.load_params_btn.clicked.connect(self._load_params)

        self.load_coloring_btn = QPushButton("\U0001f3a8 Load Coloring")
        self.load_coloring_btn.setToolTip(
            "Load the colour program (segments + transitions) that produced\n"
            "the currently displayed colour folder back into the Coloring tab.")
        self.load_coloring_btn.clicked.connect(self._load_coloring)

        load_row = QHBoxLayout()
        load_row.addWidget(self.load_params_btn)
        load_row.addWidget(self.load_coloring_btn)
        layout.addLayout(load_row)

        self.frame_paths = []
        self._current_pixmap = None
        self._set_panel = None
        self._image_panel = None

        QTimer.singleShot(100, self._refresh_series)

    def _pick_results_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select results folder", RESULTS_DIR)
        if path:
            self.results_folder.setText(path)
            self._refresh_series()

    def _refresh_series(self):
        self.series_combo.blockSignals(True)
        current = self.series_combo.currentText()
        self.series_combo.clear()
        results = self.results_folder.text().strip()
        if os.path.isdir(results):
            dirs = sorted(d for d in os.listdir(results)
                          if os.path.isdir(os.path.join(results, d)) and not d.startswith("_"))
            self.series_combo.addItems(dirs)
        idx = self.series_combo.findText(current)
        if idx >= 0:
            self.series_combo.setCurrentIndex(idx)
        self.series_combo.blockSignals(False)
        self._on_series_changed()

    def _on_series_changed(self):
        for btn in self._color_buttons:
            self._color_btn_container.removeWidget(btn)
            btn.deleteLater()
        self._color_buttons = []
        series = self.series_combo.currentText()
        results = self.results_folder.text().strip()
        if not series or not results:
            return
        series_path = os.path.join(results, series)
        if not os.path.isdir(series_path):
            return
        subdirs = sorted(d for d in os.listdir(series_path)
                         if os.path.isdir(os.path.join(series_path, d)) and not d.startswith("_"))
        for name in subdirs:
            btn = QPushButton(name)
            folder = os.path.join(series_path, name)
            btn.clicked.connect(lambda checked, f=folder: self.load_folder(f))
            self._color_btn_container.addWidget(btn)
            self._color_buttons.append(btn)

    def _load_params(self):
        series = self.series_combo.currentText()
        results = self.results_folder.text().strip()
        main_win = self.window()
        if not series or not results:
            return
        folder = os.path.join(results, series)
        meta_path = os.path.join(folder, "_raw", "meta.json")
        if not os.path.isfile(meta_path):
            if hasattr(main_win, "console"):
                main_win.console.append_text(
                    f"\u26a0 No _raw/meta.json in {folder}\n")
            return
        import json
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            if hasattr(main_win, "console"):
                main_win.console.append_text(f"\u26a0 {e}\n")
            return

        sp = self._set_panel
        rp = self._image_panel

        if sp is not None:
            if meta.get("generator"):
                idx = sp.generator.findText(meta["generator"])
                if idx >= 0:
                    sp.generator.setCurrentIndex(idx)
            if meta.get("kernel"):
                idx = sp.kernel.findText(meta["kernel"])
                if idx >= 0:
                    sp.kernel.setCurrentIndex(idx)
            if meta.get("power") is not None:
                sp.power.setValue(int(meta["power"]))
            c = meta.get("c")
            if isinstance(c, (list, tuple)) and len(c) == 2:
                sp.c_real.setValue(float(c[0]))
                sp.c_imag.setValue(float(c[1]))
            center = meta.get("center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                sp.center_real.setValue(float(center[0]))
                sp.center_imag.setValue(float(center[1]))
            if meta.get("recenter") is not None:
                sp.recenter.setChecked(bool(meta["recenter"]))
            if hasattr(sp, "apply_extra_constants"):
                sp.apply_extra_constants(meta.get("extra_constants") or [])
            flw = meta.get("flower") or {}
            if flw:
                if flw.get("petals") is not None:
                    sp.petals.setValue(int(flw["petals"]))
                if flw.get("mirror_segments") is not None:
                    sp.mirror_segments.setChecked(bool(flw["mirror_segments"]))
                if flw.get("interest_angle") is not None:
                    sp.interest_angle.setValue(float(flw["interest_angle"]))
                if flw.get("align_north"):
                    idx = sp.align_north.findText(flw["align_north"])
                    if idx >= 0:
                        sp.align_north.setCurrentIndex(idx)

        if rp is not None:
            if meta.get("width") is not None:
                rp.width.setValue(int(meta["width"]))
            if meta.get("height") is not None:
                rp.height.setValue(int(meta["height"]))
            if meta.get("max_iter") is not None:
                rp.max_iter.setValue(int(meta["max_iter"]))
            if meta.get("bailout") is not None:
                rp.bailout.setValue(float(meta["bailout"]))
            if meta.get("scale_start") is not None:
                rp.scale_start.setValue(float(meta["scale_start"]))
            if meta.get("zoom_factor") is not None:
                rp.zoom_factor.setValue(float(meta["zoom_factor"]))
            if meta.get("num_frames") is not None:
                rp.num_frames.setValue(int(meta["num_frames"]))
            if meta.get("start_frame") is not None:
                rp.start_frame.setValue(int(meta["start_frame"]))
            if meta.get("frame_step") is not None:
                rp.frame_step.setValue(int(meta["frame_step"]))
            if meta.get("mask_svg"):
                rp.mask_svg.setText(str(meta["mask_svg"]))
            if hasattr(rp, "series_name"):
                rp.series_name.setText(series)

        if hasattr(main_win, "console"):
            main_win.console.append_text(
                f"\u2713 Loaded parameters from {series}/_raw/meta.json\n")

    def _load_coloring(self):
        """Load the colour program that produced the currently displayed
        colour folder back into the Coloring tab.

        We look for ``program.json`` inside the currently loaded folder
        (written by colorize.py).  Its ``segments`` list uses user-facing
        rendered-image indices, so it can be applied to the GUI as-is.
        """
        import json
        main_win = self.window()

        # The currently displayed colour folder = directory of the first frame.
        if not self.frame_paths:
            if hasattr(main_win, "console"):
                main_win.console.append_text(
                    "\u26a0 No colour folder selected.\n")
            return
        folder = os.path.dirname(self.frame_paths[0])
        prog_path = os.path.join(folder, "program.json")
        if not os.path.isfile(prog_path):
            if hasattr(main_win, "console"):
                main_win.console.append_text(
                    f"\u26a0 No program.json in {folder}\n"
                    f"  (only colour folders produced by the new pipeline "
                    f"contain one)\n")
            return
        try:
            with open(prog_path) as f:
                prog = json.load(f)
        except (OSError, ValueError) as e:
            if hasattr(main_win, "console"):
                main_win.console.append_text(f"\u26a0 {e}\n")
            return

        cp = getattr(main_win, "coloring_panel", None)
        if cp is None:
            return
        cp.apply_program(prog)

        # Also align the Coloring tab's series selection with the one
        # currently shown in the viewer.
        series = self.series_combo.currentText()
        if series:
            idx = cp.series_combo.findText(series)
            if idx >= 0:
                cp.series_combo.setCurrentIndex(idx)

        if hasattr(main_win, "console"):
            n_seg = len(prog.get("segments", []))
            main_win.console.append_text(
                f"\u2713 Loaded colour program from "
                f"{os.path.basename(folder)}/program.json ({n_seg} segment(s))\n")

    def load_folder(self, folder):
        if not os.path.isabs(folder):
            folder = os.path.join(PROJECT_DIR, folder)
        frames = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
        prev_idx = self.slider.value() if self.slider.isEnabled() else 0
        self.frame_paths = frames
        n = len(self.frame_paths)
        if n == 0:
            self.image_label.setText(f"No frame_*.png in\n{folder}")
            self.slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.frame_label.setText("0 / 0")
            return
        new_idx = max(0, min(prev_idx, n - 1))
        self.slider.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(new_idx)
        self.slider.blockSignals(False)
        self._show_frame(new_idx)

    def _step_frame(self, delta):
        if not self.frame_paths:
            return
        new_idx = max(0, min(self.slider.value() + delta, len(self.frame_paths) - 1))
        self.slider.setValue(new_idx)

    def select_series(self, series_name):
        self._refresh_series()
        idx = self.series_combo.findText(series_name)
        if idx >= 0:
            self.series_combo.setCurrentIndex(idx)

    def _show_frame(self, idx):
        if not self.frame_paths or idx >= len(self.frame_paths):
            return
        pixmap = QPixmap(self.frame_paths[idx])
        if pixmap.isNull():
            return
        self._current_pixmap = pixmap
        self._fit_pixmap()
        self.frame_label.setText(f"{idx + 1} / {len(self.frame_paths)}")

    def _fit_pixmap(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tabbed Preview Widget (synced with input tabs)
# ═══════════════════════════════════════════════════════════════════════════════
class TabbedPreview(QTabWidget):
    """Two viewer tabs that stay in sync with the left-hand parameter tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_viewer = SetSelectionViewer()
        self.render_viewer = RenderingViewer()
        self.addTab(self.set_viewer, "Set Selection")
        self.addTab(self.render_viewer, "Image Rendering")

    def load_folder(self, folder):
        self.set_viewer.load_folder(folder)

    def refresh_quick_select(self):
        self.set_viewer.refresh_quick_select()

    def _load_frames(self):
        self.set_viewer._load_frames()


# ═══════════════════════════════════════════════════════════════════════════════
#  Output Console
# ═══════════════════════════════════════════════════════════════════════════════
class ConsoleWidget(QWidget):
    """Read-only log view for stdout/stderr from operations triggered via the GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        lbl = QLabel("Output log")
        lbl.setStyleSheet("color: #888;")
        header.addWidget(lbl)
        header.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(70)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet("background: #1a1a1a; color: #ddd;")
        self._log.setMaximumBlockCount(50000)
        layout.addWidget(self._log)

        clear_btn.clicked.connect(self._log.clear)

    def append_text(self, text: str):
        if not text:
            return
        sb = self._log.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cursor = self._log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        if at_bottom:
            sb.setValue(sb.maximum())

