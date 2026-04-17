"""
Fractal Studio – PyQt6 desktop GUI for render.py & video.py.

Usage:  python gui.py
"""
import sys
import os
import glob
import signal

from PyQt6.QtCore import Qt, QProcess, QTimer
from PyQt6.QtGui import QPixmap, QFont, QKeyEvent, QTextCursor, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QTabWidget,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QFileDialog, QPlainTextEdit, QScrollArea,
    QSizePolicy, QListWidget, QAbstractItemView, QStatusBar,
)

# ── Project imports (for registry keys) ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from generators import GENERATORS
from colorizers import COLORIZERS

PYTHON = sys.executable
# When frozen by PyInstaller, scripts are bundled as data files inside _internal
if getattr(sys, 'frozen', False):
    PROJECT_DIR = os.path.dirname(sys.executable)
    SCRIPT_DIR = os.path.join(sys._MEIPASS)
else:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_DIR = PROJECT_DIR
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PREVIEW_DIR = os.path.join(RESULTS_DIR, "_preview")


def next_preview_folder():
    """Return the next numbered preview folder path (e.g. results/_preview/003/)."""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    existing = [d for d in os.listdir(PREVIEW_DIR)
                if os.path.isdir(os.path.join(PREVIEW_DIR, d)) and d.isdigit()]
    num = max((int(d) for d in existing), default=0) + 1
    path = os.path.join(PREVIEW_DIR, f"{num:03d}")
    os.makedirs(path, exist_ok=True)
    return path, num


# ═══════════════════════════════════════════════════════════════════════════════
#  Render Parameter Panel
# ═══════════════════════════════════════════════════════════════════════════════
class RenderPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        # ── Generator ────────────────────────────────────────────────────────
        grp = QGroupBox("Generator")
        form = QFormLayout(grp)

        self.generator = QComboBox()
        self.generator.addItems(GENERATORS.keys())
        self.generator.setCurrentText("julia")
        form.addRow("Generator:", self.generator)

        self.kernel = QComboBox()
        self.kernel.addItems(["poly", "sin", "cos"])
        form.addRow("Kernel:", self.kernel)

        self.power = QSpinBox(); self.power.setRange(2, 20); self.power.setValue(2)
        form.addRow("Power (z^n):", self.power)

        self.c_real = QDoubleSpinBox(); self.c_real.setRange(-5, 5); self.c_real.setDecimals(4); self.c_real.setSingleStep(0.01); self.c_real.setValue(-0.54)
        self.c_imag = QDoubleSpinBox(); self.c_imag.setRange(-5, 5); self.c_imag.setDecimals(4); self.c_imag.setSingleStep(0.01); self.c_imag.setValue(0.54)
        form.addRow("c real:", self.c_real)
        form.addRow("c imag:", self.c_imag)

        # Preview buttons
        preview_row = QHBoxLayout()
        self.preview_lo = QPushButton("Preview 500×500")
        self.preview_hi = QPushButton("Preview 2160×2160")
        self.preview_lo.clicked.connect(lambda: self._run_preview(500))
        self.preview_hi.clicked.connect(lambda: self._run_preview(2160))
        preview_row.addWidget(self.preview_lo)
        preview_row.addWidget(self.preview_hi)
        form.addRow("Preview:", preview_row)

        layout.addWidget(grp)

        # ── Flower Options (shown only when generator == "flower") ───────────
        self.flower_group = QGroupBox("Flower Options")
        flower_form = QFormLayout(self.flower_group)

        self.petals = QSpinBox(); self.petals.setRange(1, 64); self.petals.setValue(4)
        self.mirror_segments = QCheckBox()
        self.interest_angle = QDoubleSpinBox(); self.interest_angle.setRange(0, 360); self.interest_angle.setValue(110)
        self.align_north = QComboBox(); self.align_north.addItems(["center", "edge"])
        self.align_north.setCurrentText("edge")

        flower_form.addRow("Petals:", self.petals)
        flower_form.addRow("Mirror segments:", self.mirror_segments)
        flower_form.addRow("Interest angle:", self.interest_angle)
        flower_form.addRow("Align north:", self.align_north)

        layout.addWidget(self.flower_group)

        # ── Zoom ─────────────────────────────────────────────────────────────
        grp = QGroupBox("Zoom")
        form = QFormLayout(grp)

        self.center_real = QDoubleSpinBox(); self.center_real.setRange(-10, 10); self.center_real.setDecimals(4); self.center_real.setSingleStep(0.01); self.center_real.setValue(0.0)
        self.center_imag = QDoubleSpinBox(); self.center_imag.setRange(-10, 10); self.center_imag.setDecimals(4); self.center_imag.setSingleStep(0.01); self.center_imag.setValue(0.0)
        form.addRow("Center real:", self.center_real)
        form.addRow("Center imag:", self.center_imag)

        self.scale_start = QDoubleSpinBox(); self.scale_start.setRange(0.01, 100); self.scale_start.setDecimals(2); self.scale_start.setValue(2.0)
        self.zoom_factor = QDoubleSpinBox(); self.zoom_factor.setRange(0.5, 0.9999); self.zoom_factor.setDecimals(6); self.zoom_factor.setSingleStep(0.01); self.zoom_factor.setValue(0.92)
        self.num_frames = QSpinBox(); self.num_frames.setRange(1, 100000); self.num_frames.setValue(30)
        self.start_frame = QSpinBox(); self.start_frame.setRange(0, 100000); self.start_frame.setValue(0)
        self.frame_step = QSpinBox(); self.frame_step.setRange(1, 100); self.frame_step.setValue(1)
        form.addRow("Scale start:", self.scale_start)
        form.addRow("Zoom factor:", self.zoom_factor)
        form.addRow("Num frames:", self.num_frames)
        form.addRow("Start frame:", self.start_frame)
        form.addRow("Frame step:", self.frame_step)

        # Zoom preview
        self.zoom_preview_btn = QPushButton("Preview Zoom Series (500×500)")
        self.zoom_preview_btn.clicked.connect(self._run_zoom_preview)
        form.addRow("", self.zoom_preview_btn)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Depth:"))
        self.zoom_depth = QSpinBox(); self.zoom_depth.setRange(11, 100000); self.zoom_depth.setValue(100)
        self.zoom_depth.setMaximumWidth(self.zoom_preview_btn.sizeHint().width() // 2)
        depth_row.addWidget(self.zoom_depth)
        depth_row.addStretch()
        form.addRow("", depth_row)

        layout.addWidget(grp)

        # ── Coloring Scheme ──────────────────────────────────────────────────
        grp = QGroupBox("Coloring Scheme")
        form = QFormLayout(grp)

        self.colorizers = QListWidget()
        self.colorizers.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for name in COLORIZERS:
            self.colorizers.addItem(name)
        self.colorizers.selectAll()
        self.colorizers.setMaximumHeight(70)
        form.addRow("Colorizers:", self.colorizers)

        layout.addWidget(grp)

        # ── Quality ──────────────────────────────────────────────────────────
        grp = QGroupBox("Quality")
        form = QFormLayout(grp)

        self.width = QSpinBox(); self.width.setRange(64, 7680); self.width.setSingleStep(64); self.width.setValue(640)
        self.height = QSpinBox(); self.height.setRange(64, 4320); self.height.setSingleStep(64); self.height.setValue(640)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(1024)
        self.bailout = QDoubleSpinBox(); self.bailout.setRange(0.1, 10000); self.bailout.setDecimals(1); self.bailout.setValue(2.0)
        form.addRow("Width:", self.width)
        form.addRow("Height:", self.height)
        form.addRow("Max iterations:", self.max_iter)
        form.addRow("Bailout:", self.bailout)

        layout.addWidget(grp)

        # ── Execution ────────────────────────────────────────────────────────
        grp = QGroupBox("Execution")
        form = QFormLayout(grp)

        self.workers = QSpinBox(); self.workers.setRange(1, 64); self.workers.setValue(12)
        form.addRow("Workers:", self.workers)

        self.mask_svg = QLineEdit()
        mask_btn = QPushButton("Browse…")
        mask_btn.clicked.connect(self._pick_mask)
        h = QHBoxLayout()
        h.addWidget(self.mask_svg, 1)
        h.addWidget(mask_btn)
        form.addRow("Mask SVG:", h)

        layout.addWidget(grp)

        # ── Start button ─────────────────────────────────────────────────────
        self.start_btn = QPushButton("▶  Start Render")
        self.start_btn.setStyleSheet("padding: 8px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.start_btn)

        layout.addStretch()
        self.setWidget(w)

        self.generator.currentTextChanged.connect(self._toggle_flower)
        self._toggle_flower()

    def _pick_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SVG mask", PROJECT_DIR, "SVG Files (*.svg)")
        if path:
            self.mask_svg.setText(os.path.relpath(path, PROJECT_DIR))

    def _toggle_flower(self):
        self.flower_group.setVisible(self.generator.currentText() == "flower")

    def _run_preview(self, size):
        """Render frame 0 at given resolution for all selected colorizers."""
        import threading
        from time import perf_counter as clock

        self.preview_lo.setEnabled(False)
        self.preview_hi.setEnabled(False)
        self.preview_lo.setText("Rendering…")
        self.preview_hi.setText("Rendering…")

        main_win = self.window()

        def _log(msg):
            """Thread-safe logging to console."""
            # Use invokeMethod via a stored reference to avoid QTimer from bg thread
            main_win._pending_logs.append(msg)

        # Capture ALL widget values on the main thread before spawning
        gen_name = self.generator.currentText()
        gen_cls = GENERATORS[gen_name]
        kwargs = dict(
            c=complex(self.c_real.value(), self.c_imag.value()),
            center=complex(self.center_real.value(), self.center_imag.value()),
            width=size, height=size,
            max_iter=self.max_iter.value(),
            power=self.power.value(),
            kernel=self.kernel.currentText(),
            bailout=self.bailout.value(),
        )
        if gen_name == "flower":
            kwargs.update(
                petals=self.petals.value(),
                mirror_segments=self.mirror_segments.isChecked(),
                interest_angle=self.interest_angle.value(),
                align_north=self.align_north.currentText(),
            )
        scale = self.scale_start.value()
        mask_path = self.mask_svg.text().strip()

        # Pure computation – no Qt access
        def _do_render():
            try:
                _log(f"\n⏳ Preview {size}×{size} | {gen_name} kernel={kwargs['kernel']} "
                     f"c={kwargs['c']} scale={scale}\n")

                _log(f"  Creating generator ({gen_name})…\n")
                t0 = clock()
                gen = gen_cls(**kwargs)
                _log(f"  Generator ready ({clock()-t0:.2f}s)\n")

                mask = None
                if mask_path:
                    _log(f"  Loading mask: {mask_path}…\n")
                    t0 = clock()
                    from mask import load_svg_mask, radial_fill
                    display_mask, _ = load_svg_mask(mask_path, size, size)
                    mask = radial_fill(display_mask)
                    _log(f"  Mask ready ({clock()-t0:.2f}s)\n")

                _log(f"  Rendering counts ({size}×{size}, max_iter={kwargs['max_iter']})…\n")
                t0 = clock()
                counts = gen.render(scale, mask=mask)
                _log(f"  Render done ({clock()-t0:.2f}s)\n")

                preview_dir, preview_num = next_preview_folder()

                # Greyscale: normalize counts to 0–255
                import numpy as np
                _log(f"  Saving greyscale preview…\n")
                t0 = clock()
                esc = counts < kwargs["max_iter"]
                grey = np.zeros(counts.shape, dtype=np.uint8)
                if esc.any():
                    vals = counts[esc].astype(np.float64)
                    lo, hi = np.percentile(vals, [1, 99])
                    if hi <= lo:
                        hi = lo + 1.0
                    grey[esc] = (np.clip((vals - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                from PIL import Image, ImageDraw, ImageFont
                img = Image.fromarray(grey, mode='L')
                # Stamp parameters
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("consola.ttf", max(12, size // 40))
                except Exception:
                    font = ImageFont.load_default()
                lines = [
                    f"gen={gen_name} kernel={kwargs['kernel']} n={kwargs['power']}",
                    f"c={kwargs['c']}",
                    f"scale={scale:.4f}  iter={kwargs['max_iter']}",
                ]
                y = 4
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle([2, y - 1, tw + 6, y + th + 1], fill=0)
                    draw.text((4, y), line, fill=255, font=font)
                    y += th + 3
                path = os.path.join(preview_dir, f"preview_greyscale_{size}x{size}.png")
                img.save(path)
                _log(f"  Saved {os.path.basename(path)} ({clock()-t0:.2f}s)\n")
                return [path]
            except Exception as e:
                import traceback
                return f"{e}\n{traceback.format_exc()}"

        self._preview_result = None

        def _thread():
            self._preview_result = _do_render()

        t = threading.Thread(target=_thread, daemon=True)
        t.start()

        # Poll from main thread until done
        def _poll():
            # Flush pending log messages
            while main_win._pending_logs:
                msg = main_win._pending_logs.pop(0)
                main_win.console.append_text(msg)

            if t.is_alive():
                QTimer.singleShot(100, _poll)
                return

            # Thread finished – flush remaining logs
            while main_win._pending_logs:
                msg = main_win._pending_logs.pop(0)
                main_win.console.append_text(msg)

            # Reset buttons
            self.preview_lo.setEnabled(True)
            self.preview_hi.setEnabled(True)
            self.preview_lo.setText("Preview 500×500")
            self.preview_hi.setText("Preview 2160×2160")

            result = self._preview_result
            if isinstance(result, str):
                main_win.console.append_text(f"\n⚠ Preview error: {result}\n")
            elif result:
                preview_dir = os.path.dirname(result[0])
                main_win.preview.load_folder(preview_dir)
                main_win.preview.refresh_quick_select()
                main_win.console.append_text(
                    f"\n✓ Preview rendered: {len(result)} images ({size}×{size})\n")

        QTimer.singleShot(100, _poll)

    def _run_zoom_preview(self):
        """Render a zoom series at 500×500 greyscale, 11 frames spanning the full depth."""
        import threading
        from time import perf_counter as clock

        size = 500
        depth = self.zoom_depth.value()
        # 11 evenly spaced frames from 0 to depth-1
        n_samples = 11
        step = max(1, (depth - 1) // (n_samples - 1))
        frame_indices = list(range(0, depth, step))
        if frame_indices[-1] != depth - 1:
            frame_indices.append(depth - 1)

        self.zoom_preview_btn.setEnabled(False)
        self.zoom_preview_btn.setText("Rendering…")

        main_win = self.window()

        def _log(msg):
            main_win._pending_logs.append(msg)

        # Capture all params on main thread
        gen_name = self.generator.currentText()
        gen_cls = GENERATORS[gen_name]
        kwargs = dict(
            c=complex(self.c_real.value(), self.c_imag.value()),
            center=complex(self.center_real.value(), self.center_imag.value()),
            width=size, height=size,
            max_iter=self.max_iter.value(),
            power=self.power.value(),
            kernel=self.kernel.currentText(),
            bailout=self.bailout.value(),
        )
        if gen_name == "flower":
            kwargs.update(
                petals=self.petals.value(),
                mirror_segments=self.mirror_segments.isChecked(),
                interest_angle=self.interest_angle.value(),
                align_north=self.align_north.currentText(),
            )
        scale_start = self.scale_start.value()
        zoom_factor = self.zoom_factor.value()
        mask_path = self.mask_svg.text().strip()

        def _do_render():
            try:
                import numpy as np
                from PIL import Image as PILImage

                _log(f"\n⏳ Zoom preview {size}×{size} | depth={depth} "
                     f"| {len(frame_indices)} samples | step={step}\n")

                t0_total = clock()
                gen = gen_cls(**kwargs)

                mask = None
                if mask_path:
                    _log(f"  Loading mask: {mask_path}…\n")
                    t0 = clock()
                    from mask import load_svg_mask, radial_fill
                    display_mask, _ = load_svg_mask(mask_path, size, size)
                    mask = radial_fill(display_mask)
                    _log(f"  Mask ready ({clock()-t0:.2f}s)\n")

                preview_dir, preview_num = next_preview_folder()

                saved = []
                for i, fi in enumerate(frame_indices):
                    scale = scale_start * (zoom_factor ** fi)
                    _log(f"  [{i+1}/{len(frame_indices)}] frame {fi}, scale={scale:.6f}…\n")
                    t0 = clock()
                    counts = gen.render(scale, mask=mask)

                    esc = counts < kwargs["max_iter"]
                    grey = np.zeros(counts.shape, dtype=np.uint8)
                    if esc.any():
                        vals = counts[esc].astype(np.float64)
                        lo, hi = np.percentile(vals, [1, 99])
                        if hi <= lo:
                            hi = lo + 1.0
                        grey[esc] = (np.clip((vals - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

                    img = PILImage.fromarray(grey, mode='L')
                    draw = PILImage.core  # unused, just to keep namespace
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("consola.ttf", max(12, size // 40))
                    except Exception:
                        font = ImageFont.load_default()
                    lines = [f"#{fi}"]
                    if i == 0:
                        lines = [
                            f"gen={gen_name} kernel={kwargs['kernel']} n={kwargs['power']}",
                            f"c={kwargs['c']}",
                            f"center={kwargs['center']}  zoom={zoom_factor}",
                            f"scale={scale:.6f}  frame #{fi}",
                        ]
                    y = 4
                    for line in lines:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.rectangle([2, y - 1, tw + 6, y + th + 1], fill=0)
                        draw.text((4, y), line, fill=255, font=font)
                        y += th + 3
                    path = os.path.join(preview_dir, f"frame_{fi:04d}.png")
                    img.save(path)
                    saved.append(path)
                    _log(f"    done ({clock()-t0:.1f}s)\n")

                _log(f"  Total: {clock()-t0_total:.1f}s\n")
                return saved
            except Exception as e:
                import traceback
                return f"{e}\n{traceback.format_exc()}"

        self._zoom_preview_result = None

        def _thread():
            self._zoom_preview_result = _do_render()

        t = threading.Thread(target=_thread, daemon=True)
        t.start()

        def _poll():
            while main_win._pending_logs:
                msg = main_win._pending_logs.pop(0)
                main_win.console.append_text(msg)

            if t.is_alive():
                QTimer.singleShot(100, _poll)
                return

            while main_win._pending_logs:
                msg = main_win._pending_logs.pop(0)
                main_win.console.append_text(msg)

            self.zoom_preview_btn.setEnabled(True)
            self.zoom_preview_btn.setText("Preview Zoom Series (500×500)")

            result = self._zoom_preview_result
            if isinstance(result, str):
                main_win.console.append_text(f"\n⚠ Zoom preview error: {result}\n")
            elif result:
                preview_dir = os.path.dirname(result[0])
                main_win.preview.load_folder(preview_dir)
                main_win.preview.refresh_quick_select()
                main_win.console.append_text(
                    f"\n✓ Zoom preview: {len(result)} frames rendered\n")

        QTimer.singleShot(100, _poll)

    def build_args(self):
        """Return list of CLI args for render.py."""
        args = [
            "--generator", self.generator.currentText(),
            "--kernel", self.kernel.currentText(),
            "--c-real", str(self.c_real.value()),
            "--c-imag", str(self.c_imag.value()),
            "--center-real", str(self.center_real.value()),
            "--center-imag", str(self.center_imag.value()),
            "--width", str(self.width.value()),
            "--height", str(self.height.value()),
            "--scale-start", str(self.scale_start.value()),
            "--zoom-factor", str(self.zoom_factor.value()),
            "--num-frames", str(self.num_frames.value()),
            "--start-frame", str(self.start_frame.value()),
            "--frame-step", str(self.frame_step.value()),
            "--max-iter", str(self.max_iter.value()),
            "--power", str(self.power.value()),
            "--bailout", str(self.bailout.value()),
            "--workers", str(self.workers.value()),
        ]
        sel = [item.text() for item in self.colorizers.selectedItems()]
        if sel:
            args += ["--colorizers"] + sel
        if self.generator.currentText() == "flower":
            args += [
                "--petals", str(self.petals.value()),
                "--interest-angle", str(self.interest_angle.value()),
                "--align-north", self.align_north.currentText(),
            ]
            if self.mirror_segments.isChecked():
                args.append("--mirror-segments")
        if self.mask_svg.text().strip():
            args += ["--mask-svg", self.mask_svg.text().strip()]
        return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Parameter Panel
# ═══════════════════════════════════════════════════════════════════════════════
class VideoPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Video Encoding")
        form = QFormLayout(grp)

        self.folder = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._pick_folder)
        h = QHBoxLayout()
        h.addWidget(self.folder, 1)
        h.addWidget(browse_btn)
        form.addRow("Frame folder:", h)

        self.fps = QSpinBox(); self.fps.setRange(1, 240); self.fps.setValue(30)
        self.interp = QSpinBox(); self.interp.setRange(1, 100); self.interp.setValue(1)
        self.zoom_factor = QDoubleSpinBox(); self.zoom_factor.setRange(0.5, 0.9999); self.zoom_factor.setDecimals(6); self.zoom_factor.setSingleStep(0.01); self.zoom_factor.setValue(0.92)
        self.loop = QComboBox(); self.loop.addItems(["none", "bounce", "repeat"])
        self.start_frame = QSpinBox(); self.start_frame.setRange(-1, 100000); self.start_frame.setSpecialValueText("auto"); self.start_frame.setValue(-1)
        self.end_frame = QSpinBox(); self.end_frame.setRange(-1, 100000); self.end_frame.setSpecialValueText("auto"); self.end_frame.setValue(-1)
        self.mask_svg = QLineEdit()
        mask_btn = QPushButton("Browse…")
        mask_btn.clicked.connect(self._pick_mask)
        hm = QHBoxLayout()
        hm.addWidget(self.mask_svg, 1)
        hm.addWidget(mask_btn)

        self.output = QLineEdit()
        self.output.setPlaceholderText("auto-generated")

        form.addRow("FPS:", self.fps)
        form.addRow("Interp:", self.interp)
        form.addRow("Zoom factor:", self.zoom_factor)
        form.addRow("Loop:", self.loop)
        form.addRow("Start frame:", self.start_frame)
        form.addRow("End frame:", self.end_frame)
        form.addRow("Mask SVG:", hm)
        form.addRow("Output:", self.output)

        layout.addWidget(grp)

        self.start_btn = QPushButton("▶  Encode Video")
        self.start_btn.setStyleSheet("padding: 8px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.start_btn)

        layout.addStretch()
        self.setWidget(w)

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select frame folder", RESULTS_DIR)
        if path:
            self.folder.setText(os.path.relpath(path, PROJECT_DIR))

    def _pick_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SVG mask", PROJECT_DIR, "SVG Files (*.svg)")
        if path:
            self.mask_svg.setText(os.path.relpath(path, PROJECT_DIR))

    def build_args(self):
        """Return list of CLI args for video.py."""
        folder = self.folder.text().strip()
        if not folder:
            return None
        args = [folder,
                "--fps", str(self.fps.value()),
                "--interp", str(self.interp.value()),
                "--zoom-factor", str(self.zoom_factor.value()),
                "--loop", self.loop.currentText()]
        if self.start_frame.value() >= 0:
            args += ["--start-frame", str(self.start_frame.value())]
        if self.end_frame.value() >= 0:
            args += ["--end-frame", str(self.end_frame.value())]
        if self.mask_svg.text().strip():
            args += ["--mask-svg", self.mask_svg.text().strip()]
        if self.output.text().strip():
            args += ["--output", self.output.text().strip()]
        return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame Preview Widget
# ═══════════════════════════════════════════════════════════════════════════════
class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Folder picker
        top = QHBoxLayout()
        top.addWidget(QLabel("Folder:"))
        self.folder_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_folder)
        top.addWidget(self.folder_edit, 1)
        top.addWidget(browse)
        layout.addLayout(top)

        # Quick-select row for numbered preview sets + clear button
        qs_row = QHBoxLayout()
        qs_row.addWidget(QLabel("Previews:"))
        self._qs_container = QHBoxLayout()
        qs_row.addLayout(self._qs_container)
        qs_row.addStretch()
        self.clear_btn = QPushButton("🗑 Clear All")
        self.clear_btn.setMaximumWidth(100)
        self.clear_btn.clicked.connect(self._clear_previews)
        qs_row.addWidget(self.clear_btn)
        layout.addLayout(qs_row)

        # Image label
        self.image_label = QLabel("No frames loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background: #1a1a1a; color: #888;")
        layout.addWidget(self.image_label, 1)

        # Slider + label
        bot = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        bot.addWidget(self.slider, 1)
        bot.addWidget(self.frame_label)
        layout.addLayout(bot)

        # Refresh button
        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self._load_frames)
        layout.addWidget(self.refresh_btn)

        self.frame_paths = []
        self._current_pixmap = None
        self._qs_buttons = []

        # Initial population of quick-select buttons
        self.refresh_quick_select()

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select colorizer folder", RESULTS_DIR)
        if path:
            self.folder_edit.setText(path)
            self._load_frames()

    def refresh_quick_select(self):
        """Rebuild the numbered quick-select buttons from _preview folder."""
        # Remove old buttons
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
        """Delete all numbered preview folders."""
        import shutil
        if os.path.isdir(PREVIEW_DIR):
            shutil.rmtree(PREVIEW_DIR)
        self.refresh_quick_select()
        self.frame_paths = []
        self.slider.setEnabled(False)
        self.frame_label.setText("0 / 0")
        self.image_label.setText("Previews cleared")
        self.image_label.setPixmap(QPixmap())
        self._current_pixmap = None

    def load_folder(self, folder):
        self.folder_edit.setText(folder)
        self._load_frames()

    def _load_frames(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            return
        # Try both absolute and relative
        if not os.path.isabs(folder):
            folder = os.path.join(PROJECT_DIR, folder)
        frames = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
        previews = sorted(glob.glob(os.path.join(folder, "preview_*.png")))
        self.frame_paths = frames if frames else previews
        n = len(self.frame_paths)
        if n == 0:
            self.image_label.setText(f"No frame_*.png in\n{folder}")
            self.slider.setEnabled(False)
            self.frame_label.setText("0 / 0")
            return
        self.slider.setEnabled(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self._show_frame(0)

    def _show_frame(self, idx):
        if not self.frame_paths or idx >= len(self.frame_paths):
            return
        path = self.frame_paths[idx]
        pixmap = QPixmap(path)
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
#  Interactive Console Widget (IPython in-process via qtconsole, with fallback)
# ═══════════════════════════════════════════════════════════════════════════════
class ConsoleWidget(QWidget):
    """Wraps either qtconsole RichJupyterWidget or a basic QProcess terminal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._jupyter = None
        try:
            self._init_jupyter(layout)
        except Exception:
            self._init_fallback(layout)

    # ── Jupyter QtConsole (rich IPython) ─────────────────────────────────────
    def _init_jupyter(self, layout):
        from qtconsole.rich_jupyter_widget import RichJupyterWidget
        from qtconsole.inprocess import QtInProcessKernelManager

        km = QtInProcessKernelManager()
        km.start_kernel()
        kc = km.client()
        kc.start_channels()

        widget = RichJupyterWidget()
        widget.kernel_manager = km
        widget.kernel_client = kc
        widget.setStyleSheet("background: #1a1a1a;")

        # Pre-load project modules in the kernel
        km.kernel.shell.run_cell(
            "import sys, os\n"
            f"os.chdir(r'{PROJECT_DIR}')\n"
            f"sys.path.insert(0, r'{PROJECT_DIR}')\n"
            "from generators import GENERATORS\n"
            "from colorizers import COLORIZERS\n"
            "import numpy as np\n"
            "print('Fractal Studio – IPython ready.')\n"
            "print(f'Generators: {list(GENERATORS.keys())}')\n"
            "print(f'Colorizers: {list(COLORIZERS.keys())}')\n",
            silent=False, store_history=False,
        )

        layout.addWidget(widget)
        self._jupyter = widget
        self._km = km

    # ── Fallback: plain text log ─────────────────────────────────────────────
    def _init_fallback(self, layout):
        lbl = QLabel("Console (output only – install qtconsole for interactive Python)")
        lbl.setStyleSheet("color: #888; padding: 2px;")
        layout.addWidget(lbl)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet("background: #1a1a1a; color: #ddd;")
        self._log.setMaximumBlockCount(50000)
        layout.addWidget(self._log)

    def append_text(self, text: str):
        """Append process output to the console."""
        if self._jupyter is not None:
            # Use the kernel to print so it goes through proper Jupyter output
            self._km.kernel.shell.run_cell(
                f"print({text!r}, end='')",
                silent=True, store_history=False,
            )
        else:
            self._log.appendPlainText(text.rstrip("\n"))
            self._log.verticalScrollBar().setValue(
                self._log.verticalScrollBar().maximum()
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Studio")
        self.resize(1400, 900)

        # ── Central splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: parameter tabs
        tabs = QTabWidget()
        self.render_panel = RenderPanel()
        self.video_panel = VideoPanel()
        tabs.addTab(self.render_panel, "Render")
        tabs.addTab(self.video_panel, "Video")
        tabs.setMaximumWidth(420)
        splitter.addWidget(tabs)

        # Right: preview + console
        right = QSplitter(Qt.Orientation.Vertical)

        self.preview = PreviewWidget()
        right.addWidget(self.preview)

        self.console = ConsoleWidget()
        right.addWidget(self.console)

        right.setSizes([500, 300])
        splitter.addWidget(right)

        splitter.setSizes([380, 1020])
        self.setCentralWidget(splitter)

        # ── Status bar ───────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

        # ── Connections ──────────────────────────────────────────────────────
        self.render_panel.start_btn.clicked.connect(self._start_render)
        self.video_panel.start_btn.clicked.connect(self._start_video)

        self._process = None
        self._pending_logs = []  # thread-safe log buffer for preview

    # ── Process management ───────────────────────────────────────────────────
    def _run_script(self, script, args):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self.console.append_text("\n⚠  A process is already running.\n")
            return
        self._process = QProcess(self)
        self._process.setWorkingDirectory(PROJECT_DIR)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        script_path = os.path.join(SCRIPT_DIR, script)
        cmd_display = f"python {script} {' '.join(args)}"
        self.console.append_text(f"\n$ {cmd_display}\n")
        self.statusBar().showMessage(f"Running {script}…")

        if getattr(sys, 'frozen', False):
            # In frozen mode, use the bundled exe with --run-script flag
            self._process.start(sys.executable, ["--run-script", script_path] + args)
        else:
            self._process.start(PYTHON, [script_path] + args)

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.console.append_text(data)

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self.console.append_text(data)

    def _on_finished(self, exit_code, status):
        self.console.append_text(f"\n✓ Process finished (exit code {exit_code})\n")
        self.statusBar().showMessage("Ready")
        # Auto-refresh preview if frames exist
        self.preview._load_frames()

    # ── Button handlers ──────────────────────────────────────────────────────
    def _start_render(self):
        args = self.render_panel.build_args()
        self._run_script("render.py", args)

    def _start_video(self):
        args = self.video_panel.build_args()
        if args is None:
            self.console.append_text("\n⚠  Please set a frame folder first.\n")
            return
        self._run_script("video.py", args)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Hide the console window in frozen GUI mode
    if getattr(sys, 'frozen', False):
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except Exception:
            pass

    # Allow Ctrl+C in terminal to kill the app
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 100, 100))
    palette.setColor(QPalette.ColorRole.Link, QColor(90, 150, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 120, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # In frozen mode, support re-invocation for subprocess scripts
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        script = sys.argv[2]
        sys.argv = sys.argv[2:]  # make it look like: script.py [args...]
        # Ensure project modules are importable
        script_dir = os.path.dirname(os.path.abspath(script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        with open(script) as f:
            exec(compile(f.read(), script, "exec"), {"__name__": "__main__"})
    else:
        main()

