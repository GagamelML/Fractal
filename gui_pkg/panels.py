"""Input panels (Set Selection, Image Rendering, Coloring, Stitch, Video).

Each panel is a self-contained QScrollArea/QGroupBox composition with its
own ``build_args()`` (or equivalent) used by the MainWindow to launch the
appropriate CLI script."""
import os
import sys
import glob
import json

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QFileDialog, QScrollArea, QSizePolicy,
)

# Project imports (registry keys)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators import GENERATORS
from colorizers import COLORIZERS

from .common import (
    PROJECT_DIR, RESULTS_DIR, PREVIEW_DIR,
    save_preview_params, next_preview_folder,
)
from .analysis import (
    detect_recursion, find_nearest_boundary_point,
    find_clean_boundary_points, CleanCentersDialog,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Set Selection Panel  (Tab 1)
# ══════════════════��════════════════════════════════════════════════════════════
class SetSelectionPanel(QScrollArea):
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
        self._kernel_label = QLabel("Kernel:")
        form.addRow(self._kernel_label, self.kernel)

        self.power = QSpinBox(); self.power.setRange(2, 20); self.power.setValue(2)
        form.addRow("Power (z^n):", self.power)

        self.c_real = QDoubleSpinBox(); self.c_real.setRange(-5, 5); self.c_real.setDecimals(7); self.c_real.setSingleStep(0.01); self.c_real.setValue(-0.54)
        self.c_imag = QDoubleSpinBox(); self.c_imag.setRange(-5, 5); self.c_imag.setDecimals(7); self.c_imag.setSingleStep(0.01); self.c_imag.setValue(0.54)
        self._c_real_label = QLabel("c real:")
        self._c_imag_label = QLabel("c imag:")
        form.addRow(self._c_real_label, self.c_real)
        form.addRow(self._c_imag_label, self.c_imag)

        self.scale_start = QDoubleSpinBox(); self.scale_start.setRange(0.01, 100); self.scale_start.setDecimals(2); self.scale_start.setValue(2.0)
        form.addRow("Scale start:", self.scale_start)

        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(1024)
        self.bailout = QDoubleSpinBox(); self.bailout.setRange(0.1, 10000); self.bailout.setDecimals(1); self.bailout.setValue(2.0)
        form.addRow("Max iterations:", self.max_iter)
        form.addRow("Bailout:", self.bailout)

        # Preview buttons
        preview_row = QHBoxLayout()
        self.preview_lo = QPushButton("Preview 500\u00d7500")
        self.preview_hi = QPushButton("Preview 2160\u00d72160")
        self.preview_lo.clicked.connect(lambda: self._run_preview(500))
        self.preview_hi.clicked.connect(lambda: self._run_preview(2160))
        preview_row.addWidget(self.preview_lo)
        preview_row.addWidget(self.preview_hi)
        form.addRow("Preview:", preview_row)

        layout.addWidget(grp)

        # -- Additional Constants (pan c during the series) ----------------
        self.extra_const_group = QGroupBox("Additional Constants")
        ec_layout = QVBoxLayout(self.extra_const_group)
        ec_layout.setContentsMargins(8, 8, 8, 8)
        ec_layout.setSpacing(4)
        self._extra_const_container = QWidget()
        self._extra_const_vbox = QVBoxLayout(self._extra_const_container)
        self._extra_const_vbox.setContentsMargins(0, 0, 0, 0)
        self._extra_const_vbox.setSpacing(4)
        ec_layout.addWidget(self._extra_const_container)
        self._extra_const_rows = []
        self.add_constant_btn = QPushButton("+ Add constant")
        self.add_constant_btn.clicked.connect(lambda: self._add_extra_constant())
        ec_layout.addWidget(self.add_constant_btn)
        layout.addWidget(self.extra_const_group)

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

        # ── Zoom Window ──────────────────────────────────────────────────────
        grp = QGroupBox("Zoom Window")
        form = QFormLayout(grp)

        self.center_real = QDoubleSpinBox(); self.center_real.setRange(-10, 10); self.center_real.setDecimals(15); self.center_real.setSingleStep(0.01); self.center_real.setValue(0.0)
        self.center_imag = QDoubleSpinBox(); self.center_imag.setRange(-10, 10); self.center_imag.setDecimals(15); self.center_imag.setSingleStep(0.01); self.center_imag.setValue(0.0)
        self.recenter = QCheckBox("Keep zoom point centered")
        form.addRow("Center real:", self.center_real)
        form.addRow("Center imag:", self.center_imag)
        form.addRow("", self.recenter)

        self.adjust_center_btn = QPushButton("\U0001f3af Adjust Center to Boundary")
        self.adjust_center_btn.setToolTip(
            "Find the nearest Julia set boundary point and move the center there.\n"
            "Boundary points guarantee interesting structure at every zoom level.")
        self.adjust_center_btn.clicked.connect(self._run_adjust_center)
        form.addRow("", self.adjust_center_btn)

        self.clean_center_btn = QPushButton("\u2728 Suggest Clean Centers")
        self.clean_center_btn.setToolTip(
            "Find boundary points with low floating-point precision cost.\n"
            "Ideal for planning deep zoom series: coordinates close to 0 or\n"
            "to simple dyadic rationals keep more frames numerically clean.")
        self.clean_center_btn.clicked.connect(self._run_suggest_clean_centers)
        form.addRow("", self.clean_center_btn)

        # Zoom preview
        self.zoom_preview_btn = QPushButton("Preview Zoom Series (500\u00d7500)")
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

        # ── Recursion Check ──────────────────────────────────────────────────
        grp = QGroupBox("Recursion Check")
        rec_outer = QFormLayout(grp)

        self.recursion_check_btn = QPushButton("\U0001f50d Check Recursion")
        self.recursion_check_btn.clicked.connect(self._run_recursion_check)
        rec_outer.addRow("", self.recursion_check_btn)

        self.recursion_group = QGroupBox("Recursion")
        rec_form = QFormLayout(self.recursion_group)
        self.recursion_info = QLabel("")
        self.recursion_info.setWordWrap(True)
        self.recursion_info.setStyleSheet("color: #aaa; font-size: 11px;")
        rec_form.addRow(self.recursion_info)
        rec_row = QHBoxLayout()
        self.recursion_zoom = QLineEdit()
        self.recursion_zoom.setReadOnly(True)
        self.recursion_zoom.setStyleSheet("background: #2a2a2a;")
        self.recursion_apply_btn = QPushButton("Apply")
        self.recursion_apply_btn.clicked.connect(self._apply_recursion_zoom)
        rec_row.addWidget(self.recursion_zoom, 1)
        rec_row.addWidget(self.recursion_apply_btn)
        rec_form.addRow("Looping zoom factor:", rec_row)
        self.recursion_group.setVisible(False)
        rec_outer.addRow(self.recursion_group)

        layout.addWidget(grp)

        # ── Manual Zoom ──────────────────────────────────────────────────────
        grp = QGroupBox("Manual Zoom")
        mz_form = QFormLayout(grp)
        self.manual_zoom_factor = QDoubleSpinBox()
        self.manual_zoom_factor.setRange(0.01, 0.9999)
        self.manual_zoom_factor.setDecimals(4)
        self.manual_zoom_factor.setSingleStep(0.05)
        self.manual_zoom_factor.setValue(0.50)
        mz_form.addRow("Zoom factor:", self.manual_zoom_factor)

        self.manual_zoom_steps = QSpinBox()
        self.manual_zoom_steps.setRange(1, 1000)
        self.manual_zoom_steps.setValue(1)
        mz_form.addRow("Steps per click:", self.manual_zoom_steps)

        mz_info = QLabel("Click on the image in the Set Selection viewer\n"
                         "to zoom into those fractal coordinates.")
        mz_info.setStyleSheet("color: #888; font-size: 10px;")
        mz_info.setWordWrap(True)
        mz_form.addRow(mz_info)
        layout.addWidget(grp)

        layout.addStretch()
        self.setWidget(w)

        self.generator.currentTextChanged.connect(self._toggle_generator_options)
        self._toggle_generator_options()

        # Will be set by MainWindow to point at the sibling panel
        self._image_panel = None

    def _add_extra_constant(self, c_real=0.0, c_imag=0.0, start=0, length=10):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        cr = QDoubleSpinBox(); cr.setRange(-5, 5); cr.setDecimals(7); cr.setSingleStep(0.01); cr.setValue(c_real)
        ci = QDoubleSpinBox(); ci.setRange(-5, 5); ci.setDecimals(7); ci.setSingleStep(0.01); ci.setValue(c_imag)
        ts = QSpinBox(); ts.setRange(0, 100000); ts.setValue(start)
        tl = QSpinBox(); tl.setRange(0, 100000); tl.setValue(length)
        cr.setToolTip('Target c real'); ci.setToolTip('Target c imag')
        ts.setToolTip('Transition start frame'); tl.setToolTip('Transition length (frames)')
        rm = QPushButton('x'); rm.setFixedWidth(24)
        h.addWidget(QLabel('c:'))
        h.addWidget(cr, 1); h.addWidget(ci, 1)
        h.addWidget(QLabel('start:')); h.addWidget(ts)
        h.addWidget(QLabel('len:')); h.addWidget(tl)
        h.addWidget(rm)
        entry = {'widget': row, 'c_real': cr, 'c_imag': ci,
                 'start': ts, 'length': tl}
        rm.clicked.connect(lambda _=False, e=entry: self._remove_extra_constant(e))
        self._extra_const_vbox.addWidget(row)
        self._extra_const_rows.append(entry)
        return entry

    def _remove_extra_constant(self, entry):
        if entry in self._extra_const_rows:
            self._extra_const_rows.remove(entry)
        entry['widget'].setParent(None)
        entry['widget'].deleteLater()

    def extra_constants_list(self):
        rows = []
        for e in self._extra_const_rows:
            rows.append({
                'c_real': e['c_real'].value(),
                'c_imag': e['c_imag'].value(),
                'start':  e['start'].value(),
                'length': e['length'].value(),
            })
        return rows

    def extra_constants_json(self):
        rows = self.extra_constants_list()
        return json.dumps(rows) if rows else ''

    def apply_extra_constants(self, rows):
        for e in list(self._extra_const_rows):
            self._remove_extra_constant(e)
        if not rows:
            return
        for item in rows:
            try:
                self._add_extra_constant(
                    c_real=float(item.get('c_real', 0.0)),
                    c_imag=float(item.get('c_imag', 0.0)),
                    start=int(item.get('start', 0)),
                    length=int(item.get('length', 0)),
                )
            except (TypeError, ValueError):
                continue

    def _toggle_generator_options(self, *_args):
        gen = self.generator.currentText()
        self.flower_group.setVisible(gen == "flower")
        is_mandelbrot = (gen == "mandelbrot")
        for w in (self._c_real_label, self.c_real, self._c_imag_label, self.c_imag,
                  self._kernel_label, self.kernel):
            w.setVisible(not is_mandelbrot)

    def _apply_recursion_zoom(self):
        txt = self.recursion_zoom.text().strip()
        if txt and self._image_panel:
            self._image_panel.zoom_factor.setValue(float(txt))

    # ── Boundary helpers (run in worker threads, log via main_win) ──────────
    def _run_adjust_center(self):
        import threading
        self.adjust_center_btn.setEnabled(False)
        self.adjust_center_btn.setText("Searching\u2026")
        main_win = self.window()

        c = complex(self.c_real.value(), self.c_imag.value())
        center = complex(self.center_real.value(), self.center_imag.value())
        power = self.power.value()
        kernel = self.kernel.currentText()
        max_iter = self.max_iter.value()
        bailout = self.bailout.value()
        scale = self.scale_start.value()

        def _log(msg):
            main_win._pending_logs.append(msg)

        self._adjust_result = None

        def _work():
            try:
                _log(f"\n\U0001f3af Searching for nearest boundary point\u2026\n")
                _log(f"  center={center}, c={c}, kernel={kernel}, "
                     f"power={power}, scale={scale}\n")
                pt, dist = find_nearest_boundary_point(
                    c, center, kernel, power, max_iter, bailout, scale,
                    log=_log)
                self._adjust_result = (pt, dist)
            except Exception as e:
                import traceback
                _log(f"  \u26a0 Error: {e}\n{traceback.format_exc()}\n")
                self._adjust_result = (None, None)

        t = threading.Thread(target=_work, daemon=True)
        t.start()

        def _poll():
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(50, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            self.adjust_center_btn.setEnabled(True)
            self.adjust_center_btn.setText("\U0001f3af Adjust Center to Boundary")
            pt, dist = self._adjust_result
            if pt is not None:
                self.center_real.setValue(pt.real)
                self.center_imag.setValue(pt.imag)
                main_win.console.append_text(
                    f"\u2713 Center adjusted to {pt.real:+.10f} {pt.imag:+.10f}i "
                    f"(dist={dist:.2e})\n")
            else:
                main_win.console.append_text(
                    "\u26a0 Could not find a boundary point nearby.\n")
        QTimer.singleShot(50, _poll)

    def _run_suggest_clean_centers(self):
        import threading
        main_win = self.window()

        gen = self.generator.currentText()
        if gen == "mandelbrot":
            main_win.console.append_text(
                "\u26a0 Suggest Clean Centers currently supports Julia-type "
                "generators only.\n")
            return

        self.clean_center_btn.setEnabled(False)
        self.clean_center_btn.setText("Searching\u2026")

        c = complex(self.c_real.value(), self.c_imag.value())
        power = self.power.value()
        kernel = self.kernel.currentText()
        max_iter = self.max_iter.value()
        bailout = self.bailout.value()
        scale = self.scale_start.value()
        width = 1080
        zf = 0.92
        if self._image_panel is not None:
            try:
                zf = self._image_panel.zoom_factor.value()
            except Exception:
                pass
            try:
                width = self._image_panel.width.value()
            except Exception:
                pass

        def _log(msg):
            main_win._pending_logs.append(msg)

        self._clean_result = None

        def _work():
            try:
                _log(f"\n\u2728 Searching clean boundary points "
                     f"(c={c}, kernel={kernel}, power={power}, "
                     f"zoom={zf:g}, width={width})\u2026\n")
                self._clean_result = find_clean_boundary_points(
                    c, kernel, power, zf, scale, width,
                    max_iter, bailout, log=_log)
            except Exception as e:
                import traceback
                _log(f"  \u26a0 Error: {e}\n{traceback.format_exc()}\n")
                self._clean_result = []

        t = threading.Thread(target=_work, daemon=True)
        t.start()

        def _poll():
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(50, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            self.clean_center_btn.setEnabled(True)
            self.clean_center_btn.setText("\u2728 Suggest Clean Centers")
            candidates = self._clean_result or []
            if not candidates:
                main_win.console.append_text(
                    "\u26a0 No clean boundary points found.\n")
                return
            dlg = CleanCentersDialog(candidates, zf, parent=self)
            if dlg.exec() and dlg.selected is not None:
                z = dlg.selected
                self.center_real.setValue(z.real)
                self.center_imag.setValue(z.imag)
                main_win.console.append_text(
                    f"\u2713 Center set to {z.real:+.15g} {z.imag:+.15g}i\n")
        QTimer.singleShot(50, _poll)

    def _run_recursion_check(self):
        import threading
        self.recursion_check_btn.setEnabled(False)
        self.recursion_check_btn.setText("Checking\u2026")
        main_win = self.window()

        c = complex(self.c_real.value(), self.c_imag.value())
        center = complex(self.center_real.value(), self.center_imag.value())
        power = self.power.value()
        kernel = self.kernel.currentText()
        current_zf = self._image_panel.zoom_factor.value() if self._image_panel else 0.92
        ss = self.scale_start.value()

        def _log(msg):
            main_win._pending_logs.append(msg)

        def _check():
            try:
                _log(f"\n\U0001f50d Recursion check: c={c}, center={center}, "
                     f"kernel={kernel}, power={power}\n")
                return detect_recursion(c, center, kernel, power, current_zf,
                                        log=_log, scale_start=ss)
            except Exception as e:
                import traceback
                _log(f"  \u26a0 Error: {e}\n{traceback.format_exc()}\n")
                return None

        self._recursion_result = None
        def _thread():
            self._recursion_result = _check()
        t = threading.Thread(target=_thread, daemon=True)
        t.start()

        def _poll():
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(50, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            self.recursion_check_btn.setEnabled(True)
            self.recursion_check_btn.setText("\U0001f50d Check Recursion")
            result = self._recursion_result
            self._show_recursion_result(result)
            if self._image_panel:
                self._image_panel._show_recursion_result(result)
        QTimer.singleShot(50, _poll)

    def _show_recursion_result(self, result):
        if result is not None:
            self.recursion_group.setVisible(True)
            zf = result["optimal_zf"]
            self.recursion_zoom.setText(f"{zf:.9f}")
            typ = result["type"].replace("_", " ").title()
            quality = result.get("quality", "")
            info_lines = [f"{typ} (period {result['period']})"]
            info_lines.append(f"|\u03bb| = {result['abs_lam']:.8f}, arg = {result['arg_lam_deg']:.2f}\u00b0")
            if result["type"] == "repelling_fixed":
                info_lines.append(
                    f"{result['n_rot']} self-sim steps \u00d7 {result['arg_lam_deg']:.1f}\u00b0 = "
                    f"{result['n_rot'] * result['arg_lam_deg']:.0f}\u00b0")
            loop_start = result.get('loop_start_frame', 0)
            info_lines.append(f"Loop every {result['k']} frames (zoom = {zf:.9f}) [{quality}]")
            info_lines.append(f"Loop starts at frame {loop_start}")
            self.recursion_info.setText("\n".join(info_lines))
        else:
            self.recursion_group.setVisible(False)

    def _run_preview(self, size):
        import threading
        from time import perf_counter as clock
        self.preview_lo.setEnabled(False)
        self.preview_hi.setEnabled(False)
        self.preview_lo.setText("Rendering\u2026")
        self.preview_hi.setText("Rendering\u2026")
        main_win = self.window()
        def _log(msg):
            main_win._pending_logs.append(msg)

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
            recenter=self.recenter.isChecked(),
        )
        if gen_name == "flower":
            kwargs.update(
                petals=self.petals.value(),
                mirror_segments=self.mirror_segments.isChecked(),
                interest_angle=self.interest_angle.value(),
                align_north=self.align_north.currentText(),
            )
        scale = self.scale_start.value()
        mask_path = self._image_panel.mask_svg.text().strip() if self._image_panel else ""

        def _do_render():
            try:
                _log(f"\n\u23f3 Preview {size}\u00d7{size} | {gen_name} kernel={kwargs['kernel']} "
                     f"c={kwargs['c']} scale={scale}\n")
                _log(f"  Creating generator ({gen_name})\u2026\n")
                t0 = clock()
                gen = gen_cls(**kwargs)
                _log(f"  Generator ready ({clock()-t0:.2f}s)\n")
                mask = None
                if mask_path:
                    _log(f"  Loading mask: {mask_path}\u2026\n")
                    t0 = clock()
                    from mask import load_svg_mask, radial_fill
                    display_mask, _ = load_svg_mask(mask_path, size, size)
                    mask = radial_fill(display_mask)
                    _log(f"  Mask ready ({clock()-t0:.2f}s)\n")
                _log(f"  Rendering counts ({size}\u00d7{size}, max_iter={kwargs['max_iter']})\u2026\n")
                t0 = clock()
                counts = gen.render(scale, mask=mask)
                _log(f"  Render done ({clock()-t0:.2f}s)\n")
                preview_dir, _preview_num = next_preview_folder()
                import numpy as np
                _log(f"  Saving greyscale preview\u2026\n")
                t0 = clock()
                esc = counts < kwargs["max_iter"]
                grey = np.zeros(counts.shape, dtype=np.uint8)
                if esc.any():
                    vals = counts[esc].astype(np.float64)
                    lo, hi = np.percentile(vals, [1, 99])
                    if hi <= lo: hi = lo + 1.0
                    grey[esc] = (np.clip((vals - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                from PIL import Image, ImageDraw, ImageFont
                img = Image.fromarray(grey, mode='L')
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
                save_preview_params(preview_dir, {
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
                    "extra_constants": self.extra_constants_list(),
                    **({"petals": kwargs.get("petals"),
                        "mirror_segments": kwargs.get("mirror_segments"),
                        "interest_angle": kwargs.get("interest_angle"),
                        "align_north": kwargs.get("align_north")}
                       if gen_name == "flower" else {}),
                })
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
        def _poll():
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(100, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            self.preview_lo.setEnabled(True)
            self.preview_hi.setEnabled(True)
            self.preview_lo.setText("Preview 500\u00d7500")
            self.preview_hi.setText("Preview 2160\u00d72160")
            result = self._preview_result
            if isinstance(result, str):
                main_win.console.append_text(f"\n\u26a0 Preview error: {result}\n")
            elif result:
                preview_dir = os.path.dirname(result[0])
                main_win.preview.load_folder(preview_dir)
                main_win.preview.refresh_quick_select()
                main_win.console.append_text(f"\n\u2713 Preview rendered: {len(result)} images ({size}\u00d7{size})\n")
        QTimer.singleShot(100, _poll)

    def _run_zoom_preview(self):
        import threading
        from time import perf_counter as clock
        size = 500
        depth = self.zoom_depth.value()
        n_samples = 11
        step = max(1, (depth - 1) // (n_samples - 1))
        frame_indices = list(range(0, depth, step))
        if frame_indices[-1] != depth - 1:
            frame_indices.append(depth - 1)
        self.zoom_preview_btn.setEnabled(False)
        self.zoom_preview_btn.setText("Rendering\u2026")
        main_win = self.window()
        def _log(msg):
            main_win._pending_logs.append(msg)

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
            recenter=self.recenter.isChecked(),
        )
        if gen_name == "flower":
            kwargs.update(
                petals=self.petals.value(),
                mirror_segments=self.mirror_segments.isChecked(),
                interest_angle=self.interest_angle.value(),
                align_north=self.align_north.currentText(),
            )
        scale_start = self.scale_start.value()
        zoom_factor = self._image_panel.zoom_factor.value() if self._image_panel else 0.92
        mask_path = self._image_panel.mask_svg.text().strip() if self._image_panel else ""
        try:
            from render import parse_extra_constants, compute_c_for_frame
            _extras = parse_extra_constants(self.extra_constants_json())
        except Exception:
            _extras = []
            compute_c_for_frame = None
        _base_c = kwargs["c"]

        def _do_render():
            try:
                import numpy as np
                from PIL import Image as PILImage
                _log(f"\n\u23f3 Zoom preview {size}\u00d7{size} | depth={depth} "
                     f"| {len(frame_indices)} samples | step={step}\n")
                t0_total = clock()
                gen = gen_cls(**kwargs)
                mask = None
                if mask_path:
                    _log(f"  Loading mask: {mask_path}\u2026\n")
                    t0 = clock()
                    from mask import load_svg_mask, radial_fill
                    display_mask, _ = load_svg_mask(mask_path, size, size)
                    mask = radial_fill(display_mask)
                    _log(f"  Mask ready ({clock()-t0:.2f}s)\n")
                preview_dir, _preview_num = next_preview_folder()
                save_preview_params(preview_dir, {
                    "generator": gen_name,
                    "kernel": kwargs["kernel"],
                    "power": kwargs["power"],
                    "c_real": kwargs["c"].real,
                    "c_imag": kwargs["c"].imag,
                    "center_real": kwargs["center"].real,
                    "center_imag": kwargs["center"].imag,
                    "scale_start": scale_start,
                    "zoom_factor": zoom_factor,
                    "depth": depth,
                    "max_iter": kwargs["max_iter"],
                    "bailout": kwargs["bailout"],
                    "recenter": kwargs["recenter"],
                    "extra_constants": self.extra_constants_list(),
                    **({"petals": kwargs.get("petals"),
                        "mirror_segments": kwargs.get("mirror_segments"),
                        "interest_angle": kwargs.get("interest_angle"),
                        "align_north": kwargs.get("align_north")}
                       if gen_name == "flower" else {}),
                })
                saved = []
                for i, fi in enumerate(frame_indices):
                    scale = scale_start * (zoom_factor ** fi)
                    if _extras and compute_c_for_frame is not None:
                        gen.c = compute_c_for_frame(_base_c, _extras, fi)
                    _msg = f"  [{i+1}/{len(frame_indices)}] frame {fi}, scale={scale:.6f}"
                    if _extras:
                        _msg += f", c={gen.c}"
                    _log(_msg + "\u2026\n")
                    t0 = clock()
                    counts = gen.render(scale, mask=mask)
                    esc = counts < kwargs["max_iter"]
                    grey = np.zeros(counts.shape, dtype=np.uint8)
                    if esc.any():
                        vals = counts[esc].astype(np.float64)
                        lo, hi = np.percentile(vals, [1, 99])
                        if hi <= lo: hi = lo + 1.0
                        grey[esc] = (np.clip((vals - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                    img = PILImage.fromarray(grey, mode='L')
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
                main_win.console.append_text(main_win._pending_logs.pop(0))
            if t.is_alive():
                QTimer.singleShot(100, _poll)
                return
            while main_win._pending_logs:
                main_win.console.append_text(main_win._pending_logs.pop(0))
            self.zoom_preview_btn.setEnabled(True)
            self.zoom_preview_btn.setText("Preview Zoom Series (500\u00d7500)")
            result = self._zoom_preview_result
            if isinstance(result, str):
                main_win.console.append_text(f"\n\u26a0 Zoom preview error: {result}\n")
            elif result:
                preview_dir = os.path.dirname(result[0])
                main_win.preview.load_folder(preview_dir)
                main_win.preview.refresh_quick_select()
                main_win.console.append_text(f"\n\u2713 Zoom preview: {len(result)} frames rendered\n")
        QTimer.singleShot(100, _poll)


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Rendering Panel  (Tab 2)
# ═══════════════════════════════════════════════════════════════════════════════
class ImageRenderingPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        self._set_panel = None  # set by MainWindow

        # ── Zoom Render Details ──────────────────────────────────────────────
        grp = QGroupBox("Zoom Render Details")
        form = QFormLayout(grp)

        self.recursion_check_btn = QPushButton("\U0001f50d Check Recursion")
        self.recursion_check_btn.clicked.connect(self._run_recursion_check)
        form.addRow("", self.recursion_check_btn)

        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setRange(0.5, 0.999999999)
        self.zoom_factor.setDecimals(9)
        self.zoom_factor.setSingleStep(0.01)
        self.zoom_factor.setValue(0.92)
        self.num_frames = QSpinBox(); self.num_frames.setRange(1, 100000); self.num_frames.setValue(30)
        self.start_frame = QSpinBox(); self.start_frame.setRange(0, 100000); self.start_frame.setValue(0)
        self.frame_step = QSpinBox(); self.frame_step.setRange(1, 100); self.frame_step.setValue(1)
        form.addRow("Zoom factor:", self.zoom_factor)
        form.addRow("Num frames:", self.num_frames)
        form.addRow("Start frame:", self.start_frame)
        form.addRow("Frame step:", self.frame_step)

        layout.addWidget(grp)

        # ── Recursion (hidden until detected) ────────────────────────────────
        self.recursion_group = QGroupBox("Recursion")
        rec_form = QFormLayout(self.recursion_group)
        self.recursion_info = QLabel("")
        self.recursion_info.setWordWrap(True)
        self.recursion_info.setStyleSheet("color: #aaa; font-size: 11px;")
        rec_form.addRow(self.recursion_info)
        rec_row = QHBoxLayout()
        self.recursion_zoom = QLineEdit()
        self.recursion_zoom.setReadOnly(True)
        self.recursion_zoom.setStyleSheet("background: #2a2a2a;")
        self.recursion_apply_btn = QPushButton("Apply")
        self.recursion_apply_btn.clicked.connect(self._apply_recursion_zoom)
        rec_row.addWidget(self.recursion_zoom, 1)
        rec_row.addWidget(self.recursion_apply_btn)
        rec_form.addRow("Looping zoom factor:", rec_row)
        self.recursion_group.setVisible(False)
        layout.addWidget(self.recursion_group)

        # ── Quality ──────────────────────────────────────────────────────────
        grp = QGroupBox("Quality")
        form = QFormLayout(grp)
        self.width = QSpinBox(); self.width.setRange(64, 7680); self.width.setSingleStep(64); self.width.setValue(640)
        self.height = QSpinBox(); self.height.setRange(64, 4320); self.height.setSingleStep(64); self.height.setValue(640)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(1024)
        self.bailout = QDoubleSpinBox(); self.bailout.setRange(0.1, 10000); self.bailout.setDecimals(1); self.bailout.setValue(2.0)
        self.scale_start = QDoubleSpinBox(); self.scale_start.setRange(0.01, 100); self.scale_start.setDecimals(2); self.scale_start.setValue(2.0)
        form.addRow("Width:", self.width)
        form.addRow("Height:", self.height)
        form.addRow("Max iterations:", self.max_iter)
        form.addRow("Bailout:", self.bailout)
        form.addRow("Scale start:", self.scale_start)
        layout.addWidget(grp)

        # ── Execution ────────────────────────────────────────────────────────
        grp = QGroupBox("Execution")
        form = QFormLayout(grp)
        self.workers = QSpinBox(); self.workers.setRange(1, 64); self.workers.setValue(12)
        form.addRow("Workers:", self.workers)
        self.mask_svg = QLineEdit()
        mask_btn = QPushButton("Browse\u2026")
        mask_btn.clicked.connect(self._pick_mask)
        h = QHBoxLayout()
        h.addWidget(self.mask_svg, 1)
        h.addWidget(mask_btn)
        form.addRow("Mask SVG:", h)
        self.series_name = QLineEdit()
        self.series_name.setPlaceholderText("(auto-generated if empty)")
        self.series_name.setToolTip(
            "Optional custom name for the results/<series>/ folder.\n"
            "Leave empty to use the auto-generated name.")
        form.addRow("Series name:", self.series_name)
        layout.addWidget(grp)

        # ── Start button ─────────────────────────────────────────────────────
        self.start_btn = QPushButton("\u25b6  Start Render")
        self.start_btn.setStyleSheet("padding: 8px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.start_btn)

        layout.addStretch()
        self.setWidget(w)

    def _pick_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SVG mask", PROJECT_DIR, "SVG Files (*.svg)")
        if path:
            self.mask_svg.setText(os.path.relpath(path, PROJECT_DIR))

    def _show_recursion_result(self, result):
        if result is not None:
            self.recursion_group.setVisible(True)
            zf = result["optimal_zf"]
            self.recursion_zoom.setText(f"{zf:.9f}")
            typ = result["type"].replace("_", " ").title()
            quality = result.get("quality", "")
            info_lines = [f"{typ} (period {result['period']})"]
            info_lines.append(f"|\u03bb| = {result['abs_lam']:.8f}, arg = {result['arg_lam_deg']:.2f}\u00b0")
            if result["type"] == "repelling_fixed":
                info_lines.append(
                    f"{result['n_rot']} self-sim steps \u00d7 {result['arg_lam_deg']:.1f}\u00b0 = "
                    f"{result['n_rot'] * result['arg_lam_deg']:.0f}\u00b0")
            loop_start = result.get('loop_start_frame', 0)
            info_lines.append(f"Loop every {result['k']} frames (zoom = {zf:.9f}) [{quality}]")
            info_lines.append(f"Loop starts at frame {loop_start}")
            self.recursion_info.setText("\n".join(info_lines))
        else:
            self.recursion_group.setVisible(False)

    def _run_recursion_check(self):
        if self._set_panel:
            self._set_panel._run_recursion_check()

    def _apply_recursion_zoom(self):
        txt = self.recursion_zoom.text().strip()
        if txt:
            self.zoom_factor.setValue(float(txt))

    def build_args(self):
        sp = self._set_panel
        if not sp:
            return []
        args = [
            "--generator", sp.generator.currentText(),
            "--kernel", sp.kernel.currentText(),
            "--c-real", str(sp.c_real.value()),
            "--c-imag", str(sp.c_imag.value()),
            "--center-real", str(sp.center_real.value()),
            "--center-imag", str(sp.center_imag.value()),
            "--width", str(self.width.value()),
            "--height", str(self.height.value()),
            "--scale-start", str(self.scale_start.value()),
            "--zoom-factor", str(self.zoom_factor.value()),
            "--num-frames", str(self.num_frames.value()),
            "--start-frame", str(self.start_frame.value()),
            "--frame-step", str(self.frame_step.value()),
            "--max-iter", str(self.max_iter.value()),
            "--power", str(sp.power.value()),
            "--bailout", str(self.bailout.value()),
            "--workers", str(self.workers.value()),
        ]
        if sp.generator.currentText() == "flower":
            args += [
                "--petals", str(sp.petals.value()),
                "--interest-angle", str(sp.interest_angle.value()),
                "--align-north", sp.align_north.currentText(),
            ]
            if sp.mirror_segments.isChecked():
                args.append("--mirror-segments")
        if self.mask_svg.text().strip():
            args += ["--mask-svg", self.mask_svg.text().strip()]
        if sp.recenter.isChecked():
            args.append("--recenter")
        name = self.series_name.text().strip()
        if name:
            args += ["--series-name", name]
        try:
            ec = sp.extra_constants_json()
        except AttributeError:
            ec = ''
        if ec:
            args += ["--extra-constants", ec]
        return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Coloring Panel
# ════════════════════════════════════��══════════════════════════════════════════
class SegmentWidget(QGroupBox):
    """One entry in the colour program."""
    removed = pyqtSignal(object)

    def __init__(self, is_first=False, parent=None):
        super().__init__("Base colour" if is_first else "Additional colour",
                         parent)
        self.is_first = is_first

        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        if not is_first:
            self.start_frame = QSpinBox()
            self.start_frame.setRange(0, 100000)
            self.start_frame.setValue(0)
            form.addRow("Starting frame:", self.start_frame)

            self.transition_length = QSpinBox()
            self.transition_length.setRange(0, 10000)
            self.transition_length.setValue(10)
            form.addRow("Transition length:", self.transition_length)

        self.colorizer = QComboBox()
        for name in COLORIZERS:
            self.colorizer.addItem(name)
        default = "twilight" if is_first else "log_fire"
        idx = self.colorizer.findText(default)
        if idx >= 0:
            self.colorizer.setCurrentIndex(idx)
        form.addRow("Colorizer:", self.colorizer)

        self.invert = QCheckBox("invert gradient")
        form.addRow("", self.invert)

        self.force_bg = QComboBox()
        self.force_bg.addItems(["(none)", "black", "white"])
        form.addRow("Force background:", self.force_bg)

        if not is_first:
            remove_btn = QPushButton("\u2716  Remove")
            remove_btn.setStyleSheet("color: #d77;")
            remove_btn.clicked.connect(lambda: self.removed.emit(self))
            form.addRow("", remove_btn)

    def to_dict(self):
        bg_txt = self.force_bg.currentText()
        d = {
            "colorizer": self.colorizer.currentText(),
            "invert": self.invert.isChecked(),
            "force_bg": None if bg_txt == "(none)" else bg_txt,
        }
        if self.is_first:
            d["start_frame"] = 0
            d["transition_length"] = 0
        else:
            d["start_frame"] = self.start_frame.value()
            d["transition_length"] = self.transition_length.value()
        return d


class ColoringPanel(QScrollArea):
    """Pick a rendered raw series, build a colour program, dispatch colorize.py."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        # ── Source series ───────────────────────────────────────────────────
        grp = QGroupBox("Source Series")
        form = QFormLayout(grp)

        row_root = QHBoxLayout()
        self.results_folder = QLineEdit(RESULTS_DIR)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._pick_results_folder)
        row_root.addWidget(self.results_folder, 1)
        row_root.addWidget(browse_btn)
        form.addRow("Results root:", row_root)

        row_series = QHBoxLayout()
        self.series_combo = QComboBox()
        self.series_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.series_combo.currentTextChanged.connect(self._on_series_changed)
        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self._refresh_series)
        row_series.addWidget(self.series_combo, 1)
        row_series.addWidget(refresh_btn)
        form.addRow("Series:", row_series)

        self.series_info = QLabel("")
        self.series_info.setStyleSheet("color: #888; font-size: 11px;")
        self.series_info.setWordWrap(True)
        form.addRow("", self.series_info)

        layout.addWidget(grp)

        # ── Colour program ──────────────────────────────────────────────────
        program_grp = QGroupBox("Colour Program")
        prog_layout = QVBoxLayout(program_grp)
        prog_layout.setContentsMargins(8, 8, 8, 8)

        self._segments_container = QWidget()
        self._segments_layout = QVBoxLayout(self._segments_container)
        self._segments_layout.setContentsMargins(0, 0, 0, 0)
        self._segments_layout.setSpacing(6)
        prog_layout.addWidget(self._segments_container)

        self.add_color_btn = QPushButton("\u2795  Add colour")
        self.add_color_btn.clicked.connect(self._add_segment)
        prog_layout.addWidget(self.add_color_btn)

        layout.addWidget(program_grp)

        self._segments = []
        self._add_segment(is_first=True)

        # ── Options ─────────────────────────────────────────────────────────
        grp = QGroupBox("Options")
        form = QFormLayout(grp)
        self.scale_smoothing = QDoubleSpinBox()
        self.scale_smoothing.setRange(0.0, 0.99)
        self.scale_smoothing.setDecimals(2)
        self.scale_smoothing.setSingleStep(0.05)
        self.scale_smoothing.setValue(0.7)
        form.addRow("Scale smoothing:", self.scale_smoothing)

        self.workers = QSpinBox()
        self.workers.setRange(1, 64)
        self.workers.setValue(12)
        form.addRow("Workers:", self.workers)

        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("(auto-derived from segments)")
        form.addRow("Output folder:", self.output_name)

        self.reuse_meta = QCheckBox("Use cached bounds from meta.json")
        self.reuse_meta.setChecked(True)
        self.reuse_meta.setToolTip(
            "render.py already computed raw bounds per frame – reusing them "
            "skips Pass 1 of colorize.py and makes re-coloring almost instant.")
        form.addRow("", self.reuse_meta)

        layout.addWidget(grp)

        # ── Start button ────────────────────────────────────────────────────
        self.start_btn = QPushButton("\U0001f3a8  Start Coloring")
        self.start_btn.setStyleSheet(
            "padding: 8px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.start_btn)

        layout.addStretch()
        self.setWidget(w)

        QTimer.singleShot(100, self._refresh_series)

    def _add_segment(self, is_first=False):
        seg = SegmentWidget(is_first=is_first)
        if not is_first:
            seg.removed.connect(self._remove_segment)
            last = self._segments[-1] if self._segments else None
            if last is not None and not last.is_first:
                seg.start_frame.setValue(
                    last.start_frame.value() + last.transition_length.value() + 40)
            else:
                seg.start_frame.setValue(30)
        self._segments.append(seg)
        self._segments_layout.addWidget(seg)

    def _remove_segment(self, seg):
        if seg.is_first:
            return
        if seg in self._segments:
            self._segments.remove(seg)
        self._segments_layout.removeWidget(seg)
        seg.setParent(None)
        seg.deleteLater()

    def _pick_results_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select results folder", RESULTS_DIR)
        if path:
            self.results_folder.setText(path)
            self._refresh_series()

    def _refresh_series(self):
        self.series_combo.blockSignals(True)
        current = self.series_combo.currentText()
        self.series_combo.clear()
        root = self.results_folder.text().strip()
        if os.path.isdir(root):
            dirs = []
            for d in sorted(os.listdir(root)):
                full = os.path.join(root, d)
                if (os.path.isdir(full) and not d.startswith("_")
                        and os.path.isdir(os.path.join(full, "_raw"))):
                    dirs.append(d)
            self.series_combo.addItems(dirs)
        idx = self.series_combo.findText(current)
        if idx >= 0:
            self.series_combo.setCurrentIndex(idx)
        self.series_combo.blockSignals(False)
        self._on_series_changed()

    def _on_series_changed(self):
        series = self.series_combo.currentText()
        root = self.results_folder.text().strip()
        if not series or not root:
            self.series_info.setText("")
            return
        folder = os.path.join(root, series)
        meta_path = os.path.join(folder, "_raw", "meta.json")
        info_bits = []
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                info_bits.append(
                    f"{meta.get('generator', '?')}"
                    + (f"/{meta['kernel']}" if meta.get("kernel") != "poly" else "")
                    + f"   {meta.get('width')}\u00d7{meta.get('height')}"
                    + f"   max_iter={meta.get('max_iter')}")
                c = meta.get("c", [None, None])
                if c[0] is not None:
                    info_bits.append(f"c = {c[0]:+g} {c[1]:+g}i")
                info_bits.append(
                    f"{len(meta.get('per_frame_bounds') or {})} frame(s) "
                    f"with cached bounds")
            except Exception as e:
                info_bits.append(f"(could not parse meta.json: {e})")
        else:
            info_bits.append("No meta.json — this series cannot be "
                             "re-colorized automatically.")
        try:
            existing = [d for d in sorted(os.listdir(folder))
                        if os.path.isdir(os.path.join(folder, d))
                        and not d.startswith("_")]
            if existing:
                info_bits.append("Existing: " + ", ".join(existing))
        except OSError:
            pass
        self.series_info.setText("\n".join(info_bits))

    def build_program(self):
        segments = [s.to_dict() for s in self._segments]
        prog = {
            "smoothing": self.scale_smoothing.value(),
            "segments": segments,
        }
        if self.output_name.text().strip():
            prog["output_name"] = self.output_name.text().strip()
        return prog

    def apply_program(self, prog):
        """Replace the current colour program with the values from *prog*.

        ``prog`` is the dict written to ``program.json`` by colorize.py:
            { "smoothing": float,
              "output_name": str (optional),
              "segments": [
                  {"colorizer": str, "invert": bool, "force_bg": str|None,
                   "start_frame": int, "transition_length": int}, ... ] }
        Segment frame indices are user-facing (rendered-image units), so
        they can be written back into the spinboxes verbatim.
        """
        if not isinstance(prog, dict):
            return
        # Smoothing + output name
        try:
            self.scale_smoothing.setValue(float(prog.get("smoothing", 0.7)))
        except (TypeError, ValueError):
            pass
        self.output_name.setText(str(prog.get("output_name", "") or ""))

        segments = prog.get("segments") or []
        if not segments:
            return

        # Wipe current segments completely, then rebuild from the program.
        for seg in list(self._segments):
            self._segments_layout.removeWidget(seg)
            seg.setParent(None)
            seg.deleteLater()
        self._segments = []

        for i, seg_data in enumerate(segments):
            self._add_segment(is_first=(i == 0))
            seg_widget = self._segments[-1]
            # Colorizer
            cname = seg_data.get("colorizer", "")
            if cname:
                idx = seg_widget.colorizer.findText(cname)
                if idx >= 0:
                    seg_widget.colorizer.setCurrentIndex(idx)
            # Invert
            seg_widget.invert.setChecked(bool(seg_data.get("invert", False)))
            # Force background
            bg = seg_data.get("force_bg")
            bg_text = bg if bg in ("black", "white") else "(none)"
            bg_idx = seg_widget.force_bg.findText(bg_text)
            if bg_idx >= 0:
                seg_widget.force_bg.setCurrentIndex(bg_idx)
            # Start frame / transition length (only for non-base segments)
            if i > 0:
                try:
                    seg_widget.start_frame.setValue(
                        int(seg_data.get("start_frame", 0)))
                except (TypeError, ValueError):
                    pass
                try:
                    seg_widget.transition_length.setValue(
                        int(seg_data.get("transition_length", 0)))
                except (TypeError, ValueError):
                    pass

    def _write_program_file(self):
        import time
        programs_dir = os.path.join(PROJECT_DIR, "_programs")
        os.makedirs(programs_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(programs_dir, f"program_{stamp}.json")
        with open(path, "w") as f:
            json.dump(self.build_program(), f, indent=2)
        return path

    def build_args(self):
        root = self.results_folder.text().strip()
        series = self.series_combo.currentText()
        if not series or not root:
            return None
        folder = os.path.join(root, series)
        if not self._segments:
            return None
        program_path = self._write_program_file()
        args = [folder,
                "--program", program_path,
                "--scale-smoothing", str(self.scale_smoothing.value()),
                "--workers", str(self.workers.value())]
        if self.output_name.text().strip():
            args += ["--output-name", self.output_name.text().strip()]
        if not self.reuse_meta.isChecked():
            args.append("--no-meta-bounds")
        return args

    def current_series_folder(self):
        root = self.results_folder.text().strip()
        series = self.series_combo.currentText()
        if not series or not root:
            return None
        return os.path.join(root, series)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stitch Panel  (transition between coloured series with zoom-overlay)
# ═══════════════════════════════════════════════════════════════════════════════
class StitchSegmentWidget(QGroupBox):
    """One segment in a stitch program."""
    removed = pyqtSignal(object)

    def __init__(self, results_root, index, is_first=False, parent=None):
        super().__init__(parent)
        self.is_first = is_first
        self.index = index
        self._results_root = results_root
        self.setTitle("Base series" if is_first else f"Series #{index + 1}")

        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)

        self.series_combo = QComboBox()
        self.series_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._populate_series()
        form.addRow("Series / colour:", self.series_combo)

        refresh_btn = QPushButton("\u21bb  Refresh list")
        refresh_btn.clicked.connect(self._populate_series)
        form.addRow("", refresh_btn)

        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, 100000)
        self.start_frame.setValue(0)
        form.addRow("Entry frame:", self.start_frame)

        self.end_frame = QSpinBox()
        self.end_frame.setRange(0, 100000)
        self.end_frame.setValue(100)
        form.addRow("Exit frame:", self.end_frame)

        if not is_first:
            self.transition_length = QSpinBox()
            self.transition_length.setRange(1, 10000)
            self.transition_length.setValue(80)
            self.transition_length.setToolTip(
                "Number of frames at the end of the previous segment\n"
                "over which this segment's entry frame grows from a few\n"
                "pixels to full screen.")
            form.addRow("Transition length:", self.transition_length)

            remove_btn = QPushButton("\u2716  Remove")
            remove_btn.setStyleSheet("color: #d77;")
            remove_btn.clicked.connect(lambda: self.removed.emit(self))
            form.addRow("", remove_btn)

    def set_results_root(self, root):
        self._results_root = root
        self._populate_series()

    def _populate_series(self):
        current = self.series_combo.currentText()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        entries = []
        root = self._results_root
        if os.path.isdir(root):
            for series in sorted(os.listdir(root)):
                sfull = os.path.join(root, series)
                if not os.path.isdir(sfull) or series.startswith("_"):
                    continue
                for sub in sorted(os.listdir(sfull)):
                    subfull = os.path.join(sfull, sub)
                    if (os.path.isdir(subfull) and not sub.startswith("_")
                            and glob.glob(os.path.join(subfull,
                                                      "frame_*.png"))):
                        entries.append(f"{series}/{sub}")
        self.series_combo.addItems(entries)
        idx = self.series_combo.findText(current)
        if idx >= 0:
            self.series_combo.setCurrentIndex(idx)
        self.series_combo.blockSignals(False)

    def folder(self):
        rel = self.series_combo.currentText().strip()
        if not rel:
            return None
        return os.path.join(self._results_root, rel)

    def to_dict(self):
        d = {
            "folder": self.folder() or "",
            "start_frame": int(self.start_frame.value()),
            "end_frame": int(self.end_frame.value()),
        }
        if not self.is_first:
            d["transition_length"] = int(self.transition_length.value())
        return d


class StitchPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        root_grp = QGroupBox("Results root")
        root_form = QFormLayout(root_grp)
        row = QHBoxLayout()
        self.results_folder = QLineEdit(RESULTS_DIR)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._pick_results_folder)
        row.addWidget(self.results_folder, 1)
        row.addWidget(browse_btn)
        root_form.addRow("Folder:", row)
        layout.addWidget(root_grp)

        prog_grp = QGroupBox("Stitch program")
        prog_layout = QVBoxLayout(prog_grp)
        prog_layout.setContentsMargins(8, 8, 8, 8)

        self._segments_container = QWidget()
        self._segments_layout = QVBoxLayout(self._segments_container)
        self._segments_layout.setContentsMargins(0, 0, 0, 0)
        self._segments_layout.setSpacing(6)
        prog_layout.addWidget(self._segments_container)

        self.add_series_btn = QPushButton("\u2795  Add series")
        self.add_series_btn.clicked.connect(lambda: self._add_segment(False))
        prog_layout.addWidget(self.add_series_btn)

        layout.addWidget(prog_grp)

        opt_grp = QGroupBox("Options")
        opt_form = QFormLayout(opt_grp)
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("stitched_<timestamp>")
        opt_form.addRow("Output name:", self.output_name)
        layout.addWidget(opt_grp)

        self.start_btn = QPushButton("\U0001f517  Start Stitching")
        self.start_btn.setStyleSheet(
            "padding: 8px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.start_btn)

        layout.addStretch()
        self.setWidget(w)

        self._segments = []
        self._add_segment(is_first=True)
        self._add_segment(is_first=False)

    def _pick_results_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select results folder", RESULTS_DIR)
        if path:
            self.results_folder.setText(path)
            for seg in self._segments:
                seg.set_results_root(path)

    def _add_segment(self, is_first=False):
        is_first = bool(is_first) if is_first is True else (
            not self._segments)
        idx = len(self._segments)
        seg = StitchSegmentWidget(
            self.results_folder.text().strip(), idx, is_first=is_first)
        if not is_first:
            seg.removed.connect(self._remove_segment)
        self._segments.append(seg)
        self._segments_layout.addWidget(seg)

    def _remove_segment(self, seg):
        if seg.is_first:
            return
        if seg in self._segments:
            self._segments.remove(seg)
        self._segments_layout.removeWidget(seg)
        seg.setParent(None)
        seg.deleteLater()
        for i, s in enumerate(self._segments):
            s.index = i
            if not s.is_first:
                s.setTitle(f"Series #{i + 1}")

    def build_program(self):
        segments = [s.to_dict() for s in self._segments]
        segments = [s for s in segments if s["folder"]]
        prog = {"segments": segments}
        out = self.output_name.text().strip()
        if out:
            prog["output_name"] = out
        return prog

    def _write_program_file(self):
        import time
        programs_dir = os.path.join(PROJECT_DIR, "_programs")
        os.makedirs(programs_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(programs_dir, f"stitch_{stamp}.json")
        with open(path, "w") as f:
            json.dump(self.build_program(), f, indent=2)
        return path

    def build_args(self):
        prog = self.build_program()
        if len(prog.get("segments", [])) < 2:
            return None
        path = self._write_program_file()
        args = [path]
        if self.output_name.text().strip():
            args += ["--output-name", self.output_name.text().strip()]
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

