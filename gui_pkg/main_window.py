"""Main window assembling all panels + viewers, plus the application entry point."""
import os
import sys
import signal

from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTabWidget, QMessageBox,
)

from .common import PROJECT_DIR, RESULTS_DIR, SCRIPT_DIR, PYTHON
from .analysis import detect_recursion
from .panels import (
    SetSelectionPanel, ImageRenderingPanel, ColoringPanel,
    StitchPanel, VideoPanel,
)
from .viewers import TabbedPreview, ConsoleWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Studio")
        self.resize(1400, 900)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: parameter tabs
        self.left_tabs = QTabWidget()
        self.set_panel = SetSelectionPanel()
        self.image_panel = ImageRenderingPanel()
        self.coloring_panel = ColoringPanel()
        self.stitch_panel = StitchPanel()
        self.video_panel = VideoPanel()
        self.left_tabs.addTab(self.set_panel, "Set Selection")
        self.left_tabs.addTab(self.image_panel, "Image Rendering")
        self.left_tabs.addTab(self.coloring_panel, "Coloring")
        self.left_tabs.addTab(self.stitch_panel, "Stitch")
        self.left_tabs.addTab(self.video_panel, "Video")
        self.left_tabs.setMaximumWidth(420)
        splitter.addWidget(self.left_tabs)

        # Cross-references
        self.set_panel._image_panel = self.image_panel
        self.image_panel._set_panel = self.set_panel

        def _link_spinboxes(a, b):
            a.valueChanged.connect(lambda v: b.setValue(v) if b.value() != v else None)
            b.valueChanged.connect(lambda v: a.setValue(v) if a.value() != v else None)
        _link_spinboxes(self.set_panel.max_iter, self.image_panel.max_iter)
        _link_spinboxes(self.set_panel.bailout, self.image_panel.bailout)
        _link_spinboxes(self.set_panel.scale_start, self.image_panel.scale_start)

        self.render_panel = self.image_panel

        # Right: tabbed preview + console
        right = QSplitter(Qt.Orientation.Vertical)

        self.preview = TabbedPreview()
        self.preview.set_viewer._set_panel = self.set_panel
        self.preview.render_viewer._set_panel = self.set_panel
        self.preview.render_viewer._image_panel = self.image_panel
        right.addWidget(self.preview)

        self._link_series_dropdowns(
            self.coloring_panel, self.preview.render_viewer)

        self.console = ConsoleWidget()
        right.addWidget(self.console)

        right.setSizes([500, 300])
        splitter.addWidget(right)
        splitter.setSizes([380, 1020])
        self.setCentralWidget(splitter)

        self.left_tabs.currentChanged.connect(self._sync_viewer_tab)

        self.statusBar().showMessage("Ready")

        self.image_panel.start_btn.clicked.connect(self._start_render)
        self.coloring_panel.start_btn.clicked.connect(self._start_coloring)
        self.stitch_panel.start_btn.clicked.connect(self._start_stitch)
        self.video_panel.start_btn.clicked.connect(self._start_video)

        self._process = None
        self._pending_logs = []

    def _sync_viewer_tab(self, index):
        if index == 0:
            self.preview.setCurrentIndex(0)
        elif index in (1, 2):
            self.preview.setCurrentIndex(1)

    def _link_series_dropdowns(self, coloring_panel, render_viewer):
        cp_combo = coloring_panel.series_combo
        rv_combo = render_viewer.series_combo
        cp_root  = coloring_panel.results_folder
        rv_root  = render_viewer.results_folder

        def _mirror_combo(src, dst):
            text = src.currentText()
            if not text or dst.currentText() == text:
                return
            idx = dst.findText(text)
            if idx < 0:
                try:
                    dst.parent()
                except RuntimeError:
                    return
                idx = dst.findText(text)
            if idx >= 0:
                dst.blockSignals(True)
                dst.setCurrentIndex(idx)
                dst.blockSignals(False)
                dst.currentTextChanged.emit(text)

        def _mirror_root(src_edit, dst_edit, dst_refresh):
            text = src_edit.text()
            if dst_edit.text() == text:
                return
            dst_edit.blockSignals(True)
            dst_edit.setText(text)
            dst_edit.blockSignals(False)
            dst_refresh()

        cp_combo.currentTextChanged.connect(
            lambda _=None: _mirror_combo(cp_combo, rv_combo))
        rv_combo.currentTextChanged.connect(
            lambda _=None: _mirror_combo(rv_combo, cp_combo))
        cp_root.textChanged.connect(
            lambda _=None: _mirror_root(
                cp_root, rv_root, render_viewer._refresh_series))
        rv_root.textChanged.connect(
            lambda _=None: _mirror_root(
                rv_root, cp_root, coloring_panel._refresh_series))

    # ── Process plumbing ────────────────────────────────────────────────
    def _run_script(self, script, args):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self.console.append_text("\n\u26a0  A process is already running.\n")
            return
        self._last_script = script
        self._snapshot_results()
        self._process = QProcess(self)
        self._process.setWorkingDirectory(PROJECT_DIR)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        script_path = os.path.join(SCRIPT_DIR, script)
        cmd_display = f"python {script} {' '.join(args)}"
        self.console.append_text(f"\n$ {cmd_display}\n")
        self.statusBar().showMessage(f"Running {script}\u2026")
        if getattr(sys, 'frozen', False):
            self._process.start(sys.executable, ["--run-script", script_path] + args)
        else:
            self._process.start(PYTHON, [script_path] + args)

    def _snapshot_results(self):
        snap = {}
        root = RESULTS_DIR
        if os.path.isdir(root):
            for s in os.listdir(root):
                sfull = os.path.join(root, s)
                if not os.path.isdir(sfull) or s.startswith("_"):
                    continue
                try:
                    subs = set(d for d in os.listdir(sfull)
                               if os.path.isdir(os.path.join(sfull, d))
                               and not d.startswith("_"))
                except OSError:
                    subs = set()
                snap[s] = subs
        self._results_snapshot = snap

    def _auto_select_after_finish(self):
        snap = getattr(self, "_results_snapshot", None) or {}
        script = getattr(self, "_last_script", "")
        root = RESULTS_DIR
        if not os.path.isdir(root):
            return

        current = {}
        for s in os.listdir(root):
            sfull = os.path.join(root, s)
            if not os.path.isdir(sfull) or s.startswith("_"):
                continue
            try:
                subs = set(d for d in os.listdir(sfull)
                           if os.path.isdir(os.path.join(sfull, d))
                           and not d.startswith("_"))
            except OSError:
                subs = set()
            current[s] = subs

        new_series = [s for s in current if s not in snap]
        target_series = None
        target_sub = None
        if script == "render.py":
            if new_series:
                target_series = sorted(new_series)[-1]
                if "greyscale" in current.get(target_series, set()):
                    target_sub = "greyscale"
                else:
                    subs = sorted(current.get(target_series, set()))
                    target_sub = subs[0] if subs else None
        elif script == "colorize.py":
            for s, subs in current.items():
                added = subs - snap.get(s, set())
                if added:
                    target_series = s
                    target_sub = sorted(added)[0]
                    break
        elif script == "stitch.py":
            if new_series:
                target_series = sorted(new_series)[-1]
                subs = current.get(target_series, set())
                target_sub = "frames" if "frames" in subs else (
                    sorted(subs)[0] if subs else None)

        if not target_series:
            if self.preview.render_viewer.series_combo.count() > 0:
                self.preview.render_viewer.series_combo.setCurrentIndex(
                    self.preview.render_viewer.series_combo.count() - 1)
            return

        rv = self.preview.render_viewer
        idx = rv.series_combo.findText(target_series)
        if idx >= 0:
            rv.series_combo.setCurrentIndex(idx)
        if target_sub:
            folder = os.path.join(root, target_series, target_sub)
            if os.path.isdir(folder):
                rv.load_folder(folder)

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.console.append_text(data)

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self.console.append_text(data)

    def _on_finished(self, exit_code, status):
        self.console.append_text(f"\n\u2713 Process finished (exit code {exit_code})\n")
        self.statusBar().showMessage("Ready")
        self.preview.set_viewer._load_frames()
        self.preview.render_viewer._refresh_series()
        self.coloring_panel._refresh_series()
        self._auto_select_after_finish()

    # ── Start handlers ──────────────────────────────────────────────────
    def _start_render(self):
        sp = self.set_panel
        rp = self.image_panel
        c = complex(sp.c_real.value(), sp.c_imag.value())
        center = complex(sp.center_real.value(), sp.center_imag.value())
        power = sp.power.value()
        kernel = sp.kernel.currentText()
        current_zf = rp.zoom_factor.value()
        ss = rp.scale_start.value()

        extras_json = ''
        try:
            extras_json = sp.extra_constants_json()
        except AttributeError:
            pass
        args = rp.build_args()
        if extras_json:
            self.console.append_text(
                "\nAdditional constants present -- skipping recursion check.\n")
            self._run_script("render.py", args)
            return
        self.console.append_text("\n\U0001f50d Pre-render recursion check\u2026\n")
        rec = detect_recursion(c, center, kernel, power, current_zf,
                               log=lambda msg: self.console.append_text(msg),
                               scale_start=ss)
        args = rp.build_args()

        if rec is not None:
            optimal_zf = rec["optimal_zf"]
            loop_k = rec["k"]
            loop_start = rec["loop_start_frame"]
            if abs(current_zf - optimal_zf) < 1e-8:
                self.console.append_text(
                    f"\u2713 Zoom factor matches loop period {loop_k}. "
                    f"Loop starts at frame {loop_start}. "
                    f"Duplicate frames will be copied.\n")
                args += ["--loop-period", str(loop_k),
                         "--loop-start", str(loop_start)]
            else:
                msg = QMessageBox(self)
                msg.setWindowTitle("Loop Detected")
                msg.setIcon(QMessageBox.Icon.Question)
                msg.setText(
                    f"A possible loop was detected (period {loop_k}).\n\n"
                    f"Optimal zoom factor: {optimal_zf:.9f}\n"
                    f"Current zoom factor: {current_zf:.9f}\n\n"
                    f"The current zoom factor does not produce a perfect loop.\n"
                    f"Render with these parameters anyway?"
                )
                btn_render = msg.addButton("Render anyway", QMessageBox.ButtonRole.AcceptRole)
                btn_adjust = msg.addButton(f"Adjust to {optimal_zf:.9f}", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_adjust:
                    rp.zoom_factor.setValue(optimal_zf)
                    args = rp.build_args()
                    rec2 = detect_recursion(c, center, kernel, power, optimal_zf,
                                            scale_start=ss)
                    if rec2:
                        args += ["--loop-period", str(rec2["k"]),
                                 "--loop-start", str(rec2["loop_start_frame"])]
                    self.console.append_text(
                        f"\u2713 Adjusted zoom factor to {optimal_zf:.9f}. "
                        f"Loop period = {rec2['k'] if rec2 else '?'} frames.\n")
                elif clicked == btn_render:
                    self.console.append_text("Rendering without loop optimisation.\n")
                else:
                    self.console.append_text("Render cancelled.\n")
                    return
        else:
            self.console.append_text("No loop detected \u2013 rendering all frames.\n")

        self._run_script("render.py", args)

    def _start_video(self):
        args = self.video_panel.build_args()
        if args is None:
            self.console.append_text("\n\u26a0  Please set a frame folder first.\n")
            return
        self._run_script("video.py", args)

    def _start_stitch(self):
        args = self.stitch_panel.build_args()
        if args is None:
            self.console.append_text(
                "\n\u26a0  Need at least two series (each with a selected "
                "colour folder) to stitch.\n")
            return
        self._run_script("stitch.py", args)

    def _start_coloring(self):
        args = self.coloring_panel.build_args()
        if args is None:
            self.console.append_text(
                "\n\u26a0  Pick a series with _raw/ data and at least "
                "one colorizer first.\n")
            return
        folder = self.coloring_panel.current_series_folder()
        if folder and not os.path.isdir(os.path.join(folder, "_raw")):
            self.console.append_text(
                f"\n\u26a0  {folder} has no _raw/ sub-folder. Only series "
                "rendered after the raw-data update can be re-colorized.\n")
            return
        self._run_script("colorize.py", args)


def main():
    if getattr(sys, 'frozen', False):
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except Exception:
            pass
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

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

