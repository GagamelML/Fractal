"""Fractal Studio entry point.
The actual GUI implementation lives in the ``gui_pkg`` package
(common, analysis, panels, viewers, main_window).  This file only:
  * dispatches the ``--run-script`` PyInstaller-frozen sub-process mode
  * launches the Qt application via ``gui_pkg.main_window.main``.
Usage:  python gui.py
"""
import os
import sys
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        script_dir = os.path.dirname(os.path.abspath(script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        with open(script) as f:
            exec(compile(f.read(), script, "exec"), {"__name__": "__main__"})
    else:
        from gui_pkg.main_window import main
        main()
