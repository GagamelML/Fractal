"""Constants and small filesystem helpers shared across the GUI package."""
import os
import sys
import json

PYTHON = sys.executable

# When frozen by PyInstaller, scripts are bundled as data files inside _internal
if getattr(sys, 'frozen', False):
    PROJECT_DIR = os.path.dirname(sys.executable)
    SCRIPT_DIR = os.path.join(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    # Two levels up: <project>/gui_pkg/common.py -> <project>/
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SCRIPT_DIR = PROJECT_DIR

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PREVIEW_DIR = os.path.join(RESULTS_DIR, "_preview")


def save_preview_params(folder, params):
    """Save *params* dict as ``params.json`` inside *folder*."""
    path = os.path.join(folder, "params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=str)
    return path


def load_preview_params(folder):
    """Load ``params.json`` from *folder* (or return None if missing)."""
    path = os.path.join(folder, "params.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def next_preview_folder():
    """Return ``(path, num)`` for the next numbered preview folder."""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    existing = [d for d in os.listdir(PREVIEW_DIR)
                if os.path.isdir(os.path.join(PREVIEW_DIR, d)) and d.isdigit()]
    num = max((int(d) for d in existing), default=0) + 1
    path = os.path.join(PREVIEW_DIR, f"{num:03d}")
    os.makedirs(path, exist_ok=True)
    return path, num

