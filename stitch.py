"""
Stitch two or more coloured series together with zoom-in overlay transitions.

Concept
-------
When we transition from series A to series B we take a single frame of B (the
"entry" frame), scale it down to a few pixels, paste it in the centre of the
last frames of A, then grow the overlay step-by-step at A's own zoom speed so
that after `transition_length` frames the B-image fills the screen.  From then
on we continue with B's frames starting at `entry_frame`.

Program JSON format
-------------------
{
    "output_name": "A_to_B",
    "segments": [
        # First segment: plain range, no transition.
        {"folder": "results/seriesA/twilight",
         "start_frame": 0, "end_frame": 100},
        # Each subsequent segment adds a transition of `transition_length`
        # frames at the *end of the previous segment*.  During those frames,
        # frame `start_frame` of this segment is overlaid on top of the
        # previous series and grows from a few pixels to full screen.
        {"folder": "results/seriesB/log_fire",
         "start_frame": 20, "end_frame": 200,
         "transition_length": 30}
    ]
}

Output
------
    results/<output_name>/frames/frame_XXXX.png    (1-based-alike, 4 digits)

This folder is directly feedable to video.py.
"""
import os
import json
import glob
import argparse
import numpy as np
from PIL import Image


# ── helpers ────────────────────────────────────────────────────────────────

def _frame_files(folder):
    """All PNG frames in a folder, sorted by index encoded in the filename."""
    paths = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    return paths


def _frame_path(folder, idx):
    """Exact path to frame `idx` in `folder`, or None if not present."""
    p = os.path.join(folder, f"frame_{idx:04d}.png")
    return p if os.path.isfile(p) else None


def _read_zoom_factor(folder, default=0.92):
    """Try to read zoom_factor from the series' _raw/meta.json.

    `folder` is the coloured sub-folder (e.g. '.../twilight'); meta.json
    lives one level up in `_raw/`.
    """
    parent = os.path.dirname(os.path.abspath(folder))
    meta_path = os.path.join(parent, "_raw", "meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        zf = float(meta.get("zoom_factor", default))
        if 0 < zf < 1:
            return zf
    except Exception:
        pass
    return default


def _select_range(folder, start_frame, end_frame):
    """Return the list of frame file paths within [start, end] (inclusive).

    Missing frames are skipped silently.  If start_frame > end_frame or no
    frames are present the returned list is empty.
    """
    out = []
    for idx in range(int(start_frame), int(end_frame) + 1):
        p = _frame_path(folder, idx)
        if p is not None:
            out.append((idx, p))
    return out


def _load_overlay_rgba(frame_path, series_folder):
    """Load the overlay frame as RGBA.  Pixels that are masked out in the
    source series become fully transparent so the overlay takes on the
    organic mask shape instead of a black square.

    Priority:
      1. ``<series>/../_raw/mask.npy`` (bool array written by render.py)
      2. Fallback: treat pure-black pixels (0,0,0) as transparent.
    """
    img = Image.open(frame_path).convert("RGBA")
    arr = np.array(img)                                      # (H,W,4)

    mask = None
    parent = os.path.dirname(os.path.abspath(series_folder))
    mask_path = os.path.join(parent, "_raw", "mask.npy")
    if os.path.isfile(mask_path):
        try:
            m = np.load(mask_path).astype(bool)
            if m.shape == arr.shape[:2]:
                mask = m
        except Exception:
            mask = None

    if mask is not None:
        arr[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    else:
        # Fallback: black pixels → transparent
        black = (arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
        arr[black, 3] = 0

    return Image.fromarray(arr, mode="RGBA")


def _compose(base_arr, overlay_rgba, scale):
    """Alpha-composite `overlay_rgba` centred on `base_arr`, resized to `scale`.

    `scale` is the linear size ratio (1.0 = full screen).
    Pixels in the overlay with alpha=0 reveal the base underneath, so a
    masked overlay keeps its organic shape rather than a black square.
    """
    h, w = base_arr.shape[:2]
    ow = max(2, int(round(w * scale)))
    oh = max(2, int(round(h * scale)))
    ow = min(ow, w)
    oh = min(oh, h)
    small = overlay_rgba.resize((ow, oh), Image.LANCZOS)
    ov = np.array(small)                                     # (oh,ow,4)

    # Prepare base RGB view (drop alpha if any).
    if base_arr.ndim == 2:
        base_rgb = np.stack([base_arr] * 3, axis=-1)
        was_gray = True
    elif base_arr.shape[2] == 4:
        base_rgb = base_arr[..., :3]
        was_gray = False
    else:
        base_rgb = base_arr
        was_gray = False

    out = base_rgb.copy()
    x0 = (w - ow) // 2
    y0 = (h - oh) // 2

    region = out[y0:y0 + oh, x0:x0 + ow].astype(np.float32)
    ov_rgb = ov[..., :3].astype(np.float32)
    alpha = (ov[..., 3:4].astype(np.float32)) / 255.0
    blended = ov_rgb * alpha + region * (1.0 - alpha)
    out[y0:y0 + oh, x0:x0 + ow] = blended.astype(np.uint8)

    if was_gray:
        # Convert back to grayscale to match the input image mode.
        out = np.array(Image.fromarray(out).convert("L"))
    return out


# ── main stitch routine ────────────────────────────────────────────────────

def stitch(program, output_dir):
    """Run the stitch according to the program dict, writing PNGs to
    output_dir/frame_XXXX.png."""
    segments = program["segments"]
    if len(segments) < 2:
        raise ValueError("Need at least two segments to stitch.")

    os.makedirs(output_dir, exist_ok=True)

    out_idx = 0

    # Walk segment pairs.  Segment i writes its own frames; the transition
    # into segment i+1 is overlaid on the *tail* of segment i's output.
    n_seg = len(segments)
    for i, seg in enumerate(segments):
        folder = seg["folder"]
        s_start = int(seg.get("start_frame", 0))
        s_end   = int(seg.get("end_frame",   10**9))
        own_frames = _select_range(folder, s_start, s_end)
        if not own_frames:
            print(f"[warn] segment {i}: no frames in {folder} "
                  f"[{s_start}..{s_end}]")
            continue

        # Is there a follow-up segment we need to fade into?
        overlay_img = None
        trans_len = 0
        zf_this = 1.0
        if i + 1 < n_seg:
            nxt = segments[i + 1]
            trans_len = int(nxt.get("transition_length", 0) or 0)
            if trans_len > 0:
                nxt_folder = nxt["folder"]
                nxt_entry  = int(nxt.get("start_frame", 0))
                nxt_path = _frame_path(nxt_folder, nxt_entry)
                if nxt_path is None:
                    # fall back to the first available frame in that folder
                    cand = _frame_files(nxt_folder)
                    if cand:
                        nxt_path = cand[0]
                if nxt_path is None:
                    print(f"[warn] could not find entry frame for "
                          f"segment {i+1} – skipping transition.")
                    trans_len = 0
                else:
                    overlay_img = _load_overlay_rgba(nxt_path, nxt_folder)
                    zf_this = _read_zoom_factor(folder)
                    print(f"[transition] {i}->{i+1}: "
                          f"{trans_len} frames  zf={zf_this:.6f}  "
                          f"overlay={nxt_path}")

        # Cap transition length to what's available in this segment.
        trans_len = min(trans_len, len(own_frames))
        trans_start = len(own_frames) - trans_len    # index inside own_frames

        for k, (fr_idx, path) in enumerate(own_frames):
            arr = np.array(Image.open(path))
            if overlay_img is not None and k >= trans_start:
                t = k - trans_start            # 0 .. trans_len-1
                # Linear frame-count so overlay is 1 full screen at the last
                # transition frame:  scale = zf^(trans_len - 1 - t)
                expo = (trans_len - 1) - t
                scale = zf_this ** expo if expo > 0 else 1.0
                arr = _compose(arr, overlay_img, scale)

            out_path = os.path.join(output_dir, f"frame_{out_idx:04d}.png")
            Image.fromarray(arr).save(out_path)
            out_idx += 1
            if out_idx % 25 == 0:
                print(f"  wrote {out_idx} frames…", flush=True)

    print(f"\nDone. {out_idx} frames written to {output_dir}/")
    return out_idx


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stitch coloured series with zoom-overlay transitions.")
    p.add_argument("program", help="Path to stitch program JSON.")
    p.add_argument("--output-name", default=None,
                   help="Override program's output_name.")
    return p.parse_args()


def main():
    a = parse_args()
    with open(a.program) as f:
        prog = json.load(f)

    out_name = a.output_name or prog.get("output_name") or "stitched"
    output_dir = os.path.join("results", out_name, "frames")
    stitch(prog, output_dir)


if __name__ == "__main__":
    main()




