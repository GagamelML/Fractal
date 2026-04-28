"""
Apply colorizers to an already-rendered raw series.

Two modes:

1. Classic (one folder per colorizer):
       python colorize.py <series> --colorizers log_fire twilight

2. Program mode (multi-segment with smooth cross-fades):
       python colorize.py <series> --program program.json

Program JSON format
-------------------
{
    "output_name": "twilight_to_log_fire",     # optional, auto-derived otherwise
    "segments": [
        {"start_frame": 0,  "transition_length": 0,
         "colorizer": "twilight",
         "invert": false, "force_bg": null},
        {"start_frame": 50, "transition_length": 20,
         "colorizer": "log_fire",
         "invert": true,  "force_bg": "black"}
    ]
}

Segments are sorted by start_frame.  For each frame the active segment is the
highest-indexed segment with start_frame <= frame.  If frame also lies inside
[start_frame, start_frame + transition_length) of the active segment, a
linear RGB cross-fade from the previous segment's colorizer output is
applied with weight (frame - start_frame) / transition_length.

No overlap check is performed; if transitions overlap, the later one wins.
"""
import os
import json
import glob
import argparse
import numpy as np
from time import perf_counter as clock
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

from colorizers import COLORIZERS, _calc_bounds


# ── worker globals ─────────────────────────────────────────────────────────
_worker_folder = None
_worker_max_iter = None
_worker_colors = None        # classic mode: dict[name -> fn]
_worker_program = None       # program mode: list[segment]
_worker_output_name = None   # program mode: output subfolder name
_worker_mask = None          # bool ndarray (True = keep, False = force black)


def _worker_init(folder, max_iter, color_names,
                 program=None, output_name=None):
    global _worker_folder, _worker_max_iter, _worker_colors
    global _worker_program, _worker_output_name, _worker_mask
    _worker_folder = folder
    _worker_max_iter = max_iter
    _worker_colors = {n: COLORIZERS[n] for n in (color_names or [])}
    _worker_program = program
    _worker_output_name = output_name
    mask_path = os.path.join(folder, "_raw", "mask.npy")
    if os.path.isfile(mask_path):
        try:
            _worker_mask = np.load(mask_path).astype(bool)
        except Exception:
            _worker_mask = None
    else:
        _worker_mask = None


def _apply_mask_to_image(img):
    """If a render mask exists, force masked-out pixels to pitch black."""
    if _worker_mask is None:
        return img
    arr = np.array(img)
    if arr.ndim == 2:
        arr[~_worker_mask] = 0
    else:
        arr[~_worker_mask] = 0  # broadcasts over channel axis
    return Image.fromarray(arr, mode=img.mode)


def _raw_path(folder, frame):
    return os.path.join(folder, "_raw", f"frame_{frame:04d}.npz")


def _worker_bounds(frame):
    """Compute (lo, hi) percentile bounds for this frame."""
    t0 = clock()
    with np.load(_raw_path(_worker_folder, frame)) as npz:
        counts = npz["counts"]
    return frame, _calc_bounds(counts, _worker_max_iter), clock() - t0


def _worker_colorize_classic(frame, bounds):
    """Classic mode worker: one folder per colorizer, no blending."""
    t0 = clock()
    with np.load(_raw_path(_worker_folder, frame)) as npz:
        counts = npz["counts"]
    for name, fn in _worker_colors.items():
        img = fn(counts, _worker_max_iter, bounds=bounds)
        img = _apply_mask_to_image(img)
        img.save(os.path.join(_worker_folder, name, f"frame_{frame:04d}.png"))
    return frame, clock() - t0


def _call_segment(seg, counts, max_iter, bounds):
    """Invoke a segment's colorizer with its universal parameters."""
    fn = COLORIZERS[seg["colorizer"]]
    return fn(counts, max_iter,
              bounds=bounds,
              invert=bool(seg.get("invert", False)),
              force_bg=seg.get("force_bg") or None)


def _resolve_segments(segments, frame):
    """
    Given the segments (already sorted by start_frame) and a frame number,
    return (segment_a, segment_b_or_None, weight_b ∈ [0,1]).

    If weight_b == 0 / segment_b is None the frame is rendered by segment_a
    alone.  Otherwise the output is a linear RGB cross-fade from a to b.
    """
    # Last segment whose start_frame <= frame
    k = 0
    for i, s in enumerate(segments):
        if int(s.get("start_frame", 0)) <= frame:
            k = i
        else:
            break
    active = segments[k]
    if k == 0:
        return active, None, 0.0
    tl = int(active.get("transition_length", 0))
    start = int(active.get("start_frame", 0))
    if tl > 0 and frame < start + tl:
        t = max(0.0, min(1.0, (frame - start) / tl))
        return segments[k - 1], active, t
    return active, None, 0.0


def _blend(img_a, img_b, weight_b):
    """Linear RGB cross-fade; weight_b ∈ [0, 1] = contribution of img_b."""
    if weight_b <= 0.0:
        return img_a
    if weight_b >= 1.0:
        return img_b
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    out = a * (1.0 - weight_b) + b * weight_b
    return Image.fromarray(out.astype(np.uint8), mode='RGB')


def _worker_colorize_program(frame, bounds):
    """Program mode worker: resolve segment(s) and blend if in transition."""
    t0 = clock()
    with np.load(_raw_path(_worker_folder, frame)) as npz:
        counts = npz["counts"]
    a, b, w = _resolve_segments(_worker_program, frame)
    img_a = _call_segment(a, counts, _worker_max_iter, bounds)
    if b is None:
        img = img_a
    else:
        img_b = _call_segment(b, counts, _worker_max_iter, bounds)
        img = _blend(img_a, img_b, w)
    img = _apply_mask_to_image(img)
    img.save(os.path.join(
        _worker_folder, _worker_output_name, f"frame_{frame:04d}.png"))
    return frame, clock() - t0


# Reuse the smoother from render.py to keep behaviour identical.
from render import _smooth_bounds  # noqa: E402


def discover_frames(folder):
    """Return the sorted list of frame indices present in _raw/."""
    paths = sorted(glob.glob(os.path.join(folder, "_raw", "frame_*.npz")))
    frames = []
    for p in paths:
        name = os.path.basename(p)
        try:
            frames.append(int(name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return frames


def _auto_output_name(segments):
    """Derive a folder name for program mode if none was supplied."""
    if len(segments) == 1:
        s = segments[0]
        suffix = "_inv" if s.get("invert") else ""
        bg = s.get("force_bg")
        if bg:
            suffix += f"_{bg}"
        return s["colorizer"] + suffix
    first = segments[0]["colorizer"]
    last = segments[-1]["colorizer"]
    name = f"{first}_to_{last}"
    if len(segments) > 2:
        name += f"_{len(segments)}seg"
    return name[:60]


def colorize_series(folder, color_names=None, smoothing=0.7, workers=12,
                    reuse_meta_bounds=True, program=None, output_name=None):
    """
    Colorize a raw series.

    * Classic mode: pass ``color_names`` – produces one PNG folder per name.
    * Program mode: pass ``program`` (list[dict]) – produces one PNG folder
      named ``output_name`` (auto-derived if omitted).
    """
    folder = os.path.abspath(folder)
    meta_path = os.path.join(folder, "_raw", "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"No meta.json in {folder}/_raw/ – is this a series rendered "
            "by render.py?")
    with open(meta_path) as f:
        meta = json.load(f)
    max_iter = int(meta["max_iter"])

    frames = discover_frames(folder)
    if not frames:
        raise FileNotFoundError(f"No frame_*.npz in {folder}/_raw/")

    # ── Loop optimisation (mirrors render.py, but colour-only) ───────────
    # If the series was rendered with --loop-period, smoothing is only
    # performed over the pre-loop + one full period.  All later frames
    # copy the *smoothed bounds* of their source frame inside that
    # window.  This eliminates the EMA zig-zag at loop boundaries while
    # still letting us colorize every frame individually (required for
    # program mode where segments/transitions may overlap the loop).
    loop_period = int(meta.get("loop_period") or 0)
    loop_start  = int(meta.get("loop_start") or 0)
    bounds_src_of = {}       # frame -> source frame whose bounds to reuse
    all_frames = list(frames)
    smoothing_frames = all_frames
    if loop_period > 0 and len(all_frames) > loop_start + loop_period:
        n_must = loop_start + loop_period
        source_frames = all_frames[loop_start:loop_start + loop_period]
        for idx in range(n_must, len(all_frames)):
            tgt = all_frames[idx]
            src = source_frames[(idx - loop_start) % loop_period]
            bounds_src_of[tgt] = src
        smoothing_frames = all_frames[:n_must]
        print(f"Loop optimisation: smoothing over {len(smoothing_frames)} "
              f"frame(s) (pre-loop {loop_start} + period {loop_period}); "
              f"{len(bounds_src_of)} later frame(s) reuse period bounds.")

    is_program = program is not None
    if is_program:
        if not program:
            raise ValueError("program must contain at least one segment")
        # Sort and normalise: first segment is always frame 0 / no transition
        program = sorted(program,
                         key=lambda s: int(s.get("start_frame", 0)))
        program[0]["start_frame"] = 0
        program[0]["transition_length"] = 0

        # Keep an unmodified copy of the user-facing program (frame indices
        # in *rendered-image* units) for persistence; below we translate
        # the live `program` to absolute frame numbers for processing.
        import copy as _copy
        program_user = _copy.deepcopy(program)

        # Translate user-facing segment indices (which count *rendered*
        # images, ignoring any --start-frame / --frame-step gaps) into
        # the absolute frame numbers stored on disk.  E.g. if rendering
        # was started at frame 5 with step 1 and the user specified a
        # segment starting at rendered-image #10, that maps to absolute
        # frame 5 + 10 = frames_sorted[10].
        frames_sorted = sorted(frames)
        n_rendered = len(frames_sorted)

        def _rendered_to_abs(idx):
            i = max(0, min(int(idx), n_rendered - 1))
            return frames_sorted[i]

        for seg in program:
            user_start = int(seg.get("start_frame", 0))
            user_tl = int(seg.get("transition_length", 0))
            abs_start = _rendered_to_abs(user_start)
            if user_tl > 0:
                abs_end = _rendered_to_abs(user_start + user_tl)
                abs_tl = max(0, abs_end - abs_start)
            else:
                abs_tl = 0
            seg["start_frame"] = abs_start
            seg["transition_length"] = abs_tl

        if output_name is None:
            output_name = _auto_output_name(program)
        os.makedirs(os.path.join(folder, output_name), exist_ok=True)
        # Persist the program next to the PNGs for reproducibility
        try:
            with open(os.path.join(folder, output_name, "program.json"),
                      "w") as f:
                json.dump({"output_name": output_name,
                           "smoothing": smoothing,
                           "segments": program_user}, f, indent=2)
        except OSError:
            pass
    else:
        if not color_names:
            raise ValueError("either color_names or program must be given")
        for name in color_names:
            os.makedirs(os.path.join(folder, name), exist_ok=True)

    n = len(frames)
    n_workers = min(workers, max(n, 1))
    if is_program:
        print(f"Colorizing {n} frames with program "
              f"({len(program)} segments) -> {folder}/{output_name}/")
    else:
        print(f"Colorizing {n} frames with {list(color_names)} in {folder}")

    # ── Pass 1: bounds ───────────────────────────────────────────────────
    # Only compute bounds for the smoothing window (one loop period).
    frame_bounds = {}
    meta_bounds = meta.get("per_frame_bounds") or {}
    need_bounds = [f for f in smoothing_frames
                   if not (reuse_meta_bounds and str(f) in meta_bounds
                           and meta_bounds[str(f)] is not None)]

    for f in smoothing_frames:
        b = meta_bounds.get(str(f))
        if reuse_meta_bounds and b is not None:
            frame_bounds[f] = (float(b[0]), float(b[1]))

    init_args = (folder, max_iter, color_names, program, output_name)

    if need_bounds:
        print(f"Pass 1: computing bounds for {len(need_bounds)} frame(s) "
              f"not cached in meta.json...")
        if n_workers <= 1:
            _worker_init(*init_args)
            for i, fr in enumerate(need_bounds, 1):
                f, b, el = _worker_bounds(fr)
                frame_bounds[f] = b
                print(f"[bounds {i}/{len(need_bounds)}] "
                      f"frame {f:3d}  ({el:.1f}s)")
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=init_args,
            ) as pool:
                futs = {pool.submit(_worker_bounds, fr): fr for fr in need_bounds}
                done = 0
                for fut in as_completed(futs):
                    f, b, el = fut.result()
                    frame_bounds[f] = b
                    done += 1
                    print(f"[bounds {done}/{len(need_bounds)}] "
                          f"frame {f:3d}  ({el:.1f}s)")
    else:
        print("Pass 1: all bounds read from meta.json (cached)")

    # ── Smooth ───────────────────────────────────────────────────────────
    print(f"Smoothing bounds (factor={smoothing})...")
    smoothed = _smooth_bounds(frame_bounds, smoothing)

    # Replicate smoothed bounds for looped frames so every loop iteration
    # gets the *same* bounds as its source inside the smoothing window.
    if bounds_src_of:
        for tgt, src in bounds_src_of.items():
            if smoothed.get(src) is not None:
                smoothed[tgt] = smoothed[src]

    # ── Pass 2: colorize (every frame, no copying) ───────────────────────
    tasks = [(f, smoothed.get(f)) for f in all_frames
             if smoothed.get(f) is not None]
    worker_fn = (_worker_colorize_program if is_program
                 else _worker_colorize_classic)

    t0 = clock()
    if n_workers <= 1:
        _worker_init(*init_args)
        for i, (fr, b) in enumerate(tasks, 1):
            f, el = worker_fn(fr, b)
            print(f"[colorize {i}/{len(tasks)}] frame {f:3d}  ({el:.1f}s)")
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            futs = {pool.submit(worker_fn, fr, b): fr for fr, b in tasks}
            done = 0
            for fut in as_completed(futs):
                f, el = fut.result()
                done += 1
                print(f"[colorize {done}/{len(tasks)}] "
                      f"frame {f:3d}  ({el:.1f}s)")

    dest = (f"{folder}/{output_name}/" if is_program
            else f"{folder}/{{{','.join(color_names)}}}/")
    print(f"\nDone. {len(tasks)} frames colorized in {clock()-t0:.1f}s "
          f"-> {dest}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply colorizers to a rendered raw series")
    p.add_argument("folder",
                   help="Results series folder (contains _raw/)")
    p.add_argument("--colorizers", nargs="+",
                   default=None,
                   choices=COLORIZERS.keys(),
                   help="Classic mode: one output folder per colorizer.")
    p.add_argument("--program", default=None,
                   help="Program mode: path to JSON describing multi-segment "
                        "coloring (see module docstring).")
    p.add_argument("--output-name", default=None,
                   help="Output subfolder name in program mode "
                        "(auto-derived otherwise).")
    p.add_argument("--scale-smoothing", type=float, default=0.7,
                   help="EMA smoothing for bounds (0 = off)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--no-meta-bounds", action="store_true",
                   help="Ignore cached bounds in meta.json and recompute")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    program = None
    output_name = a.output_name
    color_names = a.colorizers
    if a.program:
        with open(a.program) as f:
            prog = json.load(f)
        if isinstance(prog, list):
            program = prog
        else:
            program = prog.get("segments") or []
            output_name = output_name or prog.get("output_name")
    if program is None and not color_names:
        color_names = list(COLORIZERS.keys())
    colorize_series(a.folder,
                    color_names=color_names,
                    smoothing=a.scale_smoothing,
                    workers=a.workers,
                    reuse_meta_bounds=not a.no_meta_bounds,
                    program=program,
                    output_name=output_name)
