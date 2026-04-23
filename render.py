"""
Render a zoom series: Generator × Colorizer(s) → PNG frames.

Every render also saves the raw iteration-count arrays as compressed .npz
files in `<series>/_raw/` plus a `meta.json`.  These persist after the run
so the series can be re-colorized later via `colorize.py` without
recomputing the (expensive) fractal iteration.

Usage examples:
    python render.py                           # uses defaults below
    python render.py --generator flower --colorizers hsv_cyclic twilight
"""
import os
import json
import argparse
import numpy as np
from time import perf_counter as clock
from concurrent.futures import ProcessPoolExecutor, as_completed

from generators import GENERATORS
from colorizers import COLORIZERS, _calc_bounds


# ── defaults (edit here for quick runs) ──────────────────────────────────────
DEFAULTS = dict(
    generator="flower",
    # By default render only the greyscale preview + raw data.  Extra
    # colorizers can be produced any time via colorize.py.
    colorizers=[],
    c_real=-0.54, c_imag=0.54,
    center_real=0.0, center_imag=0.0,
    width=640, height=640,
    max_iter=1024, power=2,
    scale_start=2.0, zoom_factor=0.92,
    num_frames=30,
    petals=4, mirror_segments=False,
    interest_angle=110, align_north="edge",
    workers=12,
    kernel="poly", bailout=2.0,
    mask_svg=None,
    scale_smoothing=0.7,
    recenter=False,
)


# Always produced – used as preview when no colorizer was selected.
_ALWAYS_COLORIZERS = ("greyscale",)


# ── worker process ───────────────────────────────────────────────────────────
_worker_gen = None
_worker_colors = None
_worker_max_iter = None
_worker_folder = None
_worker_mask = None


def _worker_init(gen_cls, gen_kwargs, color_names, folder, mask_svg):
    """Called once per worker process – builds its own generator."""
    global _worker_gen, _worker_colors, _worker_max_iter, _worker_folder, _worker_mask
    _worker_gen = gen_cls(**gen_kwargs)
    _worker_colors = {n: COLORIZERS[n] for n in color_names}
    _worker_max_iter = gen_kwargs["max_iter"]
    _worker_folder = folder
    if mask_svg:
        from mask import load_svg_mask, radial_fill
        display_mask, _ = load_svg_mask(mask_svg,
                                        gen_kwargs["width"],
                                        gen_kwargs["height"])
        # Radial fill: ensure no gaps between center and any visible pixel
        _worker_mask = radial_fill(display_mask)
    else:
        _worker_mask = None


def _raw_path(folder, frame):
    return os.path.join(folder, "_raw", f"frame_{frame:04d}.npz")


def _worker_render_counts(frame, scale):
    """Pass 1: render counts, save compressed, return (frame, bounds, elapsed)."""
    t0 = clock()
    counts = _worker_gen.render(scale, mask=_worker_mask)
    np.savez_compressed(_raw_path(_worker_folder, frame), counts=counts)
    bounds = _calc_bounds(counts, _worker_max_iter)
    return frame, bounds, clock() - t0


def _worker_colorize(frame, bounds):
    """Pass 2: load counts, colorize with given bounds, keep the raw file."""
    t0 = clock()
    with np.load(_raw_path(_worker_folder, frame)) as npz:
        counts = npz["counts"]
    for name, fn in _worker_colors.items():
        img = fn(counts, _worker_max_iter, bounds=bounds)
        img.save(os.path.join(_worker_folder, name, f"frame_{frame:04d}.png"))
    return frame, clock() - t0


def _smooth_bounds(frame_bounds, smoothing):
    """Bidirectional EMA smoothing of (lo, hi) bounds across frames.
    smoothing in [0, 1): 0 = no smoothing, 0.9 = heavy smoothing."""
    if smoothing <= 0:
        return frame_bounds
    sorted_frames = sorted(frame_bounds.keys())
    # Forward pass
    fwd = {}
    prev_lo = prev_hi = None
    for f in sorted_frames:
        b = frame_bounds[f]
        if b is None:
            fwd[f] = (prev_lo, prev_hi) if prev_lo is not None else None
            continue
        lo, hi = b
        if prev_lo is None:
            prev_lo, prev_hi = lo, hi
        else:
            prev_lo = smoothing * prev_lo + (1 - smoothing) * lo
            prev_hi = smoothing * prev_hi + (1 - smoothing) * hi
        fwd[f] = (prev_lo, prev_hi)
    # Backward pass
    bwd = {}
    prev_lo = prev_hi = None
    for f in reversed(sorted_frames):
        b = frame_bounds[f]
        if b is None:
            bwd[f] = (prev_lo, prev_hi) if prev_lo is not None else None
            continue
        lo, hi = b
        if prev_lo is None:
            prev_lo, prev_hi = lo, hi
        else:
            prev_lo = smoothing * prev_lo + (1 - smoothing) * lo
            prev_hi = smoothing * prev_hi + (1 - smoothing) * hi
        bwd[f] = (prev_lo, prev_hi)
    # Average forward and backward
    smoothed = {}
    for f in sorted_frames:
        if fwd.get(f) is None or bwd.get(f) is None:
            smoothed[f] = fwd.get(f) or bwd.get(f)
        else:
            smoothed[f] = (
                (fwd[f][0] + bwd[f][0]) / 2,
                (fwd[f][1] + bwd[f][1]) / 2,
            )
    return smoothed


def _write_meta(folder, args, gen_kwargs, frame_bounds):
    """Persist the information needed to re-colorize this series later."""
    meta = {
        "generator": args.generator,
        "loop_period": int(getattr(args, "loop_period", 0) or 0),
        "loop_start":  int(getattr(args, "loop_start", 0) or 0),
        "kernel": args.kernel,
        "power": args.power,
        "bailout": args.bailout,
        "c": [args.c_real, args.c_imag],
        "center": [args.center_real, args.center_imag],
        "width": args.width,
        "height": args.height,
        "max_iter": args.max_iter,
        "scale_start": args.scale_start,
        "zoom_factor": args.zoom_factor,
        "num_frames": args.num_frames,
        "start_frame": args.start_frame,
        "frame_step": args.frame_step,
        "recenter": args.recenter,
        "mask_svg": args.mask_svg,
        "scale_smoothing_render": args.scale_smoothing,
        "flower": {
            "petals": args.petals,
            "mirror_segments": args.mirror_segments,
            "interest_angle": args.interest_angle,
            "align_north": args.align_north,
        } if args.generator == "flower" else None,
        "per_frame_bounds": {
            str(f): (None if b is None else [float(b[0]), float(b[1])])
            for f, b in frame_bounds.items()
        },
    }
    with open(os.path.join(folder, "_raw", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ── main ─────────────────────────────────────────────────────────────────────

def build_gen_kwargs(args):
    common = dict(
        c=complex(args.c_real, args.c_imag),
        center=complex(args.center_real, args.center_imag),
        width=args.width, height=args.height,
        max_iter=args.max_iter, power=args.power,
        kernel=args.kernel, bailout=args.bailout,
        recenter=args.recenter,
    )
    if args.generator == "flower":
        common.update(
            petals=args.petals,
            mirror_segments=args.mirror_segments,
            interest_angle=args.interest_angle,
            align_north=args.align_north,
        )
    return common


def render(args):
    gen_cls = GENERATORS[args.generator]
    gen_kwargs = build_gen_kwargs(args)
    smoothing = args.scale_smoothing

    # Build the effective colorizer list: user's selection + forced extras,
    # de-duplicated while preserving order (greyscale first).
    color_names = list(dict.fromkeys(list(_ALWAYS_COLORIZERS) + list(args.colorizers)))

    # Build once locally just for folder name
    gen_tmp = gen_cls(**gen_kwargs)
    custom_name = (getattr(args, "series_name", "") or "").strip()
    series_name = custom_name if custom_name else gen_tmp.folder_name(
        args.scale_start, args.zoom_factor, args.num_frames)
    folder = os.path.join("results", series_name)
    for name in color_names:
        os.makedirs(os.path.join(folder, name), exist_ok=True)
    os.makedirs(os.path.join(folder, "_raw"), exist_ok=True)

    frames = [(f, args.scale_start * (args.zoom_factor ** f))
              for f in range(args.start_frame,
                             args.start_frame + args.num_frames,
                             args.frame_step)]

    # ── Loop optimisation: if --loop-period is set, only render one period ──
    loop_period = getattr(args, 'loop_period', 0) or 0
    loop_start = getattr(args, 'loop_start', 0) or 0
    frames_to_render = frames
    frames_to_copy = []  # (target_frame, source_frame)
    if loop_period > 0 and len(frames) > loop_start + loop_period:
        # Render all frames up to loop_start + loop_period (the "pre-loop" + one full period)
        n_must_render = loop_start + loop_period
        frames_to_render = frames[:n_must_render]
        # Source frames for copying come from the loop region only (loop_start .. loop_start+period-1)
        source_frames = [f for f, _ in frames[loop_start:loop_start + loop_period]]
        for idx in range(n_must_render, len(frames)):
            tgt_frame, _ = frames[idx]
            # Map into the loop region
            src_frame = source_frames[(idx - loop_start) % loop_period]
            frames_to_copy.append((tgt_frame, src_frame))
        print(f"Loop optimisation: rendering {len(frames_to_render)} frames "
              f"(pre-loop: {loop_start}, loop period: {loop_period}), "
              f"copying {len(frames_to_copy)} frames")

    total_start = clock()
    n_render = len(frames_to_render)
    n_total = len(frames)
    n_workers = min(args.workers, max(n_render, 1))

    # Two-pass rendering is now always used – we need bounds smoothing for
    # the greyscale preview and for any additional colorizers.
    print(f"Pass 1: rendering counts + raw data…")
    frame_bounds = {}

    if n_workers <= 1:
        _worker_init(gen_cls, gen_kwargs, color_names, folder, args.mask_svg)
        for i, (frame, scale) in enumerate(frames_to_render, 1):
            f, bounds, elapsed = _worker_render_counts(frame, scale)
            frame_bounds[f] = bounds
            print(f"[{i}/{n_render}] frame {f:3d}  scale={scale:.6f}  ({elapsed:.1f}s)")
    else:
        done = 0
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(gen_cls, gen_kwargs, color_names, folder, args.mask_svg),
        ) as pool:
            futures = {pool.submit(_worker_render_counts, f, s): f for f, s in frames_to_render}
            for fut in as_completed(futures):
                f, bounds, elapsed = fut.result()
                frame_bounds[f] = bounds
                done += 1
                scale = args.scale_start * (args.zoom_factor ** f)
                print(f"[{done}/{n_render}] frame {f:3d}  "
                      f"scale={scale:.6f}  ({elapsed:.1f}s)")

    # Persist meta.json (render parameters + per-frame bounds)
    _write_meta(folder, args, gen_kwargs, frame_bounds)

    # ── Smooth bounds ────────────────────────────────────────────────
    print(f"Smoothing bounds (factor={smoothing})…")
    smoothed = _smooth_bounds(frame_bounds, smoothing)

    # ── Pass 2: colorize with smoothed bounds ────────────────────────
    print(f"Pass 2: colorizing with smoothed bounds…  colorizers={list(color_names)}")
    colorize_tasks = [(f, smoothed.get(f)) for f, _ in frames_to_render if smoothed.get(f) is not None]

    # Limit workers for colorize pass – each worker holds a full image
    # in memory plus several float32 temp arrays (~20 bytes/pixel).
    pixels = gen_kwargs.get("width", 640) * gen_kwargs.get("height", 640)
    mem_per_worker_mb = pixels * 20 / (1024 * 1024)  # rough estimate
    max_color_workers = max(1, int(2048 / max(mem_per_worker_mb, 1)))
    n_color_workers = min(n_workers, max_color_workers, len(colorize_tasks))
    if n_color_workers < n_workers:
        print(f"  (limiting to {n_color_workers} workers for colorize to avoid OOM)")

    if n_color_workers <= 1:
        _worker_init(gen_cls, gen_kwargs, color_names, folder, args.mask_svg)
        for i, (frame, bounds) in enumerate(colorize_tasks, 1):
            f, elapsed = _worker_colorize(frame, bounds)
            print(f"[colorize {i}/{len(colorize_tasks)}] frame {f:3d}  ({elapsed:.1f}s)")
    else:
        done = 0
        with ProcessPoolExecutor(
            max_workers=n_color_workers,
            initializer=_worker_init,
            initargs=(gen_cls, gen_kwargs, color_names, folder, args.mask_svg),
        ) as pool:
            futures = {pool.submit(_worker_colorize, f, b): f for f, b in colorize_tasks}
            for fut in as_completed(futures):
                f, elapsed = fut.result()
                done += 1
                print(f"[colorize {done}/{len(colorize_tasks)}] frame {f:3d}  ({elapsed:.1f}s)")

    # Raw .npz files are kept on disk (in _raw/) so the series can be
    # re-colorized later without another render pass.

    # ── Copy looped frames ────────────────────────────────────────────────
    if frames_to_copy:
        import shutil
        print(f"Copying {len(frames_to_copy)} looped frames…")
        for tgt, src in frames_to_copy:
            for name in color_names:
                src_path = os.path.join(folder, name, f"frame_{src:04d}.png")
                tgt_path = os.path.join(folder, name, f"frame_{tgt:04d}.png")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, tgt_path)
            # Also copy the raw data so colorize.py can reuse it
            src_raw = _raw_path(folder, src)
            tgt_raw = _raw_path(folder, tgt)
            if os.path.exists(src_raw):
                shutil.copy2(src_raw, tgt_raw)
        print(f"  Copied {len(frames_to_copy)} frames from loop period {loop_period}.")

    print(f"\nDone. {n_total} frames ({n_render} rendered, "
          f"{len(frames_to_copy)} copied) in {clock()-total_start:.1f}s "
          f"({n_workers} workers) -> {folder}/")
    print(f"Raw data in {folder}/_raw/ -- use colorize.py to add more colorizers.")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fractal zoom renderer")
    d = DEFAULTS
    p.add_argument("--generator", default=d["generator"],
                   choices=GENERATORS.keys())
    p.add_argument("--colorizers", nargs="*", default=d["colorizers"],
                   choices=COLORIZERS.keys(),
                   help="Additional colorizers to run during rendering. "
                        "greyscale is always produced.")
    p.add_argument("--c-real", type=float, default=d["c_real"])
    p.add_argument("--c-imag", type=float, default=d["c_imag"])
    p.add_argument("--center-real", type=float, default=d["center_real"])
    p.add_argument("--center-imag", type=float, default=d["center_imag"])
    p.add_argument("--width", type=int, default=d["width"])
    p.add_argument("--height", type=int, default=d["height"])
    p.add_argument("--max-iter", type=int, default=d["max_iter"])
    p.add_argument("--power", type=int, default=d["power"])
    p.add_argument("--kernel", default=d["kernel"],
                   choices=["poly", "sin", "cos"])
    p.add_argument("--bailout", type=float, default=d["bailout"])
    p.add_argument("--scale-start", type=float, default=d["scale_start"])
    p.add_argument("--zoom-factor", type=float, default=d["zoom_factor"])
    p.add_argument("--num-frames", type=int, default=d["num_frames"])
    p.add_argument("--start-frame", type=int, default=0,
                   help="First frame index (to continue a previous run)")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Only render every N-th frame")
    p.add_argument("--petals", type=int, default=d["petals"])
    p.add_argument("--mirror-segments", action="store_true",
                   default=d["mirror_segments"])
    p.add_argument("--interest-angle", type=float,
                   default=d["interest_angle"])
    p.add_argument("--align-north", default=d["align_north"],
                   choices=["center", "edge"])
    p.add_argument("--workers", type=int, default=d["workers"],
                   help="Number of parallel worker processes")
    p.add_argument("--mask-svg", type=str, default=d["mask_svg"],
                   help="Optional SVG file for image mask")
    p.add_argument("--scale-smoothing", type=float, default=d["scale_smoothing"],
                   help="EMA smoothing for color scale bounds (0=off, 0.7=default, 0.95=heavy)")
    p.add_argument("--recenter", action="store_true", default=d["recenter"],
                   help="Keep zoom center fixed in image center regardless of scale")
    p.add_argument("--loop-period", type=int, default=0,
                   help="If >0, only render this many frames then copy for the rest (loop optimisation)")
    p.add_argument("--loop-start", type=int, default=0,
                   help="Frame index where the loop begins (frames before this are always rendered)")
    p.add_argument("--series-name", type=str, default="",
                   help="Override the auto-generated series folder name under results/.")
    return p.parse_args()


if __name__ == "__main__":
    render(parse_args())
