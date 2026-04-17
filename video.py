"""
Assemble rendered frames into video files.

Usage examples:
    python video.py results/julia_sin_.../log_fire
    python video.py results/julia_sin_.../log_fire --fps 30 --codec libx264
    python video.py results/julia_sin_.../log_fire --end-frame 270 --loop bounce
    python video.py results/julia_sin_.../log_fire --interp 10 --zoom-factor 0.92
"""
import os
import glob
import argparse
from time import perf_counter as clock

import imageio.v3 as iio
import numpy as np
from PIL import Image


# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    fps=30,
    codec="libx264",
    quality=8,           # CRF-like: 0=lossless, 10=default, lower=better
    pixel_format="yuv420p",
    start_frame=None,    # auto-detect from files
    end_frame=None,      # auto-detect from files
    loop="none",         # "none", "bounce" (forward+reverse), "repeat"
    interp=1,            # interpolation: N frames per rendered frame (1=off)
    zoom_factor=0.92,    # must match the zoom_factor used during rendering
    mask_svg=None,       # apply tight mask to every output frame
    output=None,         # auto-generate from folder name
)


def find_frames(folder, start=None, end=None):
    """Find and sort frame PNGs in a folder.  Returns list of paths."""
    pattern = os.path.join(folder, "frame_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No frame_*.png found in {folder}")

    # Extract frame numbers
    def frame_num(path):
        return int(os.path.basename(path).split("_")[1].split(".")[0])

    if start is not None:
        files = [f for f in files if frame_num(f) >= start]
    if end is not None:
        files = [f for f in files if frame_num(f) <= end]
    return files


def build_frame_list(files, loop_mode):
    """Apply loop mode to frame list."""
    if loop_mode == "bounce":
        return files + files[-2:0:-1]
    elif loop_mode == "repeat":
        return files + files
    return files


def crop_zoom(img_array, t, zoom_factor):
    """
    Digitally zoom into the centre of img_array by an intermediate amount.

    t=0 → original image, t=1 → zoomed in by zoom_factor (matching next frame).
    We crop a centred rectangle whose size is lerped between 1.0 and zoom_factor
    of the original, then upscale back to original dimensions.
    """
    if t <= 0:
        return img_array

    h, w = img_array.shape[:2]
    # At t=1 we want to show the central (zoom_factor) portion of the image
    # which corresponds to the next rendered frame's field of view.
    scale = 1.0 - t * (1.0 - zoom_factor)   # goes from 1.0 → zoom_factor

    crop_w = int(round(w * scale))
    crop_h = int(round(h * scale))
    # Ensure at least 2×2
    crop_w = max(crop_w, 2)
    crop_h = max(crop_h, 2)

    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    cropped = img_array[y0:y0+crop_h, x0:x0+crop_w]
    # Upscale back to original size using PIL (Lanczos for quality)
    img = Image.fromarray(cropped)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def encode_video(frame_paths, output_path, fps, codec, quality, pixel_format,
                 interp, zoom_factor, display_mask=None):
    """Encode frames to video with optional inter-frame zoom interpolation."""
    first = np.array(Image.open(frame_paths[0]))
    h, w = first.shape[:2]

    total_out = (len(frame_paths) - 1) * interp + 1 if interp > 1 else len(frame_paths)
    print(f"Encoding {len(frame_paths)} source frames → {total_out} output frames "
          f"({w}×{h}) → {output_path}")
    print(f"  codec={codec}  fps={fps}  quality={quality}  "
          f"interp={interp}  pix_fmt={pixel_format}"
          f"  mask={'yes' if display_mask is not None else 'no'}")

    def _apply_mask(frame):
        """Apply display mask and ensure even dimensions. Returns a new array."""
        frame = frame.copy()
        if display_mask is not None:
            frame[~display_mask] = 0
        if pixel_format == "yuv420p":
            if frame.shape[0] % 2: frame = frame[:frame.shape[0]-1, :]
            if frame.shape[1] % 2: frame = frame[:, :frame.shape[1]-1]
        return frame

    t0 = clock()
    frame_count = 0
    with iio.imopen(output_path, "w", plugin="pyav") as writer:
        writer.init_video_stream(codec, fps=fps)

        for i, path in enumerate(frame_paths):
            img = np.array(Image.open(path))

            if interp <= 1 or i == len(frame_paths) - 1:
                writer.write_frame(_apply_mask(img))
                frame_count += 1
            else:
                for sub in range(interp):
                    t = sub / interp
                    zoomed = crop_zoom(img, t, zoom_factor)
                    writer.write_frame(_apply_mask(zoomed))
                    frame_count += 1

            if (i + 1) % 20 == 0 or i == len(frame_paths) - 1:
                print(f"  source [{i+1}/{len(frame_paths)}]  "
                      f"output [{frame_count}/{total_out}]")

    elapsed = clock() - t0
    duration = total_out / fps
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone. {elapsed:.1f}s → {output_path} "
          f"({size_mb:.1f} MB, {duration:.1f}s @ {fps}fps)")


def main(args):
    files = find_frames(args.folder, args.start_frame, args.end_frame)
    print(f"Found {len(files)} frames in {args.folder}")

    frame_list = build_frame_list(files, args.loop)
    if args.loop != "none":
        print(f"Loop '{args.loop}': {len(files)} → {len(frame_list)} frames")

    # Load display mask if provided
    display_mask = None
    if args.mask_svg:
        first = Image.open(frame_list[0])
        w, h = first.size
        from mask import load_svg_mask
        display_mask, ratio = load_svg_mask(args.mask_svg, w, h)
        print(f"Display mask: {ratio*100:.1f}% coverage")

    if args.output:
        output = args.output
    else:
        parent = os.path.dirname(args.folder.rstrip("/\\"))
        base = (os.path.basename(parent) + "_"
                + os.path.basename(args.folder.rstrip("/\\")))
        tag = f"_x{args.interp}" if args.interp > 1 else ""
        output = os.path.join(parent, f"{base}_{args.fps}fps{tag}.mp4")

    encode_video(frame_list, output, args.fps, args.codec, args.quality,
                 args.pixel_format, args.interp, args.zoom_factor, display_mask)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Render frames to video")
    d = DEFAULTS
    p.add_argument("folder", help="Folder containing frame_*.png files")
    p.add_argument("--fps", type=int, default=d["fps"])
    p.add_argument("--codec", default=d["codec"])
    p.add_argument("--quality", type=int, default=d["quality"],
                   help="CRF quality (0=lossless, ~18=good, ~28=small)")
    p.add_argument("--pixel-format", default=d["pixel_format"])
    p.add_argument("--start-frame", type=int, default=d["start_frame"])
    p.add_argument("--end-frame", type=int, default=d["end_frame"])
    p.add_argument("--loop", default=d["loop"],
                   choices=["none", "bounce", "repeat"])
    p.add_argument("--interp", type=int, default=d["interp"],
                   help="Interpolated frames per source frame (10 = 10× smoother)")
    p.add_argument("--zoom-factor", type=float, default=d["zoom_factor"],
                   help="Zoom factor used during rendering (must match)")
    p.add_argument("--mask-svg", type=str, default=d["mask_svg"],
                   help="SVG mask to apply to every output frame")
    p.add_argument("--output", "-o", default=d["output"],
                   help="Output file path (auto-generated if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

