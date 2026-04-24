"""
Fractal Studio – PyQt6 desktop GUI for render.py & video.py.

Usage:  python gui.py
"""
import sys
import os
import glob
import signal

from PyQt6.QtCore import Qt, QProcess, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QKeyEvent, QTextCursor, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QTabWidget,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QFileDialog, QPlainTextEdit, QScrollArea,
    QSizePolicy, QListWidget, QAbstractItemView, QStatusBar,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
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


def _save_preview_params(folder, params):
    """Save a dict of preview parameters as params.json in *folder*."""
    import json
    path = os.path.join(folder, "params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=str)
    return path


def _load_preview_params(folder):
    """Load params.json from *folder*, return dict or None."""
    import json
    path = os.path.join(folder, "params.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def next_preview_folder():
    """Return the next numbered preview folder path (e.g. results/_preview/003/)."""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    existing = [d for d in os.listdir(PREVIEW_DIR)
                if os.path.isdir(os.path.join(PREVIEW_DIR, d)) and d.isdigit()]
    num = max((int(d) for d in existing), default=0) + 1
    path = os.path.join(PREVIEW_DIR, f"{num:03d}")
    os.makedirs(path, exist_ok=True)
    return path, num


def detect_recursion(c, center, kernel, power, current_zf, log=None, scale_start=2.0):
    """Detect self-similarity / periodic loops for the given parameters.

    Returns a dict with keys: type, period, abs_lam, arg_lam_deg, n_rot, k,
    optimal_zf, quality, loop_start_frame — or None if nothing found.
    """
    import math, cmath

    if log is None:
        log = lambda msg: None

    def f(z):
        if kernel == "poly":
            return z ** power + c
        elif kernel == "sin":
            return c * cmath.sin(z)
        elif kernel == "cos":
            return c * cmath.cos(z)

    def f_deriv(z):
        if kernel == "poly":
            return power * z ** (power - 1)
        elif kernel == "sin":
            return c * cmath.cos(z)
        elif kernel == "cos":
            return -c * cmath.sin(z)

    results = []

    # ── Strategy A: repelling fixed point at zoom center ─────
    z0 = center
    try:
        fz0 = f(z0)
        deriv_z0 = f_deriv(z0)
    except (OverflowError, ValueError):
        fz0 = float('inf')
        deriv_z0 = 0

    abs_deriv = abs(deriv_z0)
    is_fixed = abs(fz0 - z0) < 1e-10

    log(f"  Center z0 = {z0}\n")
    log(f"  f(z0) = {fz0:.8f} {'= z0 (fixed point!)' if is_fixed else ''}\n")
    log(f"  f'(z0) = {deriv_z0:.8f}, |f'(z0)| = {abs_deriv:.8f}\n")

    if is_fixed and abs_deriv > 1.0:
        log(f"  ✓ Repelling fixed point: |f'| = {abs_deriv:.6f} > 1\n")
        lam = deriv_z0
        abs_lam = abs_deriv
        arg_lam = cmath.phase(lam)
        arg_deg = math.degrees(arg_lam)
        log(f"  Self-similarity multiplier λ = {lam:.8f}\n")
        log(f"  |λ| = {abs_lam:.10f}, arg(λ) = {arg_deg:.4f}°\n")

        if abs(arg_deg) < 0.01:
            n_rot = 1
        else:
            n_rot = None
            for n in range(1, 361):
                total_rot = n * arg_deg
                if abs(total_rot - round(total_rot / 360) * 360) < 0.5:
                    n_rot = n
                    break
            if n_rot is None:
                n_rot = round(360 / abs(arg_deg))

        total_zoom = (1.0 / abs_lam) ** n_rot
        log(f"  Rotation loop: {n_rot} self-sim steps "
            f"= {n_rot * arg_deg:.1f}° "
            f"≈ {round(n_rot * arg_deg / 360)}×360°\n")
        log(f"  Total zoom per loop: {total_zoom:.10f}\n")

        k_exact = math.log(total_zoom) / math.log(current_zf)
        k = max(1, round(k_exact))
        optimal_zf = total_zoom ** (1.0 / k)

        log(f"  k_exact = {k_exact:.2f} → k = {k} frames per loop\n")
        log(f"  Optimal zoom factor: {optimal_zf:.9f}\n")

        results.append({
            "type": "repelling_fixed",
            "period": 1,
            "multiplier": lam,
            "abs_lam": abs_lam,
            "arg_lam_deg": arg_deg,
            "n_rot": n_rot,
            "k": k,
            "optimal_zf": optimal_zf,
            "quality": "exact" if abs(arg_deg * n_rot - round(arg_deg * n_rot / 360) * 360) < 0.01 else "approximate",
        })

    # ── Strategy B: attracting periodic orbit ────────────────
    z = center if center != 0 else 0.001 + 0.001j
    escaped = False
    for _ in range(5000):
        try:
            z = f(z)
        except (OverflowError, ValueError):
            escaped = True
            break
        if abs(z) > 1e10:
            escaped = True
            break
    if not escaped:
        z_ref = z
        tol = 1e-8
        period = None
        z_test = z_ref
        for p in range(1, 2001):
            z_test = f(z_test)
            if abs(z_test - z_ref) < tol:
                period = p
                break

        if period is not None:
            log(f"\n  Attracting cycle: period {period}\n")
            cycle = [z_ref]
            z_cur = z_ref
            for _ in range(period - 1):
                z_cur = f(z_cur)
                cycle.append(z_cur)

            lam = 1.0 + 0j
            for z_i in cycle:
                lam *= f_deriv(z_i)

            abs_lam = abs(lam)
            arg_deg = math.degrees(cmath.phase(lam))
            log(f"  Cycle multiplier λ = {lam:.8f}\n")
            log(f"  |λ| = {abs_lam:.10f}, arg(λ) = {arg_deg:.4f}°\n")

            if 1e-12 < abs_lam < 1.0:
                # Rotation-aware visual loop length.
                # One self-similarity step of the cycle: scale·|λ|, rotation arg(λ).
                # Find smallest n_rot so that n_rot·arg(λ) ≈ multiple of 360°.
                if abs(arg_deg) < 0.01:
                    n_rot = 1
                else:
                    n_rot = None
                    for n in range(1, 361):
                        total_rot = n * arg_deg
                        if abs(total_rot - round(total_rot / 360) * 360) < 0.5:
                            n_rot = n
                            break
                    if n_rot is None:
                        n_rot = max(1, round(360 / abs(arg_deg)))

                total_zoom = abs_lam ** n_rot        # shrink factor per full visual loop
                k_exact = math.log(total_zoom) / math.log(current_zf)
                k = max(1, round(k_exact))
                optimal_zf = total_zoom ** (1.0 / k)

                log(f"  Rotation loop: {n_rot} self-sim steps "
                    f"= {n_rot * arg_deg:.1f}° "
                    f"≈ {round(n_rot * arg_deg / 360)}×360°\n")
                log(f"  Total zoom per loop: {total_zoom:.10f}\n")
                log(f"  k_exact = {k_exact:.2f} → k = {k} frames per loop\n")
                log(f"  Optimal zoom factor: {optimal_zf:.9f}\n")

                results.append({
                    "type": "attracting_cycle",
                    "period": period,
                    "multiplier": lam,
                    "abs_lam": abs_lam,
                    "arg_lam_deg": arg_deg,
                    "n_rot": n_rot,
                    "k": k,
                    "optimal_zf": optimal_zf,
                    "quality": "exact" if abs(arg_deg * n_rot - round(arg_deg * n_rot / 360) * 360) < 0.01 else "approximate",
                })
        elif not escaped:
            log(f"  No attracting cycle found (orbit did not converge).\n")

    if not results:
        log("  ⚠ No self-similarity detected.\n")
        return None

    best = sorted(results, key=lambda r: (r["quality"] != "exact", -r["k"]))[0]

    # ── Estimate loop start frame ────────────────────────────────────────
    # Self-similarity only holds in the linear regime near the fixed point.
    # For sin/cos: sin(z)≈z when |z|<<1, relative error ~ |z|²/6
    # For poly z^n+c near fixed point z0: need |scale| small enough that
    #   higher-order terms of f^n are negligible.
    # We use: the loop starts when scale < threshold.
    import math
    optimal_zf = best["optimal_zf"]
    if kernel in ("sin", "cos"):
        # Need scale² / 6 < 0.01 → scale < ~0.245
        threshold = 0.25
    else:
        # For polynomial: the approximation quality depends on the fixed point
        # and the power. Use a conservative threshold.
        threshold = 0.3

    if scale_start <= threshold:
        loop_start = 0
    else:
        # scale_start * zf^f < threshold → f > log(threshold/scale_start) / log(zf)
        loop_start = math.ceil(
            math.log(threshold / scale_start) / math.log(optimal_zf)
        )

    # Align loop_start to a multiple of loop period for cleaner behaviour
    k = best["k"]
    # Don't align – just use the raw start. The important thing is that
    # frames before loop_start are always rendered, never copied.
    best["loop_start_frame"] = loop_start

    typ = best["type"].replace("_", " ")
    log(f"\n  ═══ Best match: {typ} ═══\n")
    log(f"  Loop: {best['k']} frames at zoom_factor = "
        f"{best['optimal_zf']:.9f}\n")
    if best["type"] in ("repelling_fixed", "attracting_cycle"):
        log(f"  ({best['n_rot']} self-similarity steps "
            f"× {best['arg_lam_deg']:.1f}° each)\n")
    log(f"  Loop starts at frame {loop_start} "
        f"(scale {scale_start * optimal_zf**loop_start:.6f} < {threshold})\n")
    log(f"  Frames 0–{loop_start - 1}: unique (pre-loop)\n"
        if loop_start > 0 else "  Loop starts immediately (scale already small)\n")

    return best


def find_nearest_boundary_point(c, center, kernel, power, max_iter, bailout,
                                scale, log=None):
    """Find the nearest Julia set boundary point to *center*.

    Strategy:
    1. Cast rays from *center* in many directions.
    2. Binary-search along each ray for the escape-time boundary
       (transition from "escapes" to "does not escape").
    3. Return the closest boundary point found.
    4. Optionally refine via Newton's method toward a nearby repelling
       periodic point (these are dense on the Julia set boundary).

    For Mandelbrot mode (c is None), the iteration is z²+c_pixel instead.

    Returns (z_boundary, distance) or (None, None) if nothing found.
    """
    import cmath, math

    if log is None:
        log = lambda msg: None

    def escapes(z0, iters=None):
        """Return True if z0 escapes under iteration."""
        if iters is None:
            iters = max_iter
        z = z0
        for _ in range(iters):
            try:
                if kernel == "poly":
                    z = z ** power + c
                elif kernel == "sin":
                    z = c * cmath.sin(z)
                elif kernel == "cos":
                    z = c * cmath.cos(z)
            except (OverflowError, ValueError):
                return True
            if kernel == "poly":
                if abs(z) > bailout:
                    return True
            else:
                if abs(z.imag) > bailout:
                    return True
        return False

    # ── Phase 1: ray-based binary search ─────────────────────────────────
    n_rays = 64
    n_samples = 32  # samples along each ray
    search_radius = scale * 0.7
    best_pt = None
    best_dist = float('inf')

    log(f"  Searching {n_rays} rays, radius={search_radius:.6f}…\n")

    for i in range(n_rays):
        angle = 2 * math.pi * i / n_rays
        direction = cmath.exp(1j * angle)

        # Sample points along the ray to find a transition
        fracs = [j / n_samples for j in range(n_samples + 1)]
        prev_pt = center
        prev_esc = escapes(center)

        for frac in fracs[1:]:
            cur_pt = center + search_radius * frac * direction
            cur_esc = escapes(cur_pt)
            if cur_esc != prev_esc:
                # Found a transition — binary search between prev_pt and cur_pt
                near, far = prev_pt, cur_pt
                near_esc = prev_esc
                for _ in range(64):
                    mid = (near + far) / 2
                    if escapes(mid) == near_esc:
                        near = mid
                    else:
                        far = mid
                boundary = (near + far) / 2
                dist = abs(boundary - center)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = boundary
                break  # take the first (closest) transition on this ray
            prev_pt = cur_pt
            prev_esc = cur_esc

    if best_pt is None:
        log("  ⚠ No boundary found along any ray.\n")
        return None, None

    log(f"  Ray search: nearest boundary at {best_pt:.10f}, "
        f"dist={best_dist:.2e}\n")

    # ── Phase 2: local refinement via gradient descent on escape-time ────
    # Sample a tiny grid around best_pt and pick the point with the
    # steepest escape-time gradient (= most "on the boundary").
    refine_r = best_dist * 0.1 if best_dist > 0 else search_radius * 0.001
    grid_n = 21
    best_refined = best_pt
    best_gradient = 0

    for gi in range(grid_n):
        for gj in range(grid_n):
            dx = refine_r * (2 * gi / (grid_n - 1) - 1)
            dy = refine_r * (2 * gj / (grid_n - 1) - 1)
            pt = best_pt + complex(dx, dy)
            # Compute escape time
            z = pt
            count = max_iter
            for it in range(max_iter):
                try:
                    if kernel == "poly":
                        z = z ** power + c
                    elif kernel == "sin":
                        z = c * cmath.sin(z)
                    elif kernel == "cos":
                        z = c * cmath.cos(z)
                except (OverflowError, ValueError):
                    count = it
                    break
                if kernel == "poly":
                    if abs(z) > bailout:
                        count = it
                        break
                else:
                    if abs(z.imag) > bailout:
                        count = it
                        break
            # Approximate gradient: count difference with center escape time
            # We want the point where escape time is most "fragile"
            # i.e. where small moves cause large count changes
            # Use count as a proxy — boundary points have intermediate counts
            if 0 < count < max_iter:
                # Score: prefer counts that are neither too low nor too high
                score = min(count, max_iter - count)
                if score > best_gradient:
                    best_gradient = score
                    best_refined = pt

    if best_gradient > 0:
        best_pt = best_refined
        best_dist = abs(best_pt - center)
        log(f"  Refined: {best_pt:.10f}, dist={best_dist:.2e}\n")

    # ── Phase 3: Newton refinement toward repelling fixed/periodic point ──
    # Try to find a nearby repelling periodic point (period 1..4).
    # These are dense on the Julia set boundary and are ideal zoom targets.
    best_periodic = None
    best_periodic_dist = float('inf')

    for period in range(1, 5):
        # Newton's method to solve f^p(z) = z near best_pt
        z0 = best_pt
        for newton_step in range(200):
            # Compute f^p(z0) and its derivative
            z = z0
            dz = 1.0 + 0j  # d(f^p)/dz via chain rule
            ok = True
            for _ in range(period):
                try:
                    if kernel == "poly":
                        dz = power * z ** (power - 1) * dz
                        z = z ** power + c
                    elif kernel == "sin":
                        dz = c * cmath.cos(z) * dz
                        z = c * cmath.sin(z)
                    elif kernel == "cos":
                        dz = -c * cmath.sin(z) * dz
                        z = c * cmath.cos(z)
                except (OverflowError, ValueError):
                    ok = False
                    break
                if abs(z) > 1e15 or abs(dz) > 1e30:
                    ok = False
                    break
            if not ok:
                break

            residual = z - z0
            jacobian = dz - 1.0  # d(f^p(z)-z)/dz

            if abs(jacobian) < 1e-30:
                break

            step = residual / jacobian
            z0 = z0 - step

            if abs(step) < 1e-14:
                # Converged — check if it's repelling (|multiplier| > 1)
                z_check = z0
                mult = 1.0 + 0j
                for _ in range(period):
                    try:
                        if kernel == "poly":
                            mult *= power * z_check ** (power - 1)
                            z_check = z_check ** power + c
                        elif kernel == "sin":
                            mult *= c * cmath.cos(z_check)
                            z_check = c * cmath.sin(z_check)
                        elif kernel == "cos":
                            mult *= -c * cmath.sin(z_check)
                            z_check = c * cmath.cos(z_check)
                    except (OverflowError, ValueError):
                        mult = 0
                        break

                if abs(mult) > 1.0:
                    d = abs(z0 - center)
                    if d < best_periodic_dist:
                        best_periodic = z0
                        best_periodic_dist = d
                        log(f"  Found repelling period-{period} point: "
                            f"{z0:.10f}, |λ|={abs(mult):.4f}, "
                            f"dist={d:.2e}\n")
                break

    if best_periodic is not None and best_periodic_dist < best_dist * 2:
        # Prefer the periodic point if it's reasonably close
        log(f"  ✓ Using repelling periodic point: {best_periodic:.10f}\n")
        return best_periodic, best_periodic_dist
    else:
        log(f"  ✓ Using boundary point: {best_pt:.10f}\n")
        return best_pt, best_dist


def find_clean_boundary_points(c, kernel, power, zoom_factor, scale_start,
                               width, max_iter, bailout,
                               log=None, max_points=12):
    """Find precision-friendly zoom-center candidates on the Julia boundary.

    The ideal center satisfies three things:
      (1) it lies on the Julia-set boundary,
      (2) its coordinates are representable *exactly* in float64
          (otherwise every center-offset carries ~|z|·ε error from frame 1),
      (3) forward iteration from it is numerically stable – in practice
          that means it is a *repelling periodic point*: f^p(z) = z with
          |(f^p)'(z)| > 1, so the orbit is closed instead of drifting.

    Strategy:
      A. Newton-solve f^p(z) = z for p = 1..8 from a dense seed grid →
         full set of repelling periodic points.
      B. Enumerate "dyadic" grid points (k + i·m)/2ⁿ inside the viewport,
         keep those verified on the boundary.
      C. If a periodic point lies within ~1e-10 of a dyadic value, snap
         it (and verify the snap is still (nearly) periodic) – these are
         the gold standard: simple *and* stable.
      D. Score: +200 for exactly representable, +50/p for period-p
         periodic points, + simpler-denominator bonus, plus the base
         "clean frame count" estimate.  Exactly-representable centres
         drop the O(|z|·ε) representation noise completely.
    """
    import cmath, math

    if log is None:
        log = lambda m: None

    EPS_FLOAT = 2.22e-16
    EPS_ITER  = 1e-14

    # ── f, f', iterated composition ────────────────────────────────────
    def f(z):
        if kernel == "poly":
            return z ** power + c
        if kernel == "sin":
            return c * cmath.sin(z)
        return c * cmath.cos(z)

    def fprime(z):
        if kernel == "poly":
            return power * z ** (power - 1)
        if kernel == "sin":
            return c * cmath.cos(z)
        return -c * cmath.sin(z)

    def f_iter(z, n):
        """Return (f^n(z), (f^n)'(z)) via chain rule."""
        d = 1 + 0j
        for _ in range(n):
            d = fprime(z) * d
            z = f(z)
            if abs(z) > 1e12 or abs(d) > 1e30:
                raise OverflowError()
        return z, d

    def escapes(z0, iters=200):
        z = z0
        for _ in range(min(iters, max_iter)):
            try:
                z = f(z)
            except (OverflowError, ValueError):
                return True
            if kernel == "poly":
                if abs(z) > bailout:
                    return True
            else:
                if abs(z.imag) > bailout:
                    return True
        return False

    # ── Dyadic helpers ─────────────────────────────────────────────────
    def dyadic_denom_exp(x, tol=1e-10, max_n=12):
        """Smallest n such that |x − k/2ⁿ| ≤ tol; returns (n, k) or (None, None)."""
        for n in range(0, max_n + 1):
            d = 2 ** n
            k = round(x * d)
            if abs(x - k / d) <= tol:
                return n, int(k)
        return None, None

    def is_exact_dyadic(z):
        """True iff real and imag parts are bit-exact dyadic rationals."""
        nr, _ = dyadic_denom_exp(z.real, tol=0.0)
        ni, _ = dyadic_denom_exp(z.imag, tol=0.0)
        return nr is not None and ni is not None

    def max_dyadic_denom(z):
        nr, _ = dyadic_denom_exp(z.real)
        ni, _ = dyadic_denom_exp(z.imag)
        if nr is None or ni is None:
            return None
        return max(nr, ni)

    # ── A. Repelling periodic points via Newton ───────────────────────
    viewport_r = max(scale_start, 2.0)
    periodic = []   # list of (z, period, |multiplier|)

    # Seed grid: uniform + explicit simple points
    seeds = []
    GRID = 9
    for i in range(GRID):
        for j in range(GRID):
            r = viewport_r * (2 * (i + 0.5) / GRID - 1)
            s = viewport_r * (2 * (j + 0.5) / GRID - 1)
            seeds.append(complex(r, s))
    for pt in (0+0j, 1+0j, -1+0j, 1j, -1j,
               0.5+0j, -0.5+0j, 0.5j, -0.5j,
               1+1j, -1-1j, 1-1j, -1+1j,
               0.25+0j, 0.25j, -0.25+0j, -0.25j):
        if abs(pt) <= viewport_r * 1.2:
            seeds.append(pt)

    for p in range(1, 9):
        roots_p = []
        for seed in seeds:
            z = seed
            converged = False
            for _ in range(80):
                try:
                    fn_z, fn_d = f_iter(z, p)
                    g  = fn_z - z
                    gp = fn_d - 1
                    if abs(gp) < 1e-30:
                        break
                    step = g / gp
                    z = z - step
                    if abs(z) > viewport_r * 3:
                        break
                    if abs(step) < 1e-13:
                        converged = True
                        break
                except (OverflowError, ValueError, ZeroDivisionError):
                    break
            if not converged or abs(z) > viewport_r * 2:
                continue
            try:
                fn_z, fn_d = f_iter(z, p)
            except (OverflowError, ValueError):
                continue
            if abs(fn_z - z) > 1e-9 * max(1.0, abs(z)):
                continue
            mult = abs(fn_d)
            if mult <= 1.01:           # not repelling enough
                continue
            # Primitive-period check: not a lower-period point in disguise
            primitive = True
            for q in range(1, p):
                if p % q == 0:
                    try:
                        fq, _ = f_iter(z, q)
                        if abs(fq - z) < 1e-9 * max(1.0, abs(z)):
                            primitive = False
                            break
                    except (OverflowError, ValueError):
                        pass
            if not primitive:
                continue
            # Dedup against already-found roots of this period
            if any(abs(z - r[0]) < 1e-6 for r in roots_p):
                continue
            roots_p.append((z, p, mult))
        periodic.extend(roots_p)
        log(f"  Period {p}: {len(roots_p)} repelling fixed points\n")

    # ── B. Dyadic boundary grid ────────────────────────────────────────
    dyadic_points = []
    seen_keys = set()
    for n in range(0, 6):  # denominators 1..32
        denom = 2 ** n
        K = min(int(math.ceil(viewport_r * denom)) + 1, 40)
        for k in range(-K, K + 1):
            for m in range(-K, K + 1):
                # Primitive key: reduce (k, m, 2^n) so each point counted once
                key_n = n
                kk, mm = k, m
                while key_n > 0 and kk % 2 == 0 and mm % 2 == 0:
                    kk //= 2
                    mm //= 2
                    key_n -= 1
                key = (kk, mm, key_n)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                z = complex(k / denom, m / denom)
                if abs(z) > viewport_r:
                    continue
                dyadic_points.append(z)

    def verified_boundary_depth(z):
        """Return the smallest scale (as log10) at which escape-contrast is
        still detectable around z, or None if no contrast at any scale.

        Julia-set boundary points show contrast at *every* scale.  Points
        that merely happen to sit near the boundary lose contrast once the
        probe radius falls below the distance to the true boundary.
        """
        if abs(z) > viewport_r * 1.5:
            return None
        directions = [cmath.exp(1j * math.pi * k / 4) for k in range(8)]
        esc_c = escapes(z)
        last_ok = None
        for log10_r in (-2, -3, -4, -5, -6, -7, -8):
            r = max(abs(z), 1.0) * (10 ** log10_r)
            for d in directions:
                if escapes(z + r * d) != esc_c:
                    last_ok = log10_r
                    break
            else:
                # No contrast at this scale – stop testing deeper scales.
                return last_ok
        return last_ok

    def on_boundary(z):
        return verified_boundary_depth(z) is not None

    # Verify every dyadic grid point to the deepest scale we can.
    boundary_dyadic = []
    for z in dyadic_points:
        depth = verified_boundary_depth(z)
        # Require contrast down to at least scale 1e-6 to keep only points
        # that will survive meaningful zoom depth (~ frame 130 at z=0.92).
        if depth is not None and depth <= -6:
            boundary_dyadic.append((z, depth))
    log(f"  Dyadic grid: {len(dyadic_points)} tested, "
        f"{len(boundary_dyadic)} verified on boundary (contrast ≤ 1e-6)\n")

    # ── C. Build unified candidate list (with snap-to-dyadic) ──────────
    def try_snap_periodic(z, p):
        """If z is close to a dyadic point whose iterate is still ~periodic, return snapped z."""
        nr, kr = dyadic_denom_exp(z.real)
        ni, ki = dyadic_denom_exp(z.imag)
        if nr is None or ni is None:
            return None
        zs = complex(kr / (2 ** nr), ki / (2 ** ni))
        try:
            fn_zs, _ = f_iter(zs, p)
            if abs(fn_zs - zs) < 1e-6 * max(1.0, abs(zs)):
                return zs
        except (OverflowError, ValueError):
            pass
        return None

    # Attach a "verified_depth" (log10 of smallest-verified probe scale) to
    # every candidate.  Periodic points are on the boundary by definition
    # at all scales, so we give them a symbolic −∞ (represented as −18).
    PERIODIC_DEPTH = -18

    all_cands = []
    for z, p, mult in periodic:
        snap = try_snap_periodic(z, p)
        if snap is not None:
            all_cands.append({
                "z": snap, "period": p, "multiplier": mult,
                "type": f"periodic-{p} (dyadic)",
                "verified_depth": PERIODIC_DEPTH,
            })
        else:
            all_cands.append({
                "z": z, "period": p, "multiplier": mult,
                "type": f"periodic-{p}",
                "verified_depth": PERIODIC_DEPTH,
            })

    for z, depth in boundary_dyadic:
        if any(abs(z - c2["z"]) < 1e-8 for c2 in all_cands):
            continue
        all_cands.append({
            "z": z, "period": None, "multiplier": None, "type": "dyadic",
            "verified_depth": depth,
        })

    # ── D. Scoring ─────────────────────────────────────────────────────
    # "clean_frames" estimates how many frames the point will still show
    # boundary structure under zoom:
    #   – periodic points: unlimited (drift-free if exact-dyadic, otherwise
    #     bounded by the representation error)
    #   – pure dyadic points: bounded by the scale at which we could still
    #     detect boundary contrast around them.
    INF_FRAMES = 10 ** 6

    for cand in all_cands:
        z = cand["z"]
        mag = abs(z)
        cand["magnitude"] = mag
        exact = is_exact_dyadic(z)
        cand["exact_dyadic"] = exact
        depth = cand["verified_depth"]   # log10 of smallest verified scale

        # Frames until zoom enters the unverified regime
        if depth <= PERIODIC_DEPTH:
            cf_verified = INF_FRAMES
        else:
            try:
                lf = math.log(zoom_factor)
                cf_verified = 0 if lf >= 0 else max(
                    0, int(math.log(10 ** depth * width / scale_start) / lf))
            except (ValueError, ZeroDivisionError):
                cf_verified = 0

        # Frames until centre representation error dominates pixel spacing
        if exact:
            cf_drift = INF_FRAMES
        else:
            noise = max(mag * EPS_FLOAT, 1e-18)
            try:
                lf = math.log(zoom_factor)
                cf_drift = 0 if lf >= 0 else max(
                    0, int(math.log(noise * width / scale_start) / lf))
            except (ValueError, ZeroDivisionError):
                cf_drift = 0

        cand["clean_frames"] = min(cf_verified, cf_drift)

        denom = max_dyadic_denom(z)
        cand["dyadic_denom"] = denom

        # Composite score
        score = min(cand["clean_frames"], 1000)
        if cand["period"] is not None:
            score += 1500 + 200 / cand["period"]   # periodic = boundary for real
        if exact:
            score += 800                            # bit-exact centre
        if denom is not None and denom < 12:
            score += max(0, 10 - denom) * 8
        cand["score"] = score

    # Sort, dedupe tightly
    all_cands.sort(key=lambda d: -d["score"])
    unique = []
    for s in all_cands:
        if not any(abs(s["z"] - u["z"]) < 1e-8 for u in unique):
            unique.append(s)
        if len(unique) >= max_points:
            break

    n_exact = sum(1 for u in unique if u["exact_dyadic"])
    n_periodic = sum(1 for u in unique if u["period"] is not None)
    log(f"  Returning {len(unique)} candidates "
        f"({n_exact} exactly representable, {n_periodic} periodic)\n")
    return unique


class CleanCentersDialog(QDialog):
    """Modal dialog presenting clean-boundary-point candidates."""

    def __init__(self, candidates, current_zoom, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clean Boundary Point Candidates")
        self.resize(760, 420)
        self.selected = None
        self.candidates = candidates

        layout = QVBoxLayout(self)
        info = QLabel(
            "Boundary points ranked by floating-point precision friendliness.\n"
            f"‘Clean frames’ = estimated zoom depth the centre still shows "
            f"boundary structure (at zoom factor {current_zoom:g}).\n"
            "periodic-N rows are guaranteed on the fractal boundary at every "
            "scale.  ‘dyadic’ rows were only verified to a finite scale and "
            "may go flat beyond that zoom depth.")
        info.setStyleSheet("color: #888; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        table = QTableWidget()
        headers = ["Real", "Imag", "|z|", "Clean frames", "Period", "Type"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(candidates))
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        for i, cand in enumerate(candidates):
            z = cand["z"]
            exact = cand.get("exact_dyadic", False)
            # Show full repr for dyadic (they are exact), shorter for generic
            fmt = "{:+.17g}" if exact else "{:+.12g}"
            table.setItem(i, 0, QTableWidgetItem(fmt.format(z.real)))
            table.setItem(i, 1, QTableWidgetItem(fmt.format(z.imag)))
            table.setItem(i, 2, QTableWidgetItem(f"{cand['magnitude']:.4g}"))
            cf = cand['clean_frames']
            table.setItem(i, 3, QTableWidgetItem("∞" if cf > 99999 else str(cf)))
            p = cand.get("period")
            table.setItem(i, 4, QTableWidgetItem(str(p) if p else "–"))
            table.setItem(i, 5, QTableWidgetItem(cand.get("type", "")))

        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        if len(candidates):
            table.selectRow(0)
        table.doubleClicked.connect(self._accept)
        self.table = table
        layout.addWidget(table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        use_btn = QPushButton("Use as Center")
        use_btn.setDefault(True)
        use_btn.clicked.connect(self._accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(use_btn)
        layout.addLayout(btn_row)

    def _accept(self, *_):
        row = self.table.currentRow()
        if 0 <= row < len(self.candidates):
            self.selected = self.candidates[row]["z"]
            self.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  Set Selection Panel  (Tab 1)
# ═══════════════════════════════════════════════════════════════════════════════
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
        """Create a new 'extra constant' row."""
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

    def extra_constants_json(self):
        """Return the current panning transitions as a JSON string, or ''."""
        import json as _json
        rows = []
        for e in self._extra_const_rows:
            rows.append({
                'c_real': e['c_real'].value(),
                'c_imag': e['c_imag'].value(),
                'start':  e['start'].value(),
                'length': e['length'].value(),
            })
        return _json.dumps(rows) if rows else ''

    def _toggle_generator_options(self, *_args):
        gen = self.generator.currentText()
        self.flower_group.setVisible(gen == "flower")
        is_mandelbrot = (gen == "mandelbrot")
        # Hide c and kernel for Mandelbrot (c = pixel coordinate, no kernel choice)
        for w in (self._c_real_label, self.c_real, self._c_imag_label, self.c_imag,
                  self._kernel_label, self.kernel):
            w.setVisible(not is_mandelbrot)

    def _apply_recursion_zoom(self):
        """Copy detected optimal zoom factor into the zoom factor field of ImageRenderingPanel."""
        txt = self.recursion_zoom.text().strip()
        if txt and self._image_panel:
            self._image_panel.zoom_factor.setValue(float(txt))

    def _run_adjust_center(self):
        """Find the nearest Julia set boundary point and update center coords."""
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
        """Present precision-friendly boundary points as center candidates."""
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
        # Width/zoom factor come from the Image Rendering panel (fallback: sensible defaults).
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
                return detect_recursion(c, center, kernel, power, current_zf, log=_log, scale_start=ss)
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
                preview_dir, preview_num = next_preview_folder()
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
                _save_preview_params(preview_dir, {
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
        # Snapshot panning transitions so the preview matches the full render.
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
                preview_dir, preview_num = next_preview_folder()
                _save_preview_params(preview_dir, {
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

        self.zoom_factor = QDoubleSpinBox(); self.zoom_factor.setRange(0.5, 0.999999999); self.zoom_factor.setDecimals(9); self.zoom_factor.setSingleStep(0.01); self.zoom_factor.setValue(0.92
        )
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
        # Colorizers are now produced via the Coloring tab from the raw
        # series – rendering only produces raw counts + the greyscale preview.
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
# ═══════════════════════════════════════════════════════════════════════════════
class SegmentWidget(QGroupBox):
    """One entry in the colour program.

    The first segment is the *base* colour – it has no start_frame /
    transition_length controls and cannot be removed.  Subsequent segments
    define a transition that starts at ``start_frame`` (weight 0) and
    completes at ``start_frame + transition_length`` (weight 1)."""

    removed = pyqtSignal(object)      # emits self

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
        # Pick a colourful default for the base segment
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
        """Serialise to the dict form expected by colorize.py."""
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
    """Second-stage panel: pick a rendered raw series, build a colour
    program (one or more colorizers with smooth transitions), produce PNGs
    via colorize.py (no fractal iteration involved)."""

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

        # Container that holds SegmentWidgets.  Separate layout so we can
        # add/remove entries without losing the "Add colour" button order.
        self._segments_container = QWidget()
        self._segments_layout = QVBoxLayout(self._segments_container)
        self._segments_layout.setContentsMargins(0, 0, 0, 0)
        self._segments_layout.setSpacing(6)
        prog_layout.addWidget(self._segments_container)

        # Add-button below the segment list
        self.add_color_btn = QPushButton("\u2795  Add colour")
        self.add_color_btn.clicked.connect(self._add_segment)
        prog_layout.addWidget(self.add_color_btn)

        layout.addWidget(program_grp)

        # Seed with the mandatory base segment
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

    # ── Segment management ──────────────────────────────────────────────
    def _add_segment(self, is_first=False):
        seg = SegmentWidget(is_first=is_first)
        if not is_first:
            seg.removed.connect(self._remove_segment)
            # Sensible default: place the new transition after the last one
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

    # ── Series discovery ────────────────────────────────────────────────
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
                import json as _json
                with open(meta_path) as f:
                    meta = _json.load(f)
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

    # ── Program / CLI helpers ───────────────────────────────────────────
    def build_program(self):
        """Collect all segments into the JSON-ready program dict."""
        segments = [s.to_dict() for s in self._segments]
        prog = {
            "smoothing": self.scale_smoothing.value(),
            "segments": segments,
        }
        if self.output_name.text().strip():
            prog["output_name"] = self.output_name.text().strip()
        return prog

    def _write_program_file(self):
        """Write the current program to a timestamped JSON file in
        <project>/_programs/ and return its path."""
        import json
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
    """One segment in a stitch program: a coloured-series folder plus the
    frame range to take from it.  Non-first segments also carry a
    ``transition_length`` that specifies how many frames of the *previous*
    segment get the overlaid zoom-in of this segment's entry frame."""

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
    """Multi-series stitcher – zoom-overlay transitions between coloured
    series.  Produces frames via stitch.py; output goes into
    ``results/<output_name>/frames/`` ready for video.py."""

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
        import json
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
#  Set Selection Viewer – click-to-zoom + preview quick-select
# ═══════════════════════════════════════════════════════════════════════════════
class SetSelectionViewer(QWidget):
    """Image viewer for the Set Selection tab: preview browsing + click-to-zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_panel = None  # set by MainWindow
        layout = QVBoxLayout(self)

        # Quick-select row for numbered preview sets + clear button
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

        # Clickable image label
        self.image_label = QLabel("No frames loaded \u2013 use Preview buttons to generate")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background: #1a1a1a; color: #888;")
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label, 1)

        # Coordinate display
        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.coord_label)

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

        # Load params button
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
        """Load params.json from the current preview folder into the Set Selection panel."""
        if not self.frame_paths or self._set_panel is None:
            return
        folder = os.path.dirname(self.frame_paths[0])
        params = _load_preview_params(folder)
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
        # Flower-specific
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
        # Zoom factor if present
        if "zoom_factor" in params and sp._image_panel:
            sp._image_panel.zoom_factor.setValue(float(params["zoom_factor"]))

        main_win = self.window()
        main_win.console.append_text(
            f"\u2713 Loaded parameters from {os.path.basename(folder)}/params.json\n")

    # ── Coordinate mapping ───────────────────────────────────────────────────

    def _pixel_to_fractal(self, px, py):
        """Convert image-label pixel (px, py) to complex fractal coordinate."""
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
        """Click-to-zoom: render a new image zoomed into clicked coordinates."""
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

        # Determine output folder: reuse the folder the current image lives in
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
        """Render a single 500x500 greyscale frame appended to target_folder."""
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
                from PIL import Image as PILImage, ImageDraw, ImageFont

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

                # Determine output path
                if target_folder and os.path.isdir(target_folder):
                    out_dir = target_folder
                else:
                    out_dir, _ = next_preview_folder()

                # Find the next frame number in the folder
                existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
                existing += sorted(glob.glob(os.path.join(out_dir, "preview_*.png")))
                if existing:
                    # Extract highest number
                    import re
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
                _save_preview_params(out_dir, {
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
                # Reload the folder to include the new frame, scroll to end
                self._load_frames(out_dir)
                self.slider.setValue(self.slider.maximum())
                self.refresh_quick_select()
                main_win.console.append_text(f"\u2713 Frame appended.\n")

        QTimer.singleShot(100, _poll)

    def mouseMoveEvent(self, event):
        """Show fractal coordinates under cursor."""
        pos = self.image_label.mapFrom(self, event.pos())
        z = self._pixel_to_fractal(pos.x(), pos.y())
        if z is not None:
            self.coord_label.setText(f"z = {z.real:+.7f} {z.imag:+.7f}i")
        else:
            self.coord_label.setText("")
        super().mouseMoveEvent(event)

    # ── Frame loading ────────────────────────────────────────────────────────

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
            self.frame_label.setText("0 / 0")
            return
        self.slider.setEnabled(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self._show_frame(0)

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
    """Image viewer for the Image Rendering tab: browse results series + color."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ── Results browser ──────────────────────────────────────────────────
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

        # ── Image display ────────────────────────────────────────────────────
        self.image_label = QLabel("No series selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background: #1a1a1a; color: #888;")
        layout.addWidget(self.image_label, 1)

        # Slider
        bot = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        bot.addWidget(self.slider, 1)
        bot.addWidget(self.frame_label)
        layout.addLayout(bot)

        # Load params button
        self.load_params_btn = QPushButton("\U0001f4e5 Load Set Parameters")
        self.load_params_btn.setToolTip(
            "Load the parameters that generated the currently selected series\n"
            "back into the Set Selection and Image Rendering panels.")
        self.load_params_btn.clicked.connect(self._load_params)
        layout.addWidget(self.load_params_btn)

        self.frame_paths = []
        self._current_pixmap = None
        self._set_panel = None     # wired by MainWindow
        self._image_panel = None   # wired by MainWindow

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

    # ── Load saved parameters ──────────────────────────────────────────────
    def _load_params(self):
        """Load _raw/meta.json from the currently selected series back into
        the Set Selection + Image Rendering panels."""
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
        import json as _json
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
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

    def load_folder(self, folder):
        if not os.path.isabs(folder):
            folder = os.path.join(PROJECT_DIR, folder)
        frames = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
        self.frame_paths = frames
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
#  Tabbed Preview Widget (wrapper synced with input tabs)
# ═══════════════════════════════════════════════════════════════════════════════
class TabbedPreview(QTabWidget):
    """Two viewer tabs that stay in sync with the left-hand parameter tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_viewer = SetSelectionViewer()
        self.render_viewer = RenderingViewer()
        self.addTab(self.set_viewer, "Set Selection")
        self.addTab(self.render_viewer, "Image Rendering")

    # Backward-compat shims
    def load_folder(self, folder):
        self.set_viewer.load_folder(folder)

    def refresh_quick_select(self):
        self.set_viewer.refresh_quick_select()

    def _load_frames(self):
        self.set_viewer._load_frames()


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive Console Widget
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
        km.kernel.shell.run_cell(
            "import sys, os\n"
            f"os.chdir(r'{PROJECT_DIR}')\n"
            f"sys.path.insert(0, r'{PROJECT_DIR}')\n"
            "from generators import GENERATORS\n"
            "from colorizers import COLORIZERS\n"
            "import numpy as np\n"
            "print('Fractal Studio \u2013 IPython ready.')\n"
            "print(f'Generators: {list(GENERATORS.keys())}')\n"
            "print(f'Colorizers: {list(COLORIZERS.keys())}')\n",
            silent=False, store_history=False,
        )
        layout.addWidget(widget)
        self._jupyter = widget
        self._km = km

    def _init_fallback(self, layout):
        lbl = QLabel("Console (output only \u2013 install qtconsole for interactive Python)")
        lbl.setStyleSheet("color: #888; padding: 2px;")
        layout.addWidget(lbl)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet("background: #1a1a1a; color: #ddd;")
        self._log.setMaximumBlockCount(50000)
        layout.addWidget(self._log)

    def append_text(self, text: str):
        if self._jupyter is not None:
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

        # ── Link Coloring <-> RenderingViewer series selection (both ways) ──
        self._link_series_dropdowns(
            self.coloring_panel, self.preview.render_viewer)

        self.console = ConsoleWidget()
        right.addWidget(self.console)

        right.setSizes([500, 300])
        splitter.addWidget(right)
        splitter.setSizes([380, 1020])
        self.setCentralWidget(splitter)

        # Sync left tabs <-> right viewer tabs
        self.left_tabs.currentChanged.connect(self._sync_viewer_tab)

        self.statusBar().showMessage("Ready")

        self.image_panel.start_btn.clicked.connect(self._start_render)
        self.coloring_panel.start_btn.clicked.connect(self._start_coloring)
        self.stitch_panel.start_btn.clicked.connect(self._start_stitch)
        self.video_panel.start_btn.clicked.connect(self._start_video)

        self._process = None
        self._pending_logs = []

    def _sync_viewer_tab(self, index):
        # Set Selection -> set viewer; Image Rendering & Coloring share render viewer.
        if index == 0:
            self.preview.setCurrentIndex(0)
        elif index in (1, 2):
            self.preview.setCurrentIndex(1)
        # Video tab: leave viewer as-is

    def _link_series_dropdowns(self, coloring_panel, render_viewer):
        """Two-way link the series selection (and results root) between
        the Coloring tab and the Image Rendering viewer."""
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
                # Item not yet in dst list – refresh and retry.
                try:
                    dst.parent()  # no-op; ensure dst is alive
                except RuntimeError:
                    return
                idx = dst.findText(text)
            if idx >= 0:
                dst.blockSignals(True)
                dst.setCurrentIndex(idx)
                dst.blockSignals(False)
                # Fire dst's own handler so its UI updates (info label,
                # color buttons, etc.).
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

    def _run_script(self, script, args):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self.console.append_text("\n\u26a0  A process is already running.\n")
            return
        # Snapshot the results/ tree so _on_finished can pick up the newly
        # created series (after render) or the new colour sub-folder
        # (after colorize).
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
        """Record the current state of results/ as {series: set(subdirs)}."""
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
        """Diff current results/ against the pre-run snapshot and select
        the newly-produced series + colour sub-folder in the render viewer."""
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

        # New series (didn't exist before)
        new_series = [s for s in current if s not in snap]
        # Pick target series
        target_series = None
        target_sub = None
        if script == "render.py":
            # Render creates a brand-new series folder.  Prefer greyscale view.
            if new_series:
                target_series = sorted(new_series)[-1]
                if "greyscale" in current.get(target_series, set()):
                    target_sub = "greyscale"
                else:
                    subs = sorted(current.get(target_series, set()))
                    target_sub = subs[0] if subs else None
        elif script == "colorize.py":
            # Colorize adds a sub-folder to an existing series.
            for s, subs in current.items():
                added = subs - snap.get(s, set())
                if added:
                    target_series = s
                    # First colour produced (sorted for determinism)
                    target_sub = sorted(added)[0]
                    break
        elif script == "stitch.py":
            if new_series:
                target_series = sorted(new_series)[-1]
                subs = current.get(target_series, set())
                # Prefer 'frames' (the folder stitch.py writes to)
                target_sub = "frames" if "frames" in subs else (
                    sorted(subs)[0] if subs else None)

        if not target_series:
            # Fallback: keep old behaviour – pick the last alphabetical entry
            if self.preview.render_viewer.series_combo.count() > 0:
                self.preview.render_viewer.series_combo.setCurrentIndex(
                    self.preview.render_viewer.series_combo.count() - 1)
            return

        rv = self.preview.render_viewer
        idx = rv.series_combo.findText(target_series)
        if idx >= 0:
            rv.series_combo.setCurrentIndex(idx)
        # Load the chosen sub-folder by simulating a button click
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

    def _start_render(self):
        sp = self.set_panel
        rp = self.image_panel
        c = complex(sp.c_real.value(), sp.c_imag.value())
        center = complex(sp.center_real.value(), sp.center_imag.value())
        power = sp.power.value()
        kernel = sp.kernel.currentText()
        current_zf = rp.zoom_factor.value()
        ss = rp.scale_start.value()

        # Skip the recursion check when the user has configured a c-panning
        # sequence -- loops are disabled in render.py in that case anyway.
        extras_json = ''
        try:
            extras_json = sp.extra_constants_json()
        except AttributeError:
            pass
        args = rp.build_args()
        if extras_json:
            self.console.append_text(
                "\n" + "Additional constants present -- skipping recursion check.\n")
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
                from PyQt6.QtWidgets import QMessageBox
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
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
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        script_dir = os.path.dirname(os.path.abspath(script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        with open(script) as f:
            exec(compile(f.read(), script, "exec"), {"__name__": "__main__"})
    else:
        main()

