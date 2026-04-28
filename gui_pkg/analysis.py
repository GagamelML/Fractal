"""Fractal-set analysis helpers (recursion detection, boundary search) and
the dialog used to present clean-center candidates."""
import math
import cmath

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)


def detect_recursion(c, center, kernel, power, current_zf, log=None,
                     scale_start=2.0):
    """Detect self-similarity / periodic loops for the given parameters.

    Returns a dict with keys: type, period, abs_lam, arg_lam_deg, n_rot, k,
    optimal_zf, quality, loop_start_frame -- or None if nothing found.
    """
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

                total_zoom = abs_lam ** n_rot
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
            log("  No attracting cycle found (orbit did not converge).\n")

    if not results:
        log("  ⚠ No self-similarity detected.\n")
        return None

    best = sorted(results, key=lambda r: (r["quality"] != "exact", -r["k"]))[0]

    optimal_zf = best["optimal_zf"]
    if kernel in ("sin", "cos"):
        threshold = 0.25
    else:
        threshold = 0.3

    if scale_start <= threshold:
        loop_start = 0
    else:
        loop_start = math.ceil(
            math.log(threshold / scale_start) / math.log(optimal_zf)
        )

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
    """Find the nearest Julia set boundary point near *center*.

    Returns (z_boundary, distance) or (None, None)."""
    if log is None:
        log = lambda msg: None

    def escapes(z0, iters=None):
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

    n_rays = 64
    n_samples = 32
    search_radius = scale * 0.7
    best_pt = None
    best_dist = float('inf')

    log(f"  Searching {n_rays} rays, radius={search_radius:.6f}…\n")

    for i in range(n_rays):
        angle = 2 * math.pi * i / n_rays
        direction = cmath.exp(1j * angle)
        fracs = [j / n_samples for j in range(n_samples + 1)]
        prev_pt = center
        prev_esc = escapes(center)
        for frac in fracs[1:]:
            cur_pt = center + search_radius * frac * direction
            cur_esc = escapes(cur_pt)
            if cur_esc != prev_esc:
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
                break
            prev_pt = cur_pt
            prev_esc = cur_esc

    if best_pt is None:
        log("  ⚠ No boundary found along any ray.\n")
        return None, None

    log(f"  Ray search: nearest boundary at {best_pt:.10f}, "
        f"dist={best_dist:.2e}\n")

    refine_r = best_dist * 0.1 if best_dist > 0 else search_radius * 0.001
    grid_n = 21
    best_refined = best_pt
    best_gradient = 0
    for gi in range(grid_n):
        for gj in range(grid_n):
            dx = refine_r * (2 * gi / (grid_n - 1) - 1)
            dy = refine_r * (2 * gj / (grid_n - 1) - 1)
            pt = best_pt + complex(dx, dy)
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
            if 0 < count < max_iter:
                score = min(count, max_iter - count)
                if score > best_gradient:
                    best_gradient = score
                    best_refined = pt

    if best_gradient > 0:
        best_pt = best_refined
        best_dist = abs(best_pt - center)
        log(f"  Refined: {best_pt:.10f}, dist={best_dist:.2e}\n")

    best_periodic = None
    best_periodic_dist = float('inf')
    for period in range(1, 5):
        z0 = best_pt
        for _newton_step in range(200):
            z = z0
            dz = 1.0 + 0j
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
            jacobian = dz - 1.0
            if abs(jacobian) < 1e-30:
                break
            step = residual / jacobian
            z0 = z0 - step
            if abs(step) < 1e-14:
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
        log(f"  ✓ Using repelling periodic point: {best_periodic:.10f}\n")
        return best_periodic, best_periodic_dist
    else:
        log(f"  ✓ Using boundary point: {best_pt:.10f}\n")
        return best_pt, best_dist


def find_clean_boundary_points(c, kernel, power, zoom_factor, scale_start,
                               width, max_iter, bailout,
                               log=None, max_points=12):
    """Find precision-friendly zoom-center candidates on the Julia boundary."""
    if log is None:
        log = lambda m: None

    EPS_FLOAT = 2.22e-16

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

    def dyadic_denom_exp(x, tol=1e-10, max_n=12):
        for n in range(0, max_n + 1):
            d = 2 ** n
            k = round(x * d)
            if abs(x - k / d) <= tol:
                return n, int(k)
        return None, None

    def is_exact_dyadic(z):
        nr, _ = dyadic_denom_exp(z.real, tol=0.0)
        ni, _ = dyadic_denom_exp(z.imag, tol=0.0)
        return nr is not None and ni is not None

    def max_dyadic_denom(z):
        nr, _ = dyadic_denom_exp(z.real)
        ni, _ = dyadic_denom_exp(z.imag)
        if nr is None or ni is None:
            return None
        return max(nr, ni)

    viewport_r = max(scale_start, 2.0)
    periodic = []

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
            if mult <= 1.01:
                continue
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
            if any(abs(z - r[0]) < 1e-6 for r in roots_p):
                continue
            roots_p.append((z, p, mult))
        periodic.extend(roots_p)
        log(f"  Period {p}: {len(roots_p)} repelling fixed points\n")

    dyadic_points = []
    seen_keys = set()
    for n in range(0, 6):
        denom = 2 ** n
        K = min(int(math.ceil(viewport_r * denom)) + 1, 40)
        for k in range(-K, K + 1):
            for m in range(-K, K + 1):
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
                return last_ok
        return last_ok

    boundary_dyadic = []
    for z in dyadic_points:
        depth = verified_boundary_depth(z)
        if depth is not None and depth <= -6:
            boundary_dyadic.append((z, depth))
    log(f"  Dyadic grid: {len(dyadic_points)} tested, "
        f"{len(boundary_dyadic)} verified on boundary (contrast ≤ 1e-6)\n")

    def try_snap_periodic(z, p):
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

    INF_FRAMES = 10 ** 6
    for cand in all_cands:
        z = cand["z"]
        mag = abs(z)
        cand["magnitude"] = mag
        exact = is_exact_dyadic(z)
        cand["exact_dyadic"] = exact
        depth = cand["verified_depth"]

        if depth <= PERIODIC_DEPTH:
            cf_verified = INF_FRAMES
        else:
            try:
                lf = math.log(zoom_factor)
                cf_verified = 0 if lf >= 0 else max(
                    0, int(math.log(10 ** depth * width / scale_start) / lf))
            except (ValueError, ZeroDivisionError):
                cf_verified = 0

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

        score = min(cand["clean_frames"], 1000)
        if cand["period"] is not None:
            score += 1500 + 200 / cand["period"]
        if exact:
            score += 800
        if denom is not None and denom < 12:
            score += max(0, 10 - denom) * 8
        cand["score"] = score

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

