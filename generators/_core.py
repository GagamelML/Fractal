"""
Shared escape-iteration logic for Julia-type fractals.
The kernel is swappable: polynomial (z^p+c), sin, cos, etc.
"""
import numpy as np


def iterate_escape(z, c_val, max_iter, power=2, bailout=2.0, kernel="poly"):
    """
    Vectorised escape-time iteration.

    Parameters
    ----------
    z        : ndarray[complex]  – starting grid
    c_val    : complex           – Julia constant
    max_iter : int               – iteration cap
    power    : int/float         – exponent (only used for kernel="poly")
    bailout  : float             – escape radius (|z| for poly, |z.imag| for sin/cos)
    kernel   : str               – "poly" (z^p+c), "sin" (c·sin(z)), "cos" (c·cos(z))

    Returns
    -------
    counts : ndarray[int16] – iteration count per pixel (max_iter = interior)
    """
    counts = np.full(z.shape, max_iter, dtype=np.int16)
    mask = np.ones(z.shape, dtype=bool)

    if kernel == "poly":
        for i in range(max_iter):
            z[mask] = z[mask] ** power + c_val
            escaped = mask & (np.abs(z) > bailout)
            counts[escaped] = i
            mask[escaped] = False
            if not mask.any():
                break
    elif kernel == "sin":
        for i in range(max_iter):
            z[mask] = c_val * np.sin(z[mask])
            escaped = mask & (np.abs(z.imag) > bailout)
            counts[escaped] = i
            mask[escaped] = False
            if not mask.any():
                break
    elif kernel == "cos":
        for i in range(max_iter):
            z[mask] = c_val * np.cos(z[mask])
            escaped = mask & (np.abs(z.imag) > bailout)
            counts[escaped] = i
            mask[escaped] = False
            if not mask.any():
                break
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    return counts

