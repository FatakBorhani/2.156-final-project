"""One standard form for stellarator boundaries, and a physical distance between them.

The same surface can be written with different Fourier coefficients: any overall size,
the poloidal angle started half a turn later (theta -> theta + pi) or run backwards
(theta -> -theta), or the toroidal angle started half a field period later. None of these
change the physics (every Benchmark 3 metric is dimensionless). `canonical` removes them:

  1. scale so the major radius r_cos[m=0, n=0] = 1
  2. r_cos[1, 0] > 0      (else theta -> theta + pi: flip the sign of odd-m rows)
  3. z_sin[1, 0] < 0      (else theta -> -theta: n -> -n for m >= 1, and negate z_sin)
  4. r_cos[0, 1] > 0      (else shift half a field period: multiply column n by (-1)^n)

Signs 2-4 are the convention every public NFP-3 design already uses, so canonical
coefficients stay valid simulator input.

In standard form, `distance` = RMS distance between the two surfaces (Parseval), in units
of the major radius. A typical NFP-3 design has a minor radius of ~0.1, so a distance of
0.01 means the surfaces differ by ~10% of the plasma's thickness on average.
"""

from __future__ import annotations

import json

import numpy as np

N_POL, N_TOR = 5, 9          # 4x4 modes: m = 0..4, n = -4..4
MID = (N_TOR - 1) // 2       # column index of n = 0


def _arrays(boundary) -> tuple[np.ndarray, np.ndarray, int]:
    b = json.loads(boundary) if isinstance(boundary, str) else boundary
    return np.array(b["r_cos"], float), np.array(b["z_sin"], float), int(b["n_field_periods"])


def canonical(boundary) -> tuple[np.ndarray, np.ndarray, int]:
    """(r_cos, z_sin, nfp) in standard form. Input: SurfaceRZFourier JSON (str or dict)."""
    rc, zs, nfp = _arrays(boundary)
    if rc.shape != (N_POL, N_TOR):
        raise ValueError(f"expected 4x4 modes ({N_POL}x{N_TOR} arrays), got {rc.shape}")
    s = rc[0, MID]
    rc, zs = rc / s, zs / s
    m = np.arange(N_POL)[:, None]
    n = np.arange(N_TOR)[None, :] - MID
    if rc[1, MID] < 0:                      # theta -> theta + pi
        rc, zs = rc * (-1.0) ** m, zs * (-1.0) ** m
    if zs[1, MID] > 0:                      # theta -> -theta (m = 0 rows don't depend on theta)
        rc[1:], zs[1:] = rc[1:, ::-1].copy(), -zs[1:, ::-1].copy()
    if rc[0, MID + 1] < 0:                  # half a field period later
        rc, zs = rc * (-1.0) ** n, zs * (-1.0) ** n
    return rc, zs, nfp


# Free coefficients: r_cos with m = 0 needs n >= 1 (n = 0 is the fixed major radius,
# n < 0 is redundant); z_sin with m = 0 needs n >= 1. Everything with m >= 1 is free.
_R_FREE = np.ones((N_POL, N_TOR), bool); _R_FREE[0, : MID + 1] = False
_Z_FREE = np.ones((N_POL, N_TOR), bool); _Z_FREE[0, : MID + 1] = False
N_FREE = int(_R_FREE.sum() + _Z_FREE.sum())   # 80


def to_vector(boundary) -> np.ndarray:
    """The 80 free coefficients in standard form, scaled so Euclidean distance between
    two vectors is the RMS distance between the surfaces (units of major radius)."""
    rc, zs, _ = canonical(boundary)
    return np.sqrt(0.5) * np.concatenate([rc[_R_FREE], zs[_Z_FREE]])


def from_vector(v: np.ndarray, nfp: int = 3) -> dict:
    """Inverse of to_vector: a SurfaceRZFourier dict the evaluator accepts."""
    v = np.asarray(v, float) / np.sqrt(0.5)
    rc = np.zeros((N_POL, N_TOR)); zs = np.zeros((N_POL, N_TOR))
    k = int(_R_FREE.sum())
    rc[_R_FREE], zs[_Z_FREE] = v[:k], v[k:]
    rc[0, MID] = 1.0
    return {"r_cos": rc.tolist(), "z_sin": zs.tolist(), "r_sin": None, "z_cos": None,
            "n_field_periods": nfp, "is_stellarator_symmetric": True}


def aspect_ratio(X: np.ndarray, nfp: int = 3, n_phi: int = 32, n_theta: int = 64,
                 chunk: int = 2000) -> np.ndarray:
    """Aspect ratio of each standard-form vector (rows of X), from geometry alone.

    Same definition as VMEC: minor radius = sqrt(mean cross-section area / pi), major
    radius = volume / (2 pi mean area). Matches the simulator to ~0.01 on public designs.
    """
    X = np.atleast_2d(X) / np.sqrt(0.5)
    k = int(_R_FREE.sum())
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, n_phi, endpoint=False)
    m = np.arange(N_POL)[:, None]; n = np.arange(N_TOR)[None, :] - MID
    ang = m[None, None] * th[None, :, None, None] - nfp * n[None, None] * ph[:, None, None, None]
    cos, sin = np.cos(ang), np.sin(ang)                        # (phi, theta, m, n)
    out = []
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        rc = np.zeros((len(x), N_POL, N_TOR)); zs = np.zeros_like(rc)
        rc[:, _R_FREE], zs[:, _Z_FREE] = x[:, :k], x[:, k:]
        rc[:, 0, MID] = 1.0
        R = np.einsum("imn,ptmn->ipt", rc, cos); Z = np.einsum("imn,ptmn->ipt", zs, sin)
        R2, Z2 = np.roll(R, -1, axis=2), np.roll(Z, -1, axis=2)
        area = np.abs(0.5 * np.sum(R * Z2 - R2 * Z, axis=2).mean(axis=1))
        vol = np.abs(np.sum((R**2 + R * R2 + R2**2) / 6 * (Z2 - Z), axis=2).mean(axis=1)) * 2 * np.pi
        out.append(vol / (2 * np.pi * area) / np.sqrt(area / np.pi))
    return np.concatenate(out)


def distance(a, b) -> float:
    """RMS distance between two boundaries' surfaces, in units of the major radius."""
    return float(np.linalg.norm(to_vector(a) - to_vector(b)))
