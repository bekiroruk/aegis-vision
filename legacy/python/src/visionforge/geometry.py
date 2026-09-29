"""Framework-independent projective geometry helpers."""

from __future__ import annotations

from collections.abc import Sequence


def transform_point(
    point: tuple[float, float], homography: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """Apply a 3x3 homography to a 2D point using homogeneous coordinates."""
    if len(homography) != 3 or any(len(row) != 3 for row in homography):
        raise ValueError("Homography must be a 3x3 matrix")
    x, y = point
    denominator = homography[2][0] * x + homography[2][1] * y + homography[2][2]
    if abs(denominator) < 1e-12:
        raise ValueError("Point maps to infinity")
    tx = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denominator
    ty = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denominator
    return float(tx), float(ty)
