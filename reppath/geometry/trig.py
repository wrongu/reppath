import torch
from geometry.length_space import LengthSpace, Point, Scalar
from geometry.geodesic import point_along
import warnings


def angle(pt_a: Point,
          pt_b: Point,
          pt_c: Point,
          space: LengthSpace) -> Scalar:
    pt_ba, converged_ba = point_along(pt_b, pt_a, space, frac=0.02, tol=1e-6)
    pt_bc, converged_bc = point_along(pt_b, pt_c, space, frac=0.02, tol=1e-6)
    if not (converged_ba and converged_bc):
        warnings.warn("point_along failed to converge; angle may be inaccurate")
    u, v = pt_ba - pt_b, pt_bc - pt_b
    dot_cos = torch.sum(u*v) / torch.sqrt(torch.sum(u*u)*torch.sum(v*v))
    return torch.arccos(torch.clip(dot_cos, -1., +1.))


__all__ = ["angle"]