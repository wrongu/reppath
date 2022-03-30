import torch
import numpy as np
from repsim import Stress, GeneralizedShapeMetric, AffineInvariantRiemannian
from repsim import pairwise
from reppath.geometry.manifold import SPDMatrix, DistMatrix
from reppath.geometry.length_space import LengthSpace
from reppath.geometry.trig import angle
from reppath.geometry.geodesic import midpoint


def test_geodesic_stress():
    x, y = torch.randn(5, 3), torch.randn(5, 4)
    _test_geodesic_helper(x, y, Stress(), DistMatrix(5))


def test_geodesic_shape():
    x, y = torch.randn(5, 3), torch.randn(5, 4)
    _test_geodesic_helper(x, y, GeneralizedShapeMetric(), SPDMatrix(5))


def test_geodesic_riemann():
    x, y = torch.randn(5, 3), torch.randn(5, 4)
    _test_geodesic_helper(x, y, AffineInvariantRiemannian(), SPDMatrix(5))


def _test_geodesic_helper(x, y, metric, manifold):
    k_x, k_y = pairwise.compare(x, type=metric.compare_type),  pairwise.compare(y, type=metric.compare_type)
    space = LengthSpace(manifold, l=metric.compare_rdm)

    mid, converged = midpoint(k_x, k_y, space)
    # print("\n", k_x, k_y, mid, sep="\n")
    assert converged, f"Midpoint failed to converge using {metric}: {mid}"
    assert manifold.contains(mid), f"Midpoint failed contains() test using {metric}, {manifold}"

    ang = angle(k_x, mid, k_y, space).item()
    assert np.abs(ang - np.pi) < 1e-3, f"Angle through midpoint using {metric} should be pi but is {ang}"
