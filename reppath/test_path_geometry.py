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

    assert manifold.contains(k_x), \
        f"Manifold {manifold} does not contain k_x of type {metric.compare_type}"
    assert manifold.contains(k_y), \
        f"Manifold {manifold} does not contain k_y of type {metric.compare_type}"

    mid, converged = midpoint(k_x, k_y, space)

    assert converged, \
        f"Midpoint failed to converge using {metric}: {mid}"
    assert manifold.contains(mid), \
        f"Midpoint failed contains() test using {metric}, {manifold}"
    assert np.isclose(space.length(k_x, k_y), space.length(k_x, mid) + space.length(mid, k_y)), \
        f"Midpoint failed to subdivide the total length"
    assert np.isclose(space.length(k_x, mid), space.length(mid, k_y)), \
        f"Midpoint failed to split the total length into equal parts"

    ang = angle(k_x, mid, k_y, space).item()
    assert np.abs(ang - np.pi) < 1e-3, \
        f"Angle through midpoint using {metric} should be pi but is {ang}"
