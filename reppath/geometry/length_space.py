import torch
from reppath.geometry.manifold import Manifold
from typing import Callable
import warnings


Point = torch.Tensor
Scalar = torch.Tensor
LengthFun = Callable[[Point, Point], Scalar]


class LengthSpace(Manifold):
    def __init__(self, m: Manifold, l: LengthFun):
        super().__init__(dim=m.dim, shape=m.shape)
        self._manifold = m
        self._length = l

    def _project(self, pt: Point) -> Point:
        return self._manifold.project(pt)

    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        return self._manifold.contains(pt, atol)

    def length(self, pt_a: Point, pt_b: Point):
        if not self.contains(pt_a):
            warnings.warn("pt_a is not on the manifold - trying to project")
            pt_a = self._project(pt_a)

        if not self.contains(pt_b):
            warnings.warn("pt_b is not on the manifold - trying to project")
            pt_b = self._project(pt_b)

        return self._length(pt_a, pt_b)


__all__ = ["Point", "Scalar", "LengthSpace"]
