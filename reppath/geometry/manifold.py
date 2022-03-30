import torch


class Manifold(object):
    def __init__(self, *, dim: int, shape: tuple):
        self.dim = dim
        self.shape = shape
        self.ambient = torch.prod(torch.tensor(shape), dim=-1)

    def project(self, pt: torch.Tensor) -> torch.Tensor:
        """Project a point from the ambient space onto the manifold.
        """
        return self._project(pt) if not self.contains(pt) else pt

    def _project(self, pt: torch.Tensor) -> torch.Tensor:
        """Project without first checking contains()
        """
        raise NotImplementedError("Subclass must implement Manifold._project()")

    def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool:
        """Check whether the given point is within 'atol' tolerance of the manifold.

        Manifold.contains checks match to ambient shape only. Further specialization done by subclasses.
        """
        return pt.size()[-self.ambient:] == self.shape


class HyperSphere(Manifold):
    def __init__(self, dim:int):
        super(HyperSphere, self).__init__(dim=dim, shape=(dim+1,))

    def _project(self, pt: torch.Tensor) -> torch.Tensor:
        return pt / torch.linalg.norm(pt, dim=-1)

    def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool:
        return torch.abs(torch.linalg.norm(pt, dim=-1) - 1.0) < atol


class Matrix(Manifold):
    def __init__(self, rows: int, cols: int):
        super(Matrix, self).__init__(dim=rows*cols, shape=(rows, cols))

    def _project(self, pt: torch.Tensor) -> torch.Tensor:
        return pt


class SymmetricMatrix(Matrix):
    """Symmetric Matrix"""
    def __init__(self, rows: int):
        super(SymmetricMatrix, self).__init__(rows, rows)
        self.dim = rows * (rows + 1) // 2

    def _project(self, pt: torch.Tensor) -> torch.Tensor:
        return (pt + pt.transpose(-2, -1)) / 2

    def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool:
        return super(SymmetricMatrix, self).contains(pt, atol) and \
            torch.allclose(pt, pt.transpose(-2, -1), atol=atol)


class SPDMatrix(SymmetricMatrix):
    """Symmetric Positive (Semi-)Definite Matrix"""
    def _project(self, pt: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        # Thanks to https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/spd_matrices.py#L59
        sym_pt = super(SPDMatrix, self)._project(pt)
        s, v = torch.linalg.eigh(sym_pt)
        s = torch.clip(s, atol, None)  # clip eigenvalues
        return torch.einsum('id,d,jd->ij', v, s, v)

    def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool:
        # Three checks: (1) is a matrix of the right (n,n) size,
        # (2) is symmetric, up to 'atol' tolerance
        # (3) has all non-negative eigenvalues
        return super(SPDMatrix, self).contains(pt) and \
               torch.all(torch.linalg.eigvalsh(pt) >= -atol)


class DistMatrix(SymmetricMatrix):
    """Pairwise-distance matrix. A symmetric matrix with all-zero diagonals and non-negative off-diagonals"""
    def _project(self, pt: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        sym_pt = super(DistMatrix, self)._project(pt)
        mask = 1. - torch.eye(pt.size()[0], device=pt.device, dtype=pt.dtype)
        return torch.clip(sym_pt*mask, 0., None)

    def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool:
        return super(DistMatrix, self).contains(pt) and \
               torch.allclose(torch.diag(pt), pt.new_zeros(pt.size()[0])) and \
               torch.all(pt >= -atol)


__all__ = ['Manifold', 'HyperSphere', 'Matrix', 'SPDMatrix', 'DistMatrix']
