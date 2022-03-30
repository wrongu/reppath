import torch
from geometry.length_space import LengthSpace, Point, Scalar
from typing import Union, Iterable, List, Tuple
import warnings


def path_length(pts: Iterable[Point],
                space: LengthSpace) -> Scalar:
    l, pt_a = Scalar([0.]), None
    for pt_b in pts:
        if pt_a is not None:
            l += space.length(pt_a, pt_b)
        pt_a = pt_b
    return l


def subdivide_geodesic(pt_a: Point,
                       pt_b: Point,
                       space: LengthSpace,
                       octaves: int = 5,
                       **kwargs) -> List[Point]:
    midpt, converged = midpoint(pt_a, pt_b, space , **kwargs)
    if not converged:
        warnings.warn(f"midpoint() failed to converge; remaining {octaves} subdivisions may be inaccurate")
    if octaves > 1 and converged:
        # Recursively subdivide each half
        left_half = subdivide_geodesic(pt_a, midpt, space, octaves-1)
        right_half = subdivide_geodesic(midpt, pt_b, space, octaves-1)
        return left_half + right_half[1:]
    else:
        # Base case
        return [pt_a, midpt, pt_b]


def point_along(pt_a: Point,
                pt_b: Point,
                space: LengthSpace,
                frac: float,
                guess: Union[Point, None] = None,
                tol: float = 1e-6,
                init_step_size: float = 0.1,
                max_iter: int = 10000) -> Tuple[Point, bool]:

    if frac < 0. or frac > 1.:
        raise ValueError(f"'frac' must be in [0, 1] but is {frac}")

    if frac == 0.:
        return pt_a, True
    elif frac == 1.:
        return pt_b, True
    elif torch.allclose(pt_a, pt_b, atol=tol):
        return space.project((pt_a+pt_b)/2), True

    # For reference, we know we're on the geodesic when dist_ap + dist_pb = dist_ab
    dist_ab = space.length(pt_a, pt_b)

    # Default initial guess to projection of euclidean interpolated point
    pt = space.project(guess) if guess is not None else space.project((1-frac)*pt_a + frac*pt_b)
    pt.requires_grad_(True)

    def calc_error(pt_):
        # Two sources of error: total length should be dist_ab, and dist_a/(dist_a+dist_b) should equal 'frac'
        dist_a, dist_b = space.length(pt_a, pt_), space.length(pt_, pt_b)
        total_length = dist_a + dist_b
        length_error = torch.clip(total_length - dist_ab, 0., None)
        frac_error = (dist_a/total_length - frac)**2
        return length_error + frac_error

    step_size = init_step_size
    for itr in range(max_iter):
        err = calc_error(pt)
        grad = torch.autograd.grad(err, pt)[0]

        # Check convergence before trying to do any updates
        if err < tol:
            return pt.detach(), True

        # Update by gradient descent + line search to reduce step size
        with torch.no_grad():
            new_pt = space.project(pt - step_size * grad)
            new_err = calc_error(new_pt)
            norm_grad = torch.linalg.norm(grad)
            while new_err > err and norm_grad * step_size > tol/2:
                step_size /= 2
                new_pt = space.project(pt - step_size * grad)
                new_err = calc_error(new_pt)
            if new_err < err:
                pt[:] = new_pt
            else:
                # If this is reached, we've halved the step_size many many times but still aren't improving the error.
                # Break and return the current value of 'pt' along with converged=False
                return pt.detach(), False

        # Prep next loop
        itr = itr + 1

    # Max iterations reached – return final value of 'pt' along with converged=True
    return pt.detach(), False


def midpoint(pt_a: Point,
             pt_b: Point,
             space: LengthSpace,
             **kwargs) -> Tuple[Point, bool]:
    return point_along(pt_a, pt_b, space, frac=0.5, **kwargs)


__all__ = ["path_length", "subdivide_geodesic", "point_along", "midpoint"]
