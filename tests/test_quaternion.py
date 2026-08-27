import torch

from tumbleweed.geometry.quaternion import (
    conjugate,
    inverse,
    multiply,
    normalize,
    to_rotation_matrix,
)


def _random_unit_quaternions(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return normalize(torch.randn(n, 4, generator=g))


def test_multiply_conjugate_is_identity():
    q = _random_unit_quaternions(8)
    identity = multiply(q, conjugate(q))
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(identity)
    assert torch.allclose(identity, expected, atol=1e-6)


def test_inverse_matches_conjugate_for_unit_quaternion():
    q = _random_unit_quaternions(8)
    assert torch.allclose(inverse(q), conjugate(q), atol=1e-6)


def test_multiply_composes_like_rotation_matrices():
    q1 = _random_unit_quaternions(8, seed=1)
    q2 = _random_unit_quaternions(8, seed=2)

    r1 = to_rotation_matrix(q1)
    r2 = to_rotation_matrix(q2)

    r_composed_via_matrices = r2 @ r1
    r_composed_via_quaternion = to_rotation_matrix(multiply(q2, q1))

    assert torch.allclose(r_composed_via_quaternion, r_composed_via_matrices, atol=1e-5)
