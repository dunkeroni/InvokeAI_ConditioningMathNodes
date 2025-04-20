import torch
from torch.linalg import norm


def perp(a, b):
    """
    The perpendicular component of a vector `a`, relative to vector `b`.

    :param a: Input vector to be resolved into a perpendicular component
    :param b: Vector relative to which the perpendicular component of `a` is calculated
    :return: Perpendicular component of vector `a` relative to vector `b`
    """
    return a - (torch.mul(a, b).sum() / (torch.norm(b) ** 2)) * b


def proj(a, b):
    """
    Projects vector `a` onto vector `b`.

    :param a: The vector that is being projected.
    :param b: The vector onto which `a` is being projected.
    :return: The projection of vector `a` onto vector `b`.
    """
    return (torch.mul(a, b).sum() / (torch.norm(b) ** 2)) * b


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, *, no_NaN = True, DOT_THRESHOLD: float = 0.9995):
    """Spherical linear interpolation.

    :param v0: The starting vector
    :param v1: The target vector
    :param t: The interpolation factor, where 0 represents `v0` and 1 represents `v1`
    :return: interpolation vector between `v0` and `v1`.
    """
    # blend_latents.slerp exists, but it uses numpy which doesn't work with bfloa16.
    # Instead, use birch-san's implementation from https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm = norm(v0, dim=-1)
    v1_norm = norm(v1, dim=-1)

    v0_normed = v0 / v0_norm.unsqueeze(-1)
    v1_normed = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot = (v0_normed * v1_normed).sum(-1)
    dot_mag = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If the absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, torch.Tensor) else 0
    t_batch_dims: torch.Size = (
        t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
    )
    out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped = torch.lerp(v0, v1, t)

        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        # Calculate the initial angle between v0 and v1
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = theta_t.sin()
        # Finish the slerp algorithm
        s0 = (theta_0 - theta_t).sin() / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        slerped = s0 * v0 + s1 * v1

        out = slerped.where(can_slerp.unsqueeze(-1), out)
        if no_NaN:
            out = out.nan_to_num_()

    return out


functions = dict(perp=perp, proj=proj, slerp=slerp)
