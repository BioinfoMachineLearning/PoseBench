import torch
from beartype.typing import Optional


class RigidTransform:
    """Rigid Transform class."""

    def __init__(self, t: torch.Tensor, R: Optional[torch.Tensor] = None):
        """Initialize Rigid Transform."""
        self.t = t
        if R is None:
            R = t.new_zeros(*t.shape, 3)
        self.R = R

    def __getitem__(self, key):
        """Get item from Rigid Transform."""
        return RigidTransform(self.t[key], self.R[key])

    def unsqueeze(self, dim):
        """Unsqueeze Rigid Transform."""
        return RigidTransform(self.t.unsqueeze(dim), self.R.unsqueeze(dim))

    def squeeze(self, dim):
        """Squeeze Rigid Transform."""
        return RigidTransform(self.t.squeeze(dim), self.R.squeeze(dim))

    def concatenate(self, other, dim=0):
        """Concatenate Rigid Transform."""
        return RigidTransform(
            torch.cat([self.t, other.t], dim=dim),
            torch.cat([self.R, other.R], dim=dim),
        )


def get_frame_matrix(
    ri: torch.Tensor, rj: torch.Tensor, rk: torch.Tensor, eps: float = 1e-4, strict: bool = False
):
    """Get frame matrix from three points using the regularized Gram-Schmidt algorithm.

    Note that this implementation allows for shearing.
    """
    v1 = ri - rj
    v2 = rk - rj
    if strict:
        # v1 = v1 + torch.randn_like(rj).mul(eps)
        # v2 = v2 + torch.randn_like(rj).mul(eps)
        e1 = v1 / v1.norm(dim=-1, keepdim=True)
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.norm(dim=-1, keepdim=True)
    else:
        e1 = v1 / v1.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
    e3 = torch.cross(e1, e2, dim=-1)
    # Rows - lab frame, columns - internal frame
    rot_j = torch.stack([e1, e2, e3], dim=-1)
    return RigidTransform(rj, torch.nan_to_num(rot_j, 0.0))


def cartesian_to_internal(rs: torch.Tensor, frames: RigidTransform):
    """Right-multiply the pose matrix."""
    rs_loc = rs - frames.t
    rs_loc = torch.matmul(rs_loc.unsqueeze(-2), frames.R)
    return rs_loc.squeeze(-2)


def apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Apply a similarity transform to a set of points X.

    From: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X
