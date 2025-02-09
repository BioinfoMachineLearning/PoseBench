# Adapted from: https://github.com/zrqiao/NeuralPLexer

import math

import rootutils
import torch
from beartype.typing import Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils.frame_utils import RigidTransform


class GaussianFourierEncoding1D(torch.nn.Module):
    """Gaussian Fourier Encoding for 1D data."""

    def __init__(
        self,
        n_basis: int,
        eps: float = 1e-2,
    ):
        """Initialize Gaussian Fourier Encoding."""
        super().__init__()
        self.eps = eps
        self.fourier_freqs = torch.nn.Parameter(torch.randn(n_basis) * math.pi)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Forward pass of Gaussian Fourier Encoding."""
        encodings = torch.cat(
            [
                torch.sin(self.fourier_freqs.mul(x)),
                torch.cos(self.fourier_freqs.mul(x)),
            ],
            dim=-1,
        )
        return encodings


class GaussianRBFEncoding1D(torch.nn.Module):
    """Gaussian RBF Encoding for 1D data."""

    def __init__(
        self,
        n_basis: int,
        x_max: float,
        sigma: float = 1.0,
    ):
        """Initialize Gaussian RBF Encoding."""
        super().__init__()
        self.sigma = sigma
        self.rbf_centers = torch.nn.Parameter(
            torch.linspace(0, x_max, n_basis), requires_grad=False
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Forward pass of Gaussian RBF Encoding."""
        encodings = torch.exp(-((x.unsqueeze(-1) - self.rbf_centers).div(self.sigma).square()))
        return encodings


class RelativeGeometryEncoding(torch.nn.Module):
    "Compute radial basis functions and iterresidue/pseudoresidue orientations."

    def __init__(self, n_basis: int, out_dim: int, d_max: float = 20.0):
        """Initialize RelativeGeometryEncoding."""
        super().__init__()
        self.rbf_encoding = GaussianRBFEncoding1D(n_basis, d_max)
        self.rel_geom_projector = torch.nn.Linear(n_basis + 15, out_dim, bias=False)

    def forward(self, frames: RigidTransform, merged_edge_idx: Tuple[torch.Tensor, torch.Tensor]):
        """Forward pass of RelativeGeometryEncoding."""
        frame_t, frame_R = frames.t, frames.R
        pair_dists = torch.norm(
            frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]],
            dim=-1,
        )
        pair_directions_l = torch.matmul(
            (frame_t[merged_edge_idx[1]] - frame_t[merged_edge_idx[0]]).unsqueeze(-2),
            frame_R[merged_edge_idx[0]],
        ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
        pair_directions_r = torch.matmul(
            (frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]]).unsqueeze(-2),
            frame_R[merged_edge_idx[1]],
        ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
        pair_orientations = torch.matmul(
            frame_R.transpose(-2, -1).contiguous()[merged_edge_idx[0]],
            frame_R[merged_edge_idx[1]],
        )
        return self.rel_geom_projector(
            torch.cat(
                [
                    self.rbf_encoding(pair_dists),
                    pair_directions_l,
                    pair_directions_r,
                    pair_orientations.flatten(-2, -1),
                ],
                dim=-1,
            )
        )
