import random

import rootutils
import torch
import torch.nn.functional as F
from beartype.typing import Any, Dict, Literal, Optional, Tuple, Union
from lightning import LightningModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils.frame_utils import cartesian_to_internal, get_frame_matrix
from flowdock.utils.metric_utils import compute_per_atom_lddt
from flowdock.utils.model_utils import (
    distance_to_gaussian_contact_logits,
    distogram_to_gaussian_contact_logits,
    eval_true_contact_maps,
    sample_res_rowmask_from_contacts,
    sample_reslig_contact_matrix,
    segment_mean,
)

MODEL_BATCH = Dict[str, Any]
MODEL_STAGE = Literal["train", "val", "test", "predict"]
LOSS_MODES = Literal[
    "structure_prediction",
    "auxiliary_estimation",
    "auxiliary_estimation_without_structure_prediction",
]


def compute_contact_prediction_losses(
    pred_distograms: torch.Tensor,
    ref_dist_mat: torch.Tensor,
    dist_bins: torch.Tensor,
    contact_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the contact prediction losses for a given batch.

    :param pred_distograms: The predicted distograms.
    :param ref_dist_mat: The reference distance matrix.
    :param dist_bins: The distance bins.
    :param contact_scale: The contact scale.
    :return: The distogram and forward KL losses.
    """
    # True onehot distance and distogram loss
    distance_bin_idx = torch.bucketize(ref_dist_mat, dist_bins[:-1], right=True)
    distogram_loss = F.cross_entropy(pred_distograms.flatten(0, -2), distance_bin_idx.flatten())
    # Evaluate contact logits via log(\sum_j p_j \exp(-\alpha*r_j^2))
    ref_contact_logits = distance_to_gaussian_contact_logits(ref_dist_mat, contact_scale)
    pred_contact_logits = distogram_to_gaussian_contact_logits(
        pred_distograms,
        dist_bins,
        contact_scale,
    )
    forward_kl_loss = F.kl_div(
        F.log_softmax(
            pred_contact_logits.flatten(-2, -1),
            dim=-1,
        ),
        F.log_softmax(
            ref_contact_logits.flatten(-2, -1),
            dim=-1,
        ),
        log_target=True,
        reduction="batchmean",
    )
    return distogram_loss, forward_kl_loss


def compute_protein_distogram_loss(
    batch: MODEL_BATCH,
    target_coords: torch.Tensor,
    dist_bins: torch.Tensor,
    dgram_head: torch.nn.Module,
    entry: str = "res_res_grid_attr_flat",
) -> torch.Tensor:
    """Compute the protein distogram loss for a given batch.

    :param batch: A batch dictionary.
    :param target_coords: The target coordinates.
    :param dist_bins: The distance bins.
    :param dgram_head: The distogram head to use for loss calculation.
    :param entry: The entry to use.
    :return: The distogram loss.
    """
    n_protein_patches = batch["metadata"]["n_prot_patches_per_sample"]
    sampled_grid_features = batch["features"][entry]
    sampled_ca_coords = target_coords[batch["indexer"]["gather_idx_pid_a"]].view(
        batch["metadata"]["num_structid"], n_protein_patches, 3
    )
    sampled_ca_dist = torch.norm(
        sampled_ca_coords[:, :, None] - sampled_ca_coords[:, None, :], dim=-1
    )
    # Using AF2 parameters
    distance_bin_idx = torch.bucketize(sampled_ca_dist, dist_bins[:-1], right=True)
    distogram_loss = F.cross_entropy(dgram_head(sampled_grid_features), distance_bin_idx.flatten())
    return distogram_loss


def compute_fape_from_atom37(
    batch: MODEL_BATCH,
    device: Union[str, torch.device],
    pred_prot_coords: torch.Tensor,  # [N_res, 37, 3]
    target_prot_coords: torch.Tensor,  # [N_res, 37, 3]
    pred_lig_coords: Optional[torch.Tensor] = None,  # [N_atom, 3]
    target_lig_coords: Optional[torch.Tensor] = None,  # [N_atom, 3]
    lig_frame_atm_idx: Optional[torch.Tensor] = None,  # [3, N_atom]
    split_pl_views: bool = False,
    cap_size: int = 8000,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the Frame Aligned Point Error (FAPE) loss from `atom37` coordinates.

    :param batch: A batch dictionary.
    :param device: The device to use.
    :param pred_prot_coords: The predicted protein coordinates.
    :param target_prot_coords: The target protein coordinates.
    :param pred_lig_coords: The predicted ligand coordinates.
    :param target_lig_coords: The target ligand coordinates.
    :param lig_frame_atm_idx: The ligand frame atom indices.
    :param split_pl_views: Whether to split the protein-ligand views.
    :param cap_size: The capped size.
    :return: The FAPE loss.
    """
    features = batch["features"]
    batch_size = batch["metadata"]["num_structid"]
    with torch.no_grad():
        atom_mask = (
            features["res_atom_mask"].bool().view(batch["metadata"]["num_structid"], -1, 37)
        ).clone()
        atom_mask[:, :, [6, 7, 12, 13, 16, 17, 20, 21, 26, 27, 29, 30]] = False
    pred_prot_coords = pred_prot_coords.view(batch_size, -1, 37, 3)
    target_prot_coords = target_prot_coords.view(batch_size, -1, 37, 3)
    pred_bb_frames = get_frame_matrix(
        pred_prot_coords[:, :, 0, :],
        pred_prot_coords[:, :, 1, :],
        pred_prot_coords[:, :, 2, :],
    )
    # pred_bb_frames.R = pred_bb_frames.R.detach()
    target_bb_frames = get_frame_matrix(
        target_prot_coords[:, :, 0, :],
        target_prot_coords[:, :, 1, :],
        target_prot_coords[:, :, 2, :],
    )
    pred_prot_coords_flat = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
    target_prot_coords_flat = target_prot_coords[atom_mask].view(batch_size, -1, 3)
    if pred_lig_coords is not None:
        assert target_prot_coords is not None, "Target protein coordinates must be provided."
        assert lig_frame_atm_idx is not None, "Ligand frame atom indices must be provided."
        pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
        target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
        pred_coords = torch.cat([pred_prot_coords_flat, pred_lig_coords], dim=1)
        target_coords = torch.cat([target_prot_coords_flat, target_lig_coords], dim=1)
        pred_lig_frames = get_frame_matrix(
            pred_lig_coords[:, lig_frame_atm_idx[0]],
            pred_lig_coords[:, lig_frame_atm_idx[1]],
            pred_lig_coords[:, lig_frame_atm_idx[2]],
        )
        pred_frames = pred_bb_frames.concatenate(pred_lig_frames, dim=1)
        target_lig_frames = get_frame_matrix(
            target_lig_coords[:, lig_frame_atm_idx[0]],
            target_lig_coords[:, lig_frame_atm_idx[1]],
            target_lig_coords[:, lig_frame_atm_idx[2]],
        )
        target_frames = target_bb_frames.concatenate(target_lig_frames, dim=1)
    else:
        pred_coords = pred_prot_coords_flat
        target_coords = target_prot_coords_flat
        pred_frames = pred_bb_frames
        target_frames = target_bb_frames
    # Columns-frames, rows-points
    # [B, 1, N, 3] - [B, F, 1, 3]
    sampling_rate = cap_size / (batch_size * target_coords.shape[1])
    sampling_mask = torch.rand(target_coords.shape[1], device=device) < sampling_rate
    aligned_pred_points = cartesian_to_internal(
        pred_coords[:, sampling_mask].unsqueeze(1), pred_frames.unsqueeze(2)
    )
    with torch.no_grad():
        aligned_target_points = cartesian_to_internal(
            target_coords[:, sampling_mask].unsqueeze(1), target_frames.unsqueeze(2)
        )
    pair_dist_aligned = (
        torch.square(aligned_pred_points - aligned_target_points)
        .sum(-1)
        .add(1e-4)
        .sqrt()
        .sub(1e-2)
    )
    cropped_pair_dists = torch.clamp(pair_dist_aligned, max=10)
    normalized_pair_dists = (
        pair_dist_aligned / aligned_target_points.square().sum(-1).add(1e-4).sqrt()
    )
    if split_pl_views:
        fape_protframe = cropped_pair_dists[:, : target_bb_frames.t.shape[1]].mean((1, 2)) / 10
        fape_ligframe = cropped_pair_dists[:, target_bb_frames.t.shape[1] :].mean((1, 2)) / 10
        return fape_protframe, fape_ligframe, normalized_pair_dists.mean((1, 2))
    return cropped_pair_dists.mean((1, 2)) / 10, normalized_pair_dists.mean((1, 2))


def compute_aa_distance_geometry_loss(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the amino acid distance geometry loss for a given batch.

    :param batch: A batch dictionary.
    :param pred_coords: The predicted coordinates.
    :param target_coords: The target coordinates.
    :return: The distance geometry loss.
    """
    batch_size = batch["metadata"]["num_structid"]
    features = batch["features"]
    atom_mask = features["res_atom_mask"].bool()
    # Add backbone atoms from previous residue
    atom_mask = atom_mask.view(batch_size, -1, 37)
    atom_mask = torch.cat([atom_mask[:, 1:], atom_mask[:, :-1, 0:3]], dim=2).flatten(0, 1)
    pred_coords = pred_coords.view(batch_size, -1, 37, 3)
    pred_coords = torch.cat([pred_coords[:, 1:], pred_coords[:, :-1, 0:3]], dim=2).flatten(0, 1)
    target_coords = target_coords.view(batch_size, -1, 37, 3)
    target_coords = torch.cat([target_coords[:, 1:], target_coords[:, :-1, 0:3]], dim=2).flatten(
        0, 1
    )
    local_pair_dist_target = (
        (target_coords[:, None, :] - target_coords[:, :, None]).square().sum(-1).add(1e-4).sqrt()
    )
    local_pair_dist_pred = (
        (pred_coords[:, None, :] - pred_coords[:, :, None]).square().sum(-1).add(1e-4).sqrt()
    )
    local_pair_mask = (
        atom_mask[:, None, :] & atom_mask[:, :, None] & (local_pair_dist_target < 3.0)
    )
    ret = (local_pair_dist_target - local_pair_dist_pred).abs()[local_pair_mask]
    return ret.view(batch["metadata"]["num_structid"], -1).mean(dim=1)


def compute_sm_distance_geometry_loss(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the small molecule distance geometry loss for a given batch.

    :param batch: A batch dictionary.
    :param pred_coords: The predicted coordinates.
    :param target_coords: The target coordinates.
    :return: The distance geometry loss.
    """
    batch_size = batch["metadata"]["num_structid"]
    pred_coords = pred_coords.view(batch_size, -1, 3)
    target_coords = target_coords.view(batch_size, -1, 3)
    pair_dist_target = (
        (target_coords[:, None, :] - target_coords[:, :, None]).square().sum(-1).add(1e-4).sqrt()
    )
    pair_dist_pred = (
        (pred_coords[:, None, :] - pred_coords[:, :, None]).square().sum(-1).add(1e-4).sqrt()
    )
    local_pair_mask = pair_dist_target < 3.0
    ret = (pair_dist_target - pair_dist_pred).abs()[local_pair_mask]
    return ret.view(batch_size, -1).mean(dim=1)


def compute_drmsd_and_clashloss(
    batch: MODEL_BATCH,
    device: Union[str, torch.device],
    pred_prot_coords: torch.Tensor,
    target_prot_coords: torch.Tensor,
    atnum2vdw_uff: torch.nn.Parameter,
    cap_size: int = 4000,
    pred_lig_coords: Optional[torch.Tensor] = None,
    target_lig_coords: Optional[torch.Tensor] = None,
    ligatm_types: Optional[torch.Tensor] = None,
    binding_site: bool = False,
    pl_interface: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute the differentiable root-mean-square deviation (dRMSD) and optional clash loss for a
    given batch.

    :param batch: A batch dictionary.
    :param device: The device to use.
    :param pred_prot_coords: The predicted protein coordinates.
    :param target_prot_coords: The target protein coordinates.
    :param atnum2vdw_uff: The atomic number to UFF VDW parameters mapping `Parameter`.
    :param cap_size: The capped size.
    :param pred_lig_coords: The predicted ligand coordinates.
    :param target_lig_coords: The target ligand coordinates.
    :param ligatm_types: The ligand atom types.
    :param binding_site: Whether to compute the binding site.
    :param pl_interface: Whether to compute the protein-ligand interface.
    :return: The dRMSD and optional clash loss.
    """
    features = batch["features"]
    with torch.no_grad():
        if not binding_site:
            atom_mask = features["res_atom_mask"].bool().clone()
        else:
            atom_mask = (
                features["res_atom_mask"].bool() & features["binding_site_mask_clean"][:, None]
            )
    if pl_interface:
        # Removing ambiguous atoms
        atom_mask[:, [6, 7, 12, 13, 16, 17, 20, 21, 26, 27, 29, 30]] = False

    batch_size = batch["metadata"]["num_structid"]
    pred_prot_coords = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
    if pred_lig_coords is not None:
        assert target_prot_coords is not None, "Target protein coordinates must be provided."
        assert ligatm_types is not None, "Ligand atom types must be provided."
        pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
        pred_coords = torch.cat([pred_prot_coords, pred_lig_coords], dim=1)
    else:
        pred_coords = pred_prot_coords
    sampling_rate = cap_size / pred_coords.shape[1]
    sampling_mask = torch.rand(pred_coords.shape[1], device=device) < sampling_rate
    pred_coords = pred_coords[:, sampling_mask]
    if pl_interface:
        pred_dist = (
            torch.square(pred_coords[:, :, None] - pred_lig_coords[:, None, :])
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
    else:
        pred_dist = (
            torch.square(pred_coords[:, :, None] - pred_coords[:, None, :])
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
    with torch.no_grad():
        target_prot_coords = target_prot_coords[atom_mask].view(batch_size, -1, 3)
        if pred_lig_coords is not None:
            target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
            target_coords = torch.cat([target_prot_coords, target_lig_coords], dim=1)
        else:
            target_coords = target_prot_coords
        target_coords = target_coords[:, sampling_mask]
        if pl_interface:
            target_dist = (
                torch.square(target_coords[:, :, None] - target_lig_coords[:, None, :])
                .sum(-1)
                .add(1e-4)
                .sqrt()
                .sub(1e-2)
            )
        else:
            target_dist = (
                torch.square(target_coords[:, :, None] - target_coords[:, None, :])
                .sum(-1)
                .add(1e-4)
                .sqrt()
                .sub(1e-2)
            )
        # In Angstrom, using UFF params to compute clash loss
        protatm_types = features["res_atom_types"].long()[atom_mask]
        protatm_vdw = atnum2vdw_uff[protatm_types].view(batch_size, -1)
        if pred_lig_coords is not None:
            ligatm_vdw = atnum2vdw_uff[ligatm_types].view(batch_size, -1)
            atm_vdw = torch.cat([protatm_vdw, ligatm_vdw], dim=1)
        else:
            atm_vdw = protatm_vdw
        atm_vdw = atm_vdw[:, sampling_mask]
        average_vdw = (atm_vdw[:, :, None] + atm_vdw[:, None, :]) / 2
        # Use conservative cutoffs to avoid mis-penalization

    dist_errors = (pred_dist - target_dist).square()
    drmsd = dist_errors.add(1e-2).sqrt().sub(1e-1).mean(dim=(1, 2))
    if pl_interface:
        return drmsd, None

    covalent_like = target_dist < (average_vdw * 1.2)
    # Alphafold supplementary Eq. 46, modified
    clash_pairwise = torch.clamp(average_vdw * 1.1 - pred_dist.add(1e-6), min=0.0)
    clash_loss = clash_pairwise.mul(~covalent_like).sum(dim=2).mean(dim=1)
    return drmsd, clash_loss


def compute_template_weighted_centroid_drmsd(
    batch: MODEL_BATCH,
    pred_prot_coords: torch.Tensor,
) -> torch.Tensor:
    """Compute the template-weighted centroid dRMSD for a given batch.

    :param batch: A batch dictionary.
    :param pred_prot_coords: The predicted protein coordinates.
    :return: The dRMSD.
    """
    batch_size = batch["metadata"]["num_structid"]

    pred_cent_coords = (
        pred_prot_coords.mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
        .sum(dim=1)
        .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
    ).view(batch_size, -1, 3)
    pred_dist = (
        torch.square(pred_cent_coords[:, :, None] - pred_cent_coords[:, None, :])
        .sum(-1)
        .add(1e-4)
        .sqrt()
        .sub(1e-2)
    )
    with torch.no_grad():
        target_cent_coords = (
            batch["features"]["res_atom_positions"]
            .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
        ).view(batch_size, -1, 3)
        template_cent_coords = (
            batch["features"]["apo_res_atom_positions"]
            .mul(batch["features"]["apo_res_atom_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(batch["features"]["apo_res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
        ).view(batch_size, -1, 3)
        target_dist = (
            torch.square(target_cent_coords[:, :, None] - target_cent_coords[:, None, :])
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        template_dist = (
            torch.square(template_cent_coords[:, :, None] - template_cent_coords[:, None, :])
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        template_alignment_mask = (
            batch["features"]["apo_res_alignment_mask"].bool().view(batch_size, -1)
        )
        motion_mask = (
            ((target_dist - template_dist).abs() > 2.0)
            * template_alignment_mask[:, None, :]
            * template_alignment_mask[:, :, None]
        )

    dist_errors = (pred_dist - target_dist).square()
    drmsd = (dist_errors.add(1e-4).sqrt().sub(1e-2).mul(motion_mask).sum(dim=(1, 2))) / (
        motion_mask.long().sum(dim=(1, 2)) + 1
    )
    return drmsd


def compute_TMscore_lbound(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the TM-score lower bound for a given batch.

    :param batch: A batch dictionary.
    :param pred_coords: The predicted coordinates.
    :param target_coords: The target coordinates.
    :return: The TM-score lower bound.
    """
    features = batch["features"]
    atom_mask = features["res_atom_mask"].bool().view(batch["metadata"]["num_structid"], -1, 37)
    pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
    target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
    pred_bb_frames = get_frame_matrix(
        pred_coords[:, :, 0, :],
        pred_coords[:, :, 1, :],
        pred_coords[:, :, 2, :],
        strict=True,
    )
    target_bb_frames = get_frame_matrix(
        target_coords[:, :, 0, :],
        target_coords[:, :, 1, :],
        target_coords[:, :, 2, :],
        strict=True,
    )
    pred_coords_flat = pred_coords[atom_mask].view(batch["metadata"]["num_structid"], -1, 3)
    target_coords_flat = target_coords[atom_mask].view(batch["metadata"]["num_structid"], -1, 3)
    # Columns-frames, rows-points
    # [B, 1, N, 3] - [B, F, 1, 3]
    aligned_pred_points = cartesian_to_internal(
        pred_coords_flat.unsqueeze(1), pred_bb_frames.unsqueeze(2)
    )
    with torch.no_grad():
        aligned_target_points = cartesian_to_internal(
            target_coords_flat.unsqueeze(1), target_bb_frames.unsqueeze(2)
        )
    pair_dist_aligned = (aligned_pred_points - aligned_target_points).norm(dim=-1)
    tm_normalizer = 1.24 * (max(target_coords.shape[1], 19) - 15) ** (1 / 3) - 1.8
    per_frame_tm = torch.mean(1 / (1 + (pair_dist_aligned / tm_normalizer) ** 2), dim=2)
    return torch.amax(per_frame_tm, dim=1)


def compute_TMscore_raw(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the raw TM-score for a given batch.

    :param batch: A batch dictionary.
    :param pred_coords: The predicted coordinates.
    :param target_coords: The target coordinates.
    :return: The raw TM-score.
    """
    pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 3)
    target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 3)
    pair_dist_aligned = (pred_coords - target_coords).norm(dim=-1)
    tm_normalizer = 1.24 * (max(target_coords.shape[1], 19) - 15) ** (1 / 3) - 1.8
    per_struct_tm = torch.mean(1 / (1 + (pair_dist_aligned / tm_normalizer) ** 2), dim=1)
    return per_struct_tm


def compute_lddt_ca(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the local distance difference test (lDDT) for C-alpha atoms for a given batch.

    :param batch: A batch dictionary.
    :param pred_coords: The predicted coordinates.
    :param target_coords: The target coordinates.
    :return: The lDDT for C-alpha atoms.
    """
    pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
    target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
    pred_ca_flat = pred_coords[:, :, 1]
    target_ca_flat = target_coords[:, :, 1]
    target_dist = (target_ca_flat[:, :, None] - target_ca_flat[:, None, :]).norm(dim=-1)
    pred_dist = (pred_ca_flat[:, :, None] - pred_ca_flat[:, None, :]).norm(dim=-1)
    conserved_mask = target_dist < 15.0
    lddt = 0
    for threshold in [0.5, 1, 2, 4]:
        below_threshold = (pred_dist - target_dist).abs() < threshold
        lddt = lddt + below_threshold.mul(conserved_mask).sum((1, 2)) / conserved_mask.sum((1, 2))
    return lddt / 4


def compute_lddt_pli(
    batch: MODEL_BATCH,
    pred_prot_coords: torch.Tensor,
    target_prot_coords: torch.Tensor,
    pred_lig_coords: torch.Tensor,
    target_lig_coords: torch.Tensor,
) -> torch.Tensor:
    """Compute the local distance difference test (lDDT) for protein-ligand interface atoms for a
    given batch.

    :param batch: A batch dictionary.
    :param pred_prot_coords: The predicted protein coordinates.
    :param target_prot_coords: The target protein coordinates.
    :param pred_lig_coords: The predicted ligand coordinates.
    :param target_lig_coords: The target ligand coordinates.
    :return: The lDDT for protein-ligand interface atoms.
    """
    features = batch["features"]
    batch_size = batch["metadata"]["num_structid"]
    atom_mask = features["res_atom_mask"].bool()
    pred_prot_coords = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
    target_prot_coords = target_prot_coords[atom_mask].view(batch_size, -1, 3)
    pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
    target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
    target_dist = (target_prot_coords[:, :, None] - target_lig_coords[:, None, :]).norm(dim=-1)
    pred_dist = (pred_prot_coords[:, :, None] - pred_lig_coords[:, None, :]).norm(dim=-1)
    conserved_mask = target_dist < 6.0
    lddt = 0
    for threshold in [0.5, 1, 2, 4]:
        below_threshold = (pred_dist - target_dist).abs() < threshold
        lddt = lddt + below_threshold.mul(conserved_mask).sum((1, 2)) / conserved_mask.sum((1, 2))
    return lddt / 4


def eval_structure_prediction_losses(
    lit_module: LightningModule,
    batch: MODEL_BATCH,
    batch_idx: int,
    device: Union[str, torch.device],
    stage: MODEL_STAGE,
    t_1: float = 1.0,
) -> MODEL_BATCH:
    """Evaluate the structure prediction losses for a given batch.

    :param lit_module: The LightningModule object to reference.
    :param batch: A batch dictionary.
    :param batch_idx: The batch index.
    :param device: The device to use.
    :param stage: The stage of the training.
    :param t_1: The final timestep in the range [0, 1].
    :return: Batch dictionary with losses.
    """
    assert 0 <= t_1 <= 1, "`t_1` must be in the range `[0, 1]`."
    batch_size = batch["metadata"]["num_structid"]
    max(batch["metadata"]["num_a_per_sample"])

    if "num_molid" in batch["metadata"].keys() and batch["metadata"]["num_molid"] > 0:
        batch["misc"]["protein_only"] = False
    else:
        batch["misc"]["protein_only"] = True

    if "augmented_coordinates" in batch["features"].keys():
        batch["features"]["sdf_coordinates"] = batch["features"]["augmented_coordinates"]
        is_native_sample = 0
    else:
        is_native_sample = 1

    # Sample the timestep for each structure
    t = torch.rand((batch_size, 1), device=device)

    prior_training = int(random.randint(0, 10) == 1)  # nosec
    if prior_training == 1:
        t = torch.full_like(t, t_1)

    if lit_module.training and lit_module.hparams.cfg.task.use_template:
        use_template = bool(random.randint(0, 1))  # nosec
    else:
        use_template = lit_module.hparams.cfg.task.use_template

    lit_module.net.assign_timestep_encodings(batch, t)
    features = batch["features"]
    indexer = batch["indexer"]
    metadata = batch["metadata"]

    loss = 0
    forward_lat_converter = lit_module.net.resolve_latent_converter(
        [
            ("features", "res_atom_positions"),
            ("features", "input_protein_coords"),
        ],
        [("features", "sdf_coordinates"), ("features", "input_ligand_coords")],
    )
    batch = lit_module.net.prepare_protein_patch_indexers(batch)
    if not batch["misc"]["protein_only"]:
        max(metadata["num_i_per_sample"])

        # Evaluate the contact map
        ref_dist_mat, contact_logit_matrix = eval_true_contact_maps(
            batch, lit_module.net.CONTACT_SCALE
        )
        num_cont_to_sample = max(metadata["num_I_per_sample"])
        sampled_block_contacts = [
            None,
        ]
        # Onehot contact code sampling
        with torch.no_grad():
            for _ in range(num_cont_to_sample):
                sampled_block_contacts.append(
                    sample_reslig_contact_matrix(
                        batch, contact_logit_matrix, last=sampled_block_contacts[-1]
                    )
                )
        forward_lat_converter.lig_res_anchor_mask = sample_res_rowmask_from_contacts(
            batch,
            contact_logit_matrix,
            lit_module.hparams.cfg.task.single_protein_batch,
        )
        with torch.no_grad():
            batch = lit_module.net.forward_interp_plcomplex_latinp(
                batch, t[:, :, None], forward_lat_converter
            )
        if prior_training == 1:
            iter_id = random.randint(0, num_cont_to_sample)  # nosec
        else:
            iter_id = num_cont_to_sample
        batch = lit_module.forward(
            batch, contact_prediction=False, score=False, use_template=use_template
        )
        batch = lit_module.net.run_contact_map_stack(
            batch,
            iter_id=iter_id,
            observed_block_contacts=sampled_block_contacts[iter_id],
        )
        pred_distogram = batch["outputs"][f"res_lig_distogram_out_{iter_id}"]
        (
            pl_distogram_loss,
            pl_contact_loss_forward,
        ) = compute_contact_prediction_losses(
            pred_distogram, ref_dist_mat, lit_module.net.dist_bins, lit_module.net.CONTACT_SCALE
        )
        cont_loss = 0
        cont_loss = (
            cont_loss
            + pl_distogram_loss
            * lit_module.hparams.cfg.task.contact_loss_weight
            * is_native_sample
        )
        cont_loss = (
            cont_loss
            + pl_contact_loss_forward
            * lit_module.hparams.cfg.task.contact_loss_weight
            * is_native_sample
        )
        lit_module.log(
            f"{stage}_contact/contact_loss_distogram",
            pl_distogram_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_contact/contact_loss_forwardKL",
            pl_contact_loss_forward.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        if lit_module.hparams.cfg.task.freeze_contact_predictor:
            # Keep the contact prediction parameters in the computational graph but with zero gradients
            cont_loss *= 0.0
    else:
        with torch.no_grad():
            batch = lit_module.net.forward_interp_plcomplex_latinp(
                batch, t[:, :, None], forward_lat_converter
            )
        iter_id = 0
        batch = lit_module.forward(
            batch,
            iter_id=0,
            contact_prediction=True,
            score=False,
            use_template=use_template,
        )
    protein_distogram_loss = compute_protein_distogram_loss(
        batch,
        batch["features"]["res_atom_positions"][:, 1],
        lit_module.net.dist_bins,
        lit_module.net.dgram_head,
        entry=f"res_res_grid_attr_flat_out_{iter_id}",
    )
    lit_module.log(
        f"{stage}_contact/prot_distogram_loss",
        protein_distogram_loss.detach(),
        on_epoch=True,
        batch_size=batch_size,
    )
    if lit_module.hparams.cfg.task.freeze_contact_predictor:
        # Keep the distogram prediction parameters in the computational graph but with zero gradients
        protein_distogram_loss *= 0.0

    # NOTE: we keep the loss weighting time-independent since `sigma=1` for all prior distributions (where relevant)
    lambda_weighting = t.new_ones(batch_size)

    # Run score head and evaluate structure prediction losses
    res_atom_mask = features["res_atom_mask"].bool()

    scores = lit_module.net.run_score_head(batch, embedding_iter_id=iter_id)

    if lit_module.training:
        # # Sigmoid scaling
        # violation_loss_ratio = 1 / (
        #     1
        #     + math.exp(10 - 12 * lit_module.current_epoch / lit_module.trainer.max_epochs)
        # )
        # violation_loss_ratio = (lit_module.current_epoch / lit_module.trainer.max_epochs)
        violation_loss_ratio = 1.0
    else:
        violation_loss_ratio = 1.0

    if not batch["misc"]["protein_only"]:
        if "binding_site_mask_clean" not in batch["features"]:
            with torch.no_grad():
                min_lig_res_dist_clean = (
                    (
                        batch["features"]["res_atom_positions"][:, 1].view(batch_size, -1, 3)[
                            :, :, None
                        ]
                        - batch["features"]["sdf_coordinates"].view(batch_size, -1, 3)[:, None, :]
                    )
                    .norm(dim=-1)
                    .amin(dim=2)
                ).flatten(0, 1)
                binding_site_mask_clean = (
                    min_lig_res_dist_clean < lit_module.net.BINDING_SITE_CUTOFF
                )
            batch["features"]["binding_site_mask_clean"] = binding_site_mask_clean
        coords_pred_prot = scores["final_coords_prot_atom_padded"][res_atom_mask].view(
            batch_size, -1, 3
        )
        coords_ref_prot = batch["features"]["res_atom_positions"][res_atom_mask].view(
            batch_size, -1, 3
        )
        coords_pred_bs_prot = scores["final_coords_prot_atom_padded"][
            res_atom_mask & batch["features"]["binding_site_mask_clean"][:, None]
        ].view(batch_size, -1, 3)
        coords_ref_bs_prot = batch["features"]["res_atom_positions"][
            res_atom_mask & batch["features"]["binding_site_mask_clean"][:, None]
        ].view(batch_size, -1, 3)
        coords_pred_lig = scores["final_coords_lig_atom"].view(batch_size, -1, 3)
        coords_ref_lig = batch["features"]["sdf_coordinates"].view(batch_size, -1, 3)
        coords_pred = torch.cat([coords_pred_prot, coords_pred_lig], dim=1)
        coords_ref = torch.cat([coords_ref_prot, coords_ref_lig], dim=1)
        coords_pred_bs = torch.cat([coords_pred_bs_prot, coords_pred_lig], dim=1)
        coords_ref_bs = torch.cat([coords_ref_bs_prot, coords_ref_lig], dim=1)
        n_I_per_sample = max(metadata["num_I_per_sample"])
        lig_frame_atm_idx = torch.stack(
            [
                indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
                indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
                indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
            ],
            dim=0,
        )
        (
            global_fape_pview,
            global_fape_lview,
            normalized_fape,
        ) = compute_fape_from_atom37(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
            pred_lig_coords=scores["final_coords_lig_atom"],
            target_lig_coords=batch["features"]["sdf_coordinates"],
            lig_frame_atm_idx=lig_frame_atm_idx,
            split_pl_views=True,
        )
        aa_distgeom_error = compute_aa_distance_geometry_loss(
            batch,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
        )
        lig_distgeom_error = compute_sm_distance_geometry_loss(
            batch,
            scores["final_coords_lig_atom"],
            batch["features"]["sdf_coordinates"],
        )
        glob_drmsd, _ = compute_drmsd_and_clashloss(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
            lit_module.net.atnum2vdw_uff,
            pred_lig_coords=scores["final_coords_lig_atom"],
            target_lig_coords=batch["features"]["sdf_coordinates"],
            ligatm_types=batch["features"]["atomic_numbers"].long(),
        )
        bs_drmsd, clash_error = compute_drmsd_and_clashloss(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
            lit_module.net.atnum2vdw_uff,
            pred_lig_coords=scores["final_coords_lig_atom"],
            target_lig_coords=batch["features"]["sdf_coordinates"],
            ligatm_types=batch["features"]["atomic_numbers"].long(),
            binding_site=True,
        )
        pli_drmsd, _ = compute_drmsd_and_clashloss(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
            lit_module.net.atnum2vdw_uff,
            pred_lig_coords=scores["final_coords_lig_atom"],
            target_lig_coords=batch["features"]["sdf_coordinates"],
            ligatm_types=batch["features"]["atomic_numbers"].long(),
            pl_interface=True,
        )
        distgeom_loss = (
            aa_distgeom_error.mul(lambda_weighting) * max(metadata["num_a_per_sample"])
            + lig_distgeom_error.mul(lambda_weighting) * max(metadata["num_i_per_sample"])
        ).mean() / max(metadata["num_a_per_sample"])

        fape_loss = (
            (
                global_fape_pview
                + global_fape_lview
                * (
                    lit_module.hparams.cfg.task.ligand_score_loss_weight
                    / lit_module.hparams.cfg.task.global_score_loss_weight
                )
                + normalized_fape
            )
            .mul(lambda_weighting)
            .mean()
        )

        if not lit_module.hparams.cfg.task.freeze_score_head:
            loss = (
                loss
                + fape_loss
                * lit_module.hparams.cfg.task.global_score_loss_weight
                * is_native_sample
            )
            loss = (
                loss
                + glob_drmsd.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.drmsd_loss_weight
            )
        if use_template:
            twe_drmsd = compute_template_weighted_centroid_drmsd(
                batch, scores["final_coords_prot_atom_padded"]
            )
            if not lit_module.hparams.cfg.task.freeze_score_head:
                loss = (
                    loss
                    + twe_drmsd.mul(lambda_weighting).mean()
                    * lit_module.hparams.cfg.task.drmsd_loss_weight
                )
            lit_module.log(
                f"{stage}/drmsd_loss_weighted",
                twe_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            lit_module.log(
                f"{stage}/drmsd_weighted",
                twe_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        if not lit_module.hparams.cfg.task.freeze_score_head:
            loss = (
                loss
                + bs_drmsd.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.drmsd_loss_weight
            )
            loss = (
                loss
                + pli_drmsd.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.drmsd_loss_weight
            )

            loss = (
                loss
                + distgeom_loss
                * lit_module.hparams.cfg.task.local_distgeom_loss_weight
                * violation_loss_ratio
            )
            loss = (
                loss
                + clash_error.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.clash_loss_weight
                * violation_loss_ratio
            )
        if not lit_module.hparams.cfg.task.freeze_contact_predictor:
            loss = (0.1 + 0.9 * prior_training) * cont_loss + (1 - prior_training * 0.99) * loss
            loss = (
                loss + protein_distogram_loss * lit_module.hparams.cfg.task.distogram_loss_weight
            )
        with torch.no_grad():
            tm_lbound = compute_TMscore_lbound(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            lig_rmsd = segment_mean(
                (
                    (coords_pred_lig - coords_pred_prot.mean(dim=1, keepdim=True))
                    - (coords_ref_lig - coords_ref_prot.mean(dim=1, keepdim=True))
                )
                .square()
                .sum(dim=-1)
                .flatten(0, 1),
                indexer["gather_idx_i_molid"],
                metadata["num_molid"],
            ).sqrt()
            lit_module.log(
                f"{stage}/tm_lbound",
                tm_lbound.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            lit_module.log(
                f"{stage}/ligand_rmsd_ubound",
                lig_rmsd.mean().detach(),
                on_epoch=True,
                batch_size=lig_rmsd.shape[0],
            )
            # L1 score matching loss
            dsm_loss_global = (
                (
                    (coords_pred - coords_pred_prot.mean(dim=1, keepdim=True))
                    - (coords_ref - coords_ref_prot.mean(dim=1, keepdim=True))
                )
                .square()
                .sum(dim=-1)
                .add(1e-2)
                .sqrt()
                .sub(1e-1)
                .mean(dim=1)
                .mul(lambda_weighting)
            )
            dsm_loss_site = (
                (
                    (coords_pred_bs - coords_pred_bs_prot.mean(dim=1, keepdim=True))
                    - (coords_ref_bs - coords_ref_bs_prot.mean(dim=1, keepdim=True))
                )
                .square()
                .sum(dim=-1)
                .add(1e-2)
                .sqrt()
                .sub(1e-1)
                .mean(dim=1)
                .mul(lambda_weighting)
            )
            dsm_loss_ligand = (
                (
                    (coords_pred_lig - coords_pred.mean(dim=1, keepdim=True))
                    - (coords_ref_lig - coords_ref.mean(dim=1, keepdim=True))
                )
                .square()
                .sum(dim=-1)
                .add(1e-2)
                .sqrt()
                .sub(1e-1)
                .mean(dim=1)
                .mul(lambda_weighting)
            )
            lit_module.log(
                f"{stage}/denoising_loss_global",
                dsm_loss_global.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            lit_module.log(
                f"{stage}/denoising_loss_site",
                dsm_loss_site.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            lit_module.log(
                f"{stage}/denoising_loss_ligand",
                dsm_loss_ligand.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        lit_module.log(
            f"{stage}/drmsd_loss_global",
            glob_drmsd.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_loss_site",
            bs_drmsd.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_loss_pli",
            pli_drmsd.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_global",
            glob_drmsd.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_site",
            bs_drmsd.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_pli",
            pli_drmsd.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_global_proteinview",
            global_fape_pview.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_global_ligandview",
            global_fape_lview.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_normalized",
            normalized_fape.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_loss",
            fape_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/aa_distgeom_error",
            aa_distgeom_error.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/lig_distgeom_error",
            lig_distgeom_error.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/clash_error",
            clash_error.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/clash_loss",
            clash_error.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/distgeom_loss",
            distgeom_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
    else:
        coords_pred = scores["final_coords_prot_atom_padded"][res_atom_mask].view(
            batch_size, -1, 3
        )
        coords_ref = batch["features"]["res_atom_positions"][res_atom_mask].view(batch_size, -1, 3)
        global_fape_pview, normalized_fape = compute_fape_from_atom37(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
        )
        aa_distgeom_error = compute_aa_distance_geometry_loss(
            batch,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
        )
        glob_drmsd, clash_error = compute_drmsd_and_clashloss(
            batch,
            device,
            scores["final_coords_prot_atom_padded"],
            batch["features"]["res_atom_positions"],
            lit_module.net.atnum2vdw_uff,
        )
        distgeom_loss = aa_distgeom_error.mul(lambda_weighting).mean()
        fape_loss = (global_fape_pview + normalized_fape).mul(lambda_weighting).mean()

        global_fape_pview.detach()
        if not lit_module.hparams.cfg.task.freeze_score_head:
            loss = (
                loss
                + distgeom_loss
                * lit_module.hparams.cfg.task.local_distgeom_loss_weight
                * violation_loss_ratio
            )
            loss = loss + fape_loss * lit_module.hparams.cfg.task.global_score_loss_weight
            loss = (
                loss
                + glob_drmsd.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.drmsd_loss_weight
            )
        if use_template:
            twe_drmsd = compute_template_weighted_centroid_drmsd(
                batch, scores["final_coords_prot_atom_padded"]
            )
            if not lit_module.hparams.cfg.task.freeze_score_head:
                loss = (
                    loss
                    + twe_drmsd.mul(lambda_weighting).mean()
                    * lit_module.hparams.cfg.task.drmsd_loss_weight
                )
            lit_module.log(
                f"{stage}/drmsd_loss_weighted",
                twe_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            lit_module.log(
                f"{stage}/drmsd_weighted",
                twe_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        if not lit_module.hparams.cfg.task.freeze_score_head:
            loss = (
                loss
                + clash_error.mul(lambda_weighting).mean()
                * lit_module.hparams.cfg.task.clash_loss_weight
                * violation_loss_ratio
            )
        if not lit_module.hparams.cfg.task.freeze_contact_predictor:
            loss = (
                loss + protein_distogram_loss * lit_module.hparams.cfg.task.distogram_loss_weight
            )

        with torch.no_grad():
            dsm_loss_global = (
                (
                    (coords_pred - coords_pred.mean(dim=1, keepdim=True))
                    - (coords_ref - coords_ref.mean(dim=1, keepdim=True))
                )
                .square()
                .sum(dim=-1)
                .add(1e-2)
                .sqrt()
                .sub(1e-1)
                .mean(dim=1)
                .mul(lambda_weighting)
            )
            lit_module.log(
                f"{stage}/denoising_loss_global",
                dsm_loss_global.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            tm_lbound = compute_TMscore_lbound(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            lit_module.log(
                f"{stage}/tm_lbound",
                tm_lbound.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        lit_module.log(
            f"{stage}/drmsd_loss_global",
            glob_drmsd.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/drmsd_global",
            glob_drmsd.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_global_proteinview",
            global_fape_pview.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_normalized",
            normalized_fape.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}/fape_loss",
            fape_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/aa_distgeom_error",
            aa_distgeom_error.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/clash_error",
            clash_error.mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/clash_loss",
            clash_error.mul(lambda_weighting).mean().detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_violation/distgeom_loss",
            distgeom_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
    if torch.is_tensor(loss) and not torch.isnan(loss):
        lit_module.log(
            f"{stage}/loss",
            loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=(stage != "train"),
        )
    batch["outputs"]["loss"] = loss
    if not torch.is_tensor(batch["outputs"]["loss"]) and batch["outputs"]["loss"] == 0:
        batch["outputs"]["loss"] = None
    return batch


def eval_auxiliary_estimation_losses(
    lit_module: LightningModule,
    batch: MODEL_BATCH,
    stage: MODEL_STAGE,
    loss_mode: LOSS_MODES,
    **kwargs: Dict[str, Any],
) -> MODEL_BATCH:
    """Evaluate the auxiliary estimation losses for a given batch.

    :param lit_module: The LightningModule object to reference.
    :param batch: A batch dictionary.
    :param stage: The stage of the training.
    :param loss_mode: The loss mode to use.
    :param kwargs: Additional keyword arguments.
    :return: Batch dictionary with losses.
    """
    use_template = bool(random.randint(0, 1))  # nosec
    if use_template:
        # Enable higher ligand diversity when using backbone template
        start_time = 1.0
    else:
        start_time = random.randint(1, 5) / 5  # nosec
    with torch.no_grad():
        if loss_mode == "auxiliary_estimation_without_structure_prediction":
            # Sample the structure without using the structure prediction head
            # i.e., Provide the holo (ground-truth) protein and ligand structures for affinity estimation
            output_struct = {
                "receptor": batch["features"]["res_atom_positions"].flatten(0, 1),
                "receptor_padded": batch["features"]["res_atom_positions"],
                "ligands": batch["features"]["sdf_coordinates"],
            }
        else:
            output_struct = lit_module.net.sample_pl_complex_structures(
                batch,
                sampler="VDODE",
                sampler_eta=1.0,
                num_steps=int(5 / start_time),
                start_time=start_time,
                exact_prior=True,
                use_template=use_template,
                cutoff=20.0,  # Hot logits
            )
    batch_size = batch["metadata"]["num_structid"]
    batch = lit_module.net.run_auxiliary_estimation(batch, output_struct, **kwargs)
    if lit_module.hparams.cfg.confidence.enabled:
        with torch.no_grad():
            # Receptor centroids
            ref_coords = (
                (
                    batch["features"]["res_atom_positions"]
                    .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                    .sum(dim=1)
                    .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
                )
                .contiguous()
                .view(batch_size, -1, 3)
            )
            pred_coords = (
                (
                    output_struct["receptor_padded"]
                    .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                    .sum(dim=1)
                    .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
                )
                .contiguous()
                .view(batch_size, -1, 3)
            )
            # The number of effective protein atoms used in plddt calculation
            n_protatm_per_sample = pred_coords.shape[1]
            if output_struct["ligands"] is not None:
                ref_lig_coords = (
                    batch["features"]["sdf_coordinates"].contiguous().view(batch_size, -1, 3)
                )
                ref_coords = torch.cat([ref_coords, ref_lig_coords], dim=1)
                pred_lig_coords = output_struct["ligands"].contiguous().view(batch_size, -1, 3)
                pred_coords = torch.cat([pred_coords, pred_lig_coords], dim=1)
            per_atom_lddt, per_atom_lddt_gram = compute_per_atom_lddt(
                batch, pred_coords, ref_coords
            )

        plddt_dev = (per_atom_lddt - batch["outputs"]["plddt"]).abs().mean()
        confidence_loss = (
            F.cross_entropy(
                batch["outputs"]["plddt_logits"].flatten(0, 1),
                per_atom_lddt_gram.flatten(0, 1),
                reduction="none",
            )
            .contiguous()
            .view(batch_size, -1)
        )
        conf_loss = confidence_loss.mean()
        if output_struct["ligands"] is not None:
            plddt_dev_lig = (
                (
                    per_atom_lddt.view(batch_size, -1)[:, n_protatm_per_sample:]
                    - batch["outputs"]["plddt"].view(batch_size, -1)[:, n_protatm_per_sample:]
                )
                .abs()
                .mean()
            )
            conf_loss_lig = confidence_loss[:, n_protatm_per_sample:].mean()
            conf_loss = conf_loss + conf_loss_lig  # + plddt_dev_lig * 0.1
            lit_module.log(
                f"{stage}_confidence/plddt_dev_lig",
                plddt_dev_lig.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        lit_module.log(
            f"{stage}_confidence/plddt_dev",
            plddt_dev.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        lit_module.log(
            f"{stage}_confidence/loss",
            conf_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=(stage != "train"),
        )
        if lit_module.hparams.cfg.task.freeze_confidence:
            # Keep the confidence prediction parameters in the computational graph but with zero gradients
            conf_loss *= 0
    else:
        conf_loss = 0
    if lit_module.hparams.cfg.affinity.enabled:
        num_molid_per_sample = batch["metadata"]["num_molid"] // batch_size
        gather_idx_molid_structid = torch.arange(
            batch_size, device=batch["outputs"]["affinity_logits"].device
        ).repeat_interleave(num_molid_per_sample)

        # Calculate affinity loss as the mean squared error between the predicted affinity logits and the ground-truth affinity values
        affinity_logits = batch["outputs"]["affinity_logits"]
        # Substitute missing ground-truth affinity values with the affinity head's (detached) predicted logits to indicate no learning signal for these examples
        affinity = torch.where(
            batch["features"]["affinity"].isnan(),
            affinity_logits.detach(),
            batch["features"]["affinity"],
        )
        aff_loss = segment_mean(
            # Find the (batched) mean squared error over all ligand chains in the same complex, then calculate the mean of each batch
            (affinity_logits - affinity).square(),
            gather_idx_molid_structid,
            batch_size,
        ).mean()
        lit_module.log(
            f"{stage}_affinity/loss",
            aff_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=(stage != "train"),
        )
        if lit_module.hparams.cfg.task.freeze_affinity:
            # Keep the affinity prediction parameters in the computational graph but with zero gradients
            aff_loss *= 0
    else:
        aff_loss = 0
    plddt_loss = conf_loss * lit_module.hparams.cfg.task.plddt_loss_weight
    affinity_loss = aff_loss * lit_module.hparams.cfg.task.affinity_loss_weight
    batch["outputs"]["loss"] = plddt_loss + affinity_loss
    if not torch.is_tensor(batch["outputs"]["loss"]) and batch["outputs"]["loss"] == 0:
        batch["outputs"]["loss"] = None
    return batch
