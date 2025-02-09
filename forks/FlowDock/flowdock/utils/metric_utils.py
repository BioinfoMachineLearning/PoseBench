import subprocess  # nosec

import torch
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple

MODEL_BATCH = Dict[str, Any]


@beartype
def calculate_usalign_metrics(
    pred_pdb_filepath: str,
    reference_pdb_filepath: str,
    usalign_exec_path: str,
    flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates US-align structural metrics between predicted and reference macromolecular
    structures.

    :param pred_pdb_filepath: Filepath to predicted macromolecular structure in PDB format.
    :param reference_pdb_filepath: Filepath to reference macromolecular structure in PDB format.
    :param usalign_exec_path: Path to US-align executable.
    :param flags: Command-line flags to pass to US-align, optional.
    :return: Dictionary containing macromolecular US-align structural metrics and metadata.
    """
    # run US-align with subprocess and capture output
    cmd = [usalign_exec_path, pred_pdb_filepath, reference_pdb_filepath]
    if flags is not None:
        cmd += flags
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)  # nosec

    # parse US-align output to extract structural metrics
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Name of Structure_1:"):
            metrics["Name of Structure_1"] = line.split(": ", 1)[1]
        elif line.startswith("Name of Structure_2:"):
            metrics["Name of Structure_2"] = line.split(": ", 1)[1]
        elif line.startswith("Length of Structure_1:"):
            metrics["Length of Structure_1"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Length of Structure_2:"):
            metrics["Length of Structure_2"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Aligned length="):
            aligned_length = line.split("=")[1].split(",")[0]
            rmsd = line.split("=")[2].split(",")[0]
            seq_id = line.split("=")[4]
            metrics["Aligned length"] = int(aligned_length.strip())
            metrics["RMSD"] = float(rmsd.strip())
            metrics["Seq_ID"] = float(seq_id.strip())
        elif line.startswith("TM-score="):
            if "normalized by length of Structure_1" in line:
                metrics["TM-score_1"] = float(line.split("=")[1].split()[0])
            elif "normalized by length of Structure_2" in line:
                metrics["TM-score_2"] = float(line.split("=")[1].split()[0])

    return metrics


def compute_per_atom_lddt(
    batch: MODEL_BATCH, pred_coords: torch.Tensor, target_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes per-atom local distance difference test (LDDT) between predicted and target
    coordinates.

    :param batch: Dictionary containing metadata and target coordinates.
    :param pred_coords: Predicted atomic coordinates.
    :param target_coords: Target atomic coordinates.
    :return: Tuple of lDDT and lDDT list.
    """
    pred_coords = pred_coords.contiguous().view(batch["metadata"]["num_structid"], -1, 3)
    target_coords = target_coords.contiguous().view(batch["metadata"]["num_structid"], -1, 3)
    target_dist = (target_coords[:, :, None] - target_coords[:, None, :]).norm(dim=-1)
    pred_dist = (pred_coords[:, :, None] - pred_coords[:, None, :]).norm(dim=-1)
    conserved_mask = target_dist < 15.0
    lddt_list = []
    thresholds = [0, 0.5, 1, 2, 4, 6, 8, 12, 1e9]
    for threshold_idx in range(8):
        distdiff = (pred_dist - target_dist).abs()
        bin_fraction = (distdiff > thresholds[threshold_idx]) & (
            distdiff < thresholds[threshold_idx + 1]
        )
        lddt_list.append(
            bin_fraction.mul(conserved_mask).long().sum(dim=2) / conserved_mask.long().sum(dim=2)
        )
    lddt_list = torch.stack(lddt_list, dim=-1)
    lddt = torch.cumsum(lddt_list[:, :, :4], dim=-1).mean(dim=-1)
    return lddt, lddt_list
