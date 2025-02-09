import os

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Any, Dict, List, Optional, Tuple
from lightning import LightningModule
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import AllChem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import (
    collate_numpy_samples,
    process_mol_file,
)
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    FDProtein,
    merge_protein_and_ligands,
    pdb_filepath_to_protein,
    prepare_batch,
    process_protein,
)
from flowdock.utils.model_utils import inplace_to_device, inplace_to_torch, segment_mean
from flowdock.utils.visualization_utils import (
    write_conformer_sdf,
    write_pdb_models,
    write_pdb_single,
)

log = RankedLogger(__name__, rank_zero_only=True)


def featurize_protein_and_ligands(
    rec_path: str,
    lig_paths: List[str],
    n_lig_patches: int,
    apo_rec_path: Optional[str] = None,
    chain_id: Optional[str] = None,
    protein: Optional[FDProtein] = None,
    sequences_to_embeddings: Optional[Dict[str, np.ndarray]] = None,
    enforce_sanitization: bool = False,
    discard_sdf_coords: bool = False,
    **kwargs: Dict[str, Any],
):
    """Featurize a protein-ligand complex.

    :param rec_path: Path to the receptor file.
    :param lig_paths: List of paths to the ligand files.
    :param n_lig_patches: Number of ligand patches.
    :param apo_rec_path: Path to the apo receptor file.
    :param chain_id: Chain ID of the receptor.
    :param protein: Optional protein object.
    :param sequences_to_embeddings: Mapping of sequences to embeddings.
    :param enforce_sanitization: Whether to enforce sanitization.
    :param discard_sdf_coords: Whether to discard SDF coordinates.
    :param kwargs: Additional keyword arguments.
    :return: Featurized protein-ligand complex.
    """
    assert rec_path is not None
    if lig_paths is None:
        lig_paths = []
    if isinstance(lig_paths, str):
        lig_paths = [lig_paths]
    out_mol = None
    lig_samples = []
    for lig_path in lig_paths:
        try:
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                sanitize=True,
                return_mol=True,
                discard_coords=discard_sdf_coords,
            )
        except Exception as e:
            if enforce_sanitization:
                raise
            log.warning(
                f"RDKit sanitization failed for ligand {lig_path} due to: {e}. Loading raw attributes."
            )
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                sanitize=False,
                return_mol=True,
                discard_coords=discard_sdf_coords,
            )
        lig_samples.append(lig_sample)
        if out_mol is None:
            out_mol = mol_ref
        else:
            out_mol = AllChem.CombineMols(out_mol, mol_ref)
    protein = protein if protein is not None else pdb_filepath_to_protein(rec_path)
    rec_sample = process_protein(
        protein,
        chain_id=chain_id,
        sequences_to_embeddings=None if apo_rec_path is not None else sequences_to_embeddings,
        **kwargs,
    )
    if apo_rec_path is not None:
        apo_protein = pdb_filepath_to_protein(apo_rec_path)
        apo_rec_sample = process_protein(
            apo_protein,
            chain_id=chain_id,
            sequences_to_embeddings=sequences_to_embeddings,
            **kwargs,
        )
        for key in rec_sample.keys():
            for subkey, value in apo_rec_sample[key].items():
                rec_sample[key]["apo_" + subkey] = value
    merged_sample = merge_protein_and_ligands(
        lig_samples,
        rec_sample,
        n_lig_patches=n_lig_patches,
        label=None,
    )
    return merged_sample, out_mol


def multi_pose_sampling(
    receptor_path: str,
    ligand_path: str,
    cfg: DictConfig,
    lit_module: LightningModule,
    out_path: str,
    save_pdb: bool = True,
    separate_pdb: bool = True,
    chain_id: Optional[str] = None,
    apo_receptor_path: Optional[str] = None,
    sample_id: Optional[str] = None,
    protein: Optional[FDProtein] = None,
    sequences_to_embeddings: Optional[Dict[str, np.ndarray]] = None,
    confidence: bool = True,
    affinity: bool = True,
    return_all_states: bool = False,
    auxiliary_estimation_only: bool = False,
    **kwargs: Dict[str, Any],
) -> Tuple[
    Optional[Chem.Mol],
    Optional[List[float]],
    Optional[List[float]],
    Optional[List[float]],
    Optional[List[Any]],
    Optional[Any],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Sample multiple poses of a protein-ligand complex.

    :param receptor_path: Path to the receptor file.
    :param ligand_path: Path to the ligand file.
    :param cfg: Config dictionary.
    :param lit_module: LightningModule instance.
    :param out_path: Path to save the output files.
    :param save_pdb: Whether to save PDB files.
    :param separate_pdb: Whether to save separate PDB files for each pose.
    :param chain_id: Chain ID of the receptor.
    :param apo_receptor_path: Path to the optional apo receptor file.
    :param sample_id: Optional sample ID.
    :param protein: Optional protein object.
    :param sequences_to_embeddings: Mapping of sequences to embeddings.
    :param confidence: Whether to estimate confidence scores.
    :param affinity: Whether to estimate affinity scores.
    :param return_all_states: Whether to return all states.
    :param auxiliary_estimation_only: Whether to only estimate auxiliary outputs (e.g., confidence,
        affinity) for the input (generated) samples (potentially derived from external sources).
    :param kwargs: Additional keyword arguments.
    :return: Reference molecule, protein plDDTs, ligand plDDTs, ligand fragment plDDTs, estimated
        binding affinities, structure trajectories, input batch, B-factors, and structure rankings.
    """
    if return_all_states and auxiliary_estimation_only:
        # NOTE: If auxiliary estimation is solely enabled, structure trajectory sampling will be disabled
        return_all_states = False
    struct_res_all, lig_res_all = [], []
    plddt_all, plddt_lig_all, plddt_ligs_all, res_plddt_all = [], [], [], []
    affinity_all, ligs_affinity_all = [], []
    frames_all = []
    chunk_size = cfg.chunk_size
    for _ in range(cfg.n_samples // chunk_size):
        # Resample anchor node frames
        np_sample, mol = featurize_protein_and_ligands(
            receptor_path,
            ligand_path,
            n_lig_patches=lit_module.hparams.cfg.mol_encoder.n_patches,
            apo_rec_path=apo_receptor_path,
            chain_id=chain_id,
            protein=protein,
            sequences_to_embeddings=sequences_to_embeddings,
            discard_sdf_coords=cfg.discard_sdf_coords and not auxiliary_estimation_only,
            **kwargs,
        )
        np_sample_batched = collate_numpy_samples([np_sample for _ in range(chunk_size)])
        sample = inplace_to_device(inplace_to_torch(np_sample_batched), device=lit_module.device)
        prepare_batch(sample)
        if auxiliary_estimation_only:
            # Predict auxiliary quantities using the provided input protein and ligand structures
            if "num_molid" in sample["metadata"].keys() and sample["metadata"]["num_molid"] > 0:
                sample["misc"]["protein_only"] = False
            else:
                sample["misc"]["protein_only"] = True
            output_struct = {
                "receptor": sample["features"]["res_atom_positions"].flatten(0, 1),
                "receptor_padded": sample["features"]["res_atom_positions"],
                "ligands": sample["features"]["sdf_coordinates"],
            }
        else:
            output_struct = lit_module.net.sample_pl_complex_structures(
                sample,
                sampler=cfg.sampler,
                sampler_eta=cfg.sampler_eta,
                num_steps=cfg.num_steps,
                return_all_states=return_all_states,
                start_time=cfg.start_time,
                exact_prior=cfg.exact_prior,
            )
        frames_all.append(output_struct.get("all_frames", None))
        if mol is not None:
            ref_mol = AllChem.Mol(mol)
            out_x1 = np.split(output_struct["ligands"].cpu().numpy(), cfg.chunk_size)
        out_x2 = np.split(output_struct["receptor_padded"].cpu().numpy(), cfg.chunk_size)
        if confidence and affinity:
            assert (
                lit_module.net.confidence_cfg.enabled
            ), "Confidence estimation must be enabled in the model configuration."
            assert (
                lit_module.net.affinity_cfg.enabled
            ), "Affinity estimation must be enabled in the model configuration."
            plddt, plddt_lig, plddt_ligs = lit_module.net.run_auxiliary_estimation(
                sample,
                output_struct,
                return_avg_stats=True,
                training=False,
            )
            aff = sample["outputs"]["affinity_logits"]
        elif confidence:
            assert (
                lit_module.net.confidence_cfg.enabled
            ), "Confidence estimation must be enabled in the model configuration."
            plddt, plddt_lig, plddt_ligs = lit_module.net.run_auxiliary_estimation(
                sample,
                output_struct,
                return_avg_stats=True,
                training=False,
            )
        elif affinity:
            assert (
                lit_module.net.affinity_cfg.enabled
            ), "Affinity estimation must be enabled in the model configuration."
            lit_module.net.run_auxiliary_estimation(
                sample, output_struct, return_avg_stats=True, training=False
            )
            plddt, plddt_lig = None, None
            aff = sample["outputs"]["affinity_logits"].cpu()

        mol_idx_i_structid = segment_mean(
            sample["indexer"]["gather_idx_i_structid"],
            sample["indexer"]["gather_idx_i_molid"],
            sample["metadata"]["num_molid"],
        ).long()
        for struct_idx in range(cfg.chunk_size):
            struct_res = {
                "features": {
                    "asym_id": np_sample["features"]["res_chain_id"],
                    "residue_index": np.arange(len(np_sample["features"]["res_type"])) + 1,
                    "aatype": np_sample["features"]["res_type"],
                },
                "structure_module": {
                    "final_atom_positions": out_x2[struct_idx],
                    "final_atom_mask": sample["features"]["res_atom_mask"].bool().cpu().numpy(),
                },
            }
            struct_res_all.append(struct_res)
            if mol is not None:
                lig_res_all.append(out_x1[struct_idx])
            if confidence:
                plddt_all.append(plddt[struct_idx].item())
                res_plddt_all.append(
                    sample["outputs"]["plddt"][
                        struct_idx, : sample["metadata"]["num_a_per_sample"][0]
                    ]
                    .cpu()
                    .numpy()
                )
                if plddt_lig is None:
                    plddt_lig_all.append(None)
                else:
                    plddt_lig_all.append(plddt_lig[struct_idx].item())
                if plddt_ligs is None:
                    plddt_ligs_all.append(None)
                else:
                    plddt_ligs_all.append(plddt_ligs[mol_idx_i_structid == struct_idx].tolist())
            if affinity:
                # collect the average affinity across all ligands in each complex
                ligs_aff = aff[mol_idx_i_structid == struct_idx]
                affinity_all.append(ligs_aff.mean().item())
                ligs_affinity_all.append(ligs_aff.tolist())
    if confidence and cfg.rank_outputs_by_confidence:
        plddt_lig_predicted = all(plddt_lig_all)
        if cfg.plddt_ranking_type == "protein":
            struct_plddts = np.array(plddt_all)  # rank outputs using average protein plDDT
        elif cfg.plddt_ranking_type == "ligand":
            struct_plddts = np.array(
                plddt_lig_all if plddt_lig_predicted else plddt_all
            )  # rank outputs using average ligand plDDT if available
            if not plddt_lig_predicted:
                log.warning(
                    "Ligand plDDT not available for all samples, using protein plDDT instead"
                )
        elif cfg.plddt_ranking_type == "protein_ligand":
            struct_plddts = np.array(
                plddt_all + plddt_lig_all if plddt_lig_predicted else plddt_all
            )  # rank outputs using the sum of the average protein and ligand plDDTs if ligand plDDT is available
            if not plddt_lig_predicted:
                log.warning(
                    "Ligand plDDT not available for all samples, using protein plDDT instead"
                )
        struct_plddt_rankings = np.argsort(
            -struct_plddts
        ).argsort()  # ensure that higher plDDTs have a higher rank (e.g., `rank1`)
    receptor_plddt = np.array(res_plddt_all) if confidence else None
    b_factors = (
        np.repeat(
            receptor_plddt[..., None],
            struct_res_all[0]["structure_module"]["final_atom_mask"].shape[-1],
            axis=-1,
        )
        if confidence
        else None
    )
    if save_pdb:
        if separate_pdb:
            for struct_id, struct_res in enumerate(struct_res_all):
                if confidence and cfg.rank_outputs_by_confidence:
                    write_pdb_single(
                        struct_res,
                        out_path=os.path.join(
                            out_path,
                            f"prot_rank{struct_plddt_rankings[struct_id] + 1}_plddt{struct_plddts[struct_id]:.7f}{f'_affinity{affinity_all[struct_id]:.7f}' if affinity else ''}.pdb",
                        ),
                        b_factors=b_factors[struct_id] if confidence else None,
                    )
                else:
                    write_pdb_single(
                        struct_res,
                        out_path=os.path.join(
                            out_path,
                            f"prot_{struct_id}{f'_affinity{affinity_all[struct_id]:.7f}' if affinity else ''}.pdb",
                        ),
                        b_factors=b_factors[struct_id] if confidence else None,
                    )
        write_pdb_models(
            struct_res_all, out_path=os.path.join(out_path, "prot_all.pdb"), b_factors=b_factors
        )
    if mol is not None:
        write_conformer_sdf(ref_mol, None, out_path=os.path.join(out_path, "lig_ref.sdf"))
        lig_res_all = np.array(lig_res_all)
        write_conformer_sdf(mol, lig_res_all, out_path=os.path.join(out_path, "lig_all.sdf"))
        for struct_id in range(len(lig_res_all)):
            if confidence and cfg.rank_outputs_by_confidence:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(
                        out_path,
                        f"lig_rank{struct_plddt_rankings[struct_id] + 1}_plddt{struct_plddts[struct_id]:.7f}{f'_affinity{affinity_all[struct_id]:.7f}' if affinity else ''}.sdf",
                    ),
                )
            else:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(
                        out_path,
                        f"lig_{struct_id}{f'_affinity{affinity_all[struct_id]:.7f}' if affinity else ''}.sdf",
                    ),
                )
        if confidence:
            aux_estimation_all_df = pd.DataFrame(
                {
                    "sample_id": [sample_id] * len(struct_res_all),
                    "rank": struct_plddt_rankings + 1 if cfg.rank_outputs_by_confidence else None,
                    "plddt_ligs": plddt_ligs_all,
                    "affinity_ligs": ligs_affinity_all,
                }
            )
            aux_estimation_all_df.to_csv(
                os.path.join(out_path, "auxiliary_estimation.csv"), index=False
            )
    else:
        ref_mol = None
    if not confidence:
        plddt_all, plddt_lig_all, plddt_ligs_all = None, None, None
    if not affinity:
        affinity_all = None
    if return_all_states:
        if mol is not None:
            np_sample["metadata"]["sample_ID"] = sample_id if sample_id is not None else "sample"
            np_sample["metadata"]["mol"] = ref_mol
        batch_all = inplace_to_torch(
            collate_numpy_samples([np_sample for _ in range(cfg.n_samples)])
        )
        merge_frames_all = frames_all[0]
        for frames in frames_all[1:]:
            for frame_index, frame in enumerate(frames):
                for key in frame.keys():
                    merge_frames_all[frame_index][key] = torch.cat(
                        [merge_frames_all[frame_index][key], frame[key]], dim=0
                    )
        frames_all = merge_frames_all
    else:
        frames_all = None
        batch_all = None
    if not (confidence and cfg.rank_outputs_by_confidence):
        struct_plddt_rankings = None
    return (
        ref_mol,
        plddt_all,
        plddt_lig_all,
        plddt_ligs_all,
        affinity_all,
        frames_all,
        batch_all,
        b_factors,
        struct_plddt_rankings,
    )
