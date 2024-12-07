import os
import random
from functools import partial

import rootutils
import torch
import torch.nn.functional as F
import tqdm
from beartype.typing import Any, Callable, Dict, Literal, Optional, Tuple, Union
from omegaconf import DictConfig
from pytorch3d.ops import corresponding_points_alignment

from flowdock.utils.inspect_ode_samplers import clamp_tensor

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import collate_numpy_samples
from flowdock.data.components.physical import (
    get_vdw_radii_array,
    get_vdw_radii_array_uff,
)
from flowdock.models.components.cpm import (
    resolve_pl_contact_stack,
    resolve_protein_encoder,
)
from flowdock.models.components.esdm import (
    resolve_affinity_head,
    resolve_confidence_head,
    resolve_score_head,
)
from flowdock.models.components.losses import (
    compute_fape_from_atom37,
    compute_lddt_ca,
    compute_lddt_pli,
    compute_TMscore_lbound,
    compute_TMscore_raw,
)
from flowdock.models.components.mht_encoder import (
    resolve_ligand_encoder,
    resolve_relational_reasoning_module,
)
from flowdock.models.components.noise import (
    sample_complex_harmonic_prior,
    sample_esmfold_prior,
    sample_gaussian_prior,
    sample_ligand_harmonic_prior,
)
from flowdock.models.components.transforms import (
    DefaultPLCoordinateConverter,
    LatentCoordinateConverter,
)
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    erase_holo_coordinates,
    get_standard_aa_features,
    prepare_batch,
)
from flowdock.utils.frame_utils import apply_similarity_transform
from flowdock.utils.model_utils import (
    distogram_to_gaussian_contact_logits,
    eval_true_contact_maps,
    inplace_to_device,
    inplace_to_torch,
    sample_res_rowmask_from_contacts,
    sample_reslig_contact_matrix,
    segment_argmin,
    segment_mean,
    segment_sum,
    topk_edge_mask_from_logits,
)

MODEL_BATCH = Dict[str, Any]
NANOMETERS_TO_ANGSTROM = 10.0

log = RankedLogger(__name__, rank_zero_only=True)


class FlowDock(torch.nn.Module):
    """A geometric conditional flow matching model for protein-ligand docking."""

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """Initialize a `FlowDock` module.

        :param cfg: A model configuration dictionary.
        """
        super().__init__()
        self.cfg = cfg
        self.ligand_cfg = cfg.mol_encoder
        self.protein_cfg = cfg.protein_encoder
        self.relational_reasoning_cfg = cfg.relational_reasoning
        self.contact_cfg = cfg.contact_predictor
        self.score_cfg = cfg.score_head
        self.confidence_cfg = cfg.confidence
        self.affinity_cfg = cfg.affinity
        self.global_cfg = cfg.task
        self.protatm_padding_dim = self.protein_cfg.atom_padding_dim  # := 37
        self.max_n_edges = self.global_cfg.edge_crop_size
        self.latent_model = cfg.latent_model
        self.prior_type = cfg.prior_type

        # VDW radius mapping, in Angstrom
        self.atnum2vdw = torch.nn.Parameter(
            torch.tensor(get_vdw_radii_array() / 100.0),
            requires_grad=False,
        )
        self.atnum2vdw_uff = torch.nn.Parameter(
            torch.tensor(get_vdw_radii_array_uff() / 100.0),
            requires_grad=False,
        )

        # graph hyperparameters
        self.BINDING_SITE_CUTOFF = 6.0
        self.INTERNAL_TARGET_VARIANCE_SCALE = self.global_cfg.internal_max_sigma
        self.GLOBAL_TARGET_VARIANCE_SCALE = self.global_cfg.global_max_sigma
        self.CONTACT_SCALE = 5.0  # fixed hyperparameter

        (
            standard_aa_template_featset,
            standard_aa_graph_featset,
        ) = get_standard_aa_features()
        self.standard_aa_template_featset = inplace_to_torch(standard_aa_template_featset)
        self.standard_aa_molgraph_featset = inplace_to_torch(
            collate_numpy_samples(standard_aa_graph_featset)
        )

        # load pretrained weights as desired
        from_pretrained = (
            self.ligand_cfg.from_pretrained
            or self.protein_cfg.from_pretrained
            or self.relational_reasoning_cfg.from_pretrained
            or self.contact_cfg.from_pretrained
            or self.score_cfg.from_pretrained
            or self.confidence_cfg.from_pretrained
            or self.affinity_cfg.from_pretrained
        )
        if from_pretrained:
            assert cfg.mol_encoder.checkpoint_file is not None and os.path.exists(
                cfg.mol_encoder.checkpoint_file
            ), "Pretrained model weights not found."
        pretrained_state_dict = (
            torch.load(cfg.mol_encoder.checkpoint_file)["state_dict"] if from_pretrained else None
        )

        # ligand encoder
        self.ligand_encoder = resolve_ligand_encoder(
            self.ligand_cfg, self.global_cfg, state_dict=pretrained_state_dict
        )
        self.lig_masking_rate = self.global_cfg.max_masking_rate

        # protein structure encoder
        self.protein_encoder, res_in_projector = resolve_protein_encoder(
            self.protein_cfg, self.global_cfg, state_dict=pretrained_state_dict
        )
        # protein sequence encoder
        if self.protein_cfg.use_esm_embedding:
            # protein sequence language model
            self.plm_adapter = res_in_projector
        else:
            # one-hot amino acid types
            self.res_in_projector = res_in_projector

        # relational reasoning module
        (
            self.molgraph_single_projector,
            self.molgraph_pair_projector,
            self.covalent_embed,
        ) = resolve_relational_reasoning_module(
            self.protein_cfg,
            self.ligand_cfg,
            self.relational_reasoning_cfg,
            state_dict=pretrained_state_dict,
        )

        # contact prediction module
        (
            self.pl_contact_stack,
            self.contact_code_embed,
            self.dist_bins,
            self.dgram_head,
        ) = resolve_pl_contact_stack(
            self.protein_cfg,
            self.ligand_cfg,
            self.contact_cfg,
            self.global_cfg,
            state_dict=pretrained_state_dict,
        )

        # structure denoising module
        self.score_head = resolve_score_head(
            self.protein_cfg, self.score_cfg, self.global_cfg, state_dict=pretrained_state_dict
        )

        # confidence prediction module
        if self.confidence_cfg.enabled:
            self.confidence_head, self.plddt_gram_head = resolve_confidence_head(
                self.protein_cfg,
                self.confidence_cfg,
                self.global_cfg,
                state_dict=pretrained_state_dict,
            )

        # affinity prediction module
        if self.affinity_cfg.enabled:
            (
                self.affinity_head,
                self.ligand_pooling,
                self.affinity_proj_head,
            ) = resolve_affinity_head(
                self.ligand_cfg,
                self.affinity_cfg,
                self.global_cfg,
                learnable_pooling=True,
                state_dict=pretrained_state_dict,
            )

        self.freeze_pretraining_params()

    def freeze_pretraining_params(self):
        """Freeze pretraining parameters."""
        if self.global_cfg.freeze_mol_encoder:
            log.info("Freezing ligand encoder parameters.")
            self.ligand_encoder.eval()
            for p in self.ligand_encoder.parameters():
                p.requires_grad = False
        if self.global_cfg.freeze_protein_encoder:
            log.info("Freezing protein encoder parameters.")
            for module in [
                (
                    self.plm_adapter
                    if self.protein_cfg.use_esm_embedding
                    else self.res_in_projector
                ),
                self.protein_encoder,
            ]:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.global_cfg.freeze_relational_reasoning:
            log.info("Freezing relational reasoning module parameters.")
            for module in [
                self.molgraph_single_projector,
                self.molgraph_pair_projector,
                self.covalent_embed,
            ]:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.global_cfg.freeze_contact_predictor:
            log.info("Freezing contact prediction module parameters.")
            for module in [
                self.pl_contact_stack,
                self.contact_code_embed,
                self.dgram_head,
            ]:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.global_cfg.freeze_score_head:
            log.info("Freezing structure denoising module parameters.")
            self.score_head.eval()
            for p in self.score_head.parameters():
                p.requires_grad = False
        if self.confidence_cfg.enabled and self.global_cfg.freeze_confidence:
            log.info("Freezing confidence prediction module parameters.")
            for module in [self.confidence_head, self.plddt_gram_head]:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.affinity_cfg.enabled and self.global_cfg.freeze_affinity:
            log.info("Freezing affinity prediction module parameters.")
            self.affinity_head.eval()
            for p in self.affinity_head.parameters():
                p.requires_grad = False

    @staticmethod
    def assign_timestep_encodings(batch: MODEL_BATCH, t_normalized: Union[float, torch.Tensor]):
        """Assign timestep encodings to the batch.

        :param batch: A batch dictionary.
        :param t_normalized: The normalized timestep.
        """
        # NOTE: `t_normalized` must be in the range `[0, 1]`
        features = batch["features"]
        indexer = batch["indexer"]
        device = features["res_type"].device
        if not isinstance(t_normalized, torch.Tensor):
            t_normalized = torch.full(
                (batch["metadata"]["num_structid"], 1),
                t_normalized,
                device=device,
            )
        if t_normalized.shape != (batch["metadata"]["num_structid"], 1):
            assert (
                t_normalized.numel() == 1
            ), f"To properly shape-coerce time step tensor of shape {t_normalized.shape}, the input tensor must contain a single value."
            t_normalized = torch.full(
                (batch["metadata"]["num_structid"], 1),
                t_normalized.item(),
                device=device,
            )
        t_prot = t_normalized[indexer["gather_idx_a_structid"]]
        batch["features"]["timestep_encoding_prot"] = t_prot

        if not batch["misc"]["protein_only"]:
            batch["features"]["timestep_encoding_lig"] = t_normalized[
                indexer["gather_idx_i_structid"]
            ]

    def resolve_latent_converter(self, *args):
        if self.latent_model == "default":
            return DefaultPLCoordinateConverter(self.global_cfg, *args)
        else:
            raise NotImplementedError

    def forward_interp(
        self,
        batch: MODEL_BATCH,
        x_int_0: torch.Tensor,
        t: torch.Tensor,
        latent_converter: LatentCoordinateConverter,
        umeyama_correction: bool = True,
        erase_data: bool = False,
    ) -> torch.Tensor:
        """Interpolate latent internal coordinates.

        Note that this function adds small amounts of Gaussian noise
        to the ground-truth protein and ligand coordinates, to discourage
        the model from overfitting to experimental noise in the training data.
        Reference: https://www.science.org/doi/10.1126/science.add2187

        :param batch: A batch dictionary.
        :param x_int_0: Dimension-less (ground-truth) internal coordinates.
        :param t: The current normalized timestep.
        :param latent_converter: The latent coordinate converter.
        :param umeyama_correction: Whether to apply the Umeyama correction.
        :param erase_data: Whether to erase data.
        :return: Interpolated latent internal coordinates.
        """
        (
            ca_lat,
            apo_ca_lat,
            cother_lat,
            apo_cother_lat,
            ca_lat_centroid_coords,
            apo_ca_lat_centroid_coords,
            lig_lat,
        ) = torch.split(
            x_int_0,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_int_0_ = torch.cat(
            [
                ca_lat,
                cother_lat,
                lig_lat,
            ],
            dim=1,
        )
        try:
            assert self.global_cfg.single_protein_batch, "Only single protein batch is supported."
            if self.prior_type == "gaussian":
                noisy_x_int_0, noisy_x_int_1 = sample_gaussian_prior(
                    x_int_0_, latent_converter, sigma=1.0, x0_sigma=1e-4
                )
            elif self.prior_type == "harmonic":
                noisy_x_int_0, noisy_x_int_1 = sample_complex_harmonic_prior(
                    x_int_0_, latent_converter, batch, x0_sigma=1e-4
                )
            elif self.prior_type == "esmfold":
                # NOTE: the following unnormalization step assumes that `self.latent_model == "default"`
                apo_lig_lat_ = sample_ligand_harmonic_prior(
                    lig_lat, apo_ca_lat * latent_converter.ca_scale, batch
                )
                x_int_1_ = torch.cat(
                    [
                        apo_ca_lat,  # NOTE: already normalized
                        apo_cother_lat,  # NOTE: already normalized
                        apo_lig_lat_ / latent_converter.other_scale,
                    ],
                    dim=1,
                )
                noisy_x_int_0, noisy_x_int_1 = sample_esmfold_prior(
                    x_int_0_, x_int_1_, sigma=1e-4, x0_sigma=1e-4
                )
            else:
                raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        except Exception as e:
            log.error(
                f"Failed to converge within `{self.prior_type}` noise function of `forward_interp()` due to: {e}."
            )
            raise e

        # NOTE: by this point, both `noisy_x_int_0` and `noisy_x_int_1` are normalized
        if umeyama_correction and not erase_data:
            try:
                # align the complex structure based solely the optimal Ca atom alignment
                # NOTE: we do not perform such alignments during the initial (`t=1`) sampling timestep
                noisy_x_ca_int_1 = noisy_x_int_1.split(
                    [
                        latent_converter._n_res_per_sample,
                        latent_converter._n_cother_per_sample,
                        latent_converter._n_ligha_per_sample,
                    ],
                    dim=1,
                )[0]
                noisy_x_ca_int_0 = noisy_x_int_0.split(
                    [
                        latent_converter._n_res_per_sample,
                        latent_converter._n_cother_per_sample,
                        latent_converter._n_ligha_per_sample,
                    ],
                    dim=1,
                )[0]
                similarity_transform = corresponding_points_alignment(
                    X=noisy_x_ca_int_1, Y=noisy_x_ca_int_0, estimate_scale=False
                )
                noisy_x_int_1 = apply_similarity_transform(noisy_x_int_1, *similarity_transform)
            except Exception as e:
                log.warning(
                    f"Failed optimal noise alignment within `forward_interp()` due to: {e}. Skipping optimal noise alignment..."
                )
                raise e

        # interpolate between target and prior distributions
        noisy_x_int_t = (1 - t) * noisy_x_int_0 + t * noisy_x_int_1

        # recalculate (and renormalize) centroid coordinates
        (
            x_int_t_ca_lat,
            x_int_t_cother_lat,
            x_int_t_lig_lat,
        ) = torch.split(
            noisy_x_int_t,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        (
            x_int_1_ca_lat,
            x_int_1_cother_lat,
            _,
        ) = torch.split(
            noisy_x_int_1,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_int = torch.cat(
            [
                x_int_t_ca_lat,
                x_int_1_ca_lat,
                x_int_t_cother_lat,
                x_int_1_cother_lat,
                ca_lat_centroid_coords,
                apo_ca_lat_centroid_coords,
                x_int_t_lig_lat,
            ],
            dim=1,
        )
        return x_int

    def forward_interp_plcomplex_latinp(
        self,
        batch: MODEL_BATCH,
        t: torch.Tensor,
        latent_converter: LatentCoordinateConverter,
        umeyama_correction: bool = True,
        erase_data: bool = False,
    ) -> MODEL_BATCH:
        """Noise-interpolate protein-ligand complex latent internal coordinates.

        :param batch: A batch dictionary.
        :param t: The current normalized timestep.
        :param latent_converter: The latent coordinate converter.
        :param umeyama_correction: Whether to apply the Umeyama correction.
        :param erase_data: Whether to erase data.
        :return: Batch dictionary with interpolated latent internal coordinates.
        """
        # Dimension-less internal coordinates
        # [B, N, 3]
        x_int = latent_converter.to_latent(batch)
        if erase_data:
            x_int = erase_holo_coordinates(batch, x_int, latent_converter)
        x_int_t = self.forward_interp(
            batch,
            x_int,
            t,
            latent_converter,
            umeyama_correction=umeyama_correction,
            erase_data=erase_data,
        )
        return latent_converter.assign_to_batch(batch, x_int_t)

    def prepare_protein_patch_indexers(
        self, batch: MODEL_BATCH, randomize_anchors: bool = False
    ) -> MODEL_BATCH:
        """Prepare protein patch indexers for the batch.

        :param batch: A batch dictionary.
        :param randomize_anchors: Whether to randomize the anchors.
        :return: Batch dictionary with protein patch indexers.
        """
        features = batch["features"]
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        device = features["res_type"].device

        # Prepare indexers
        # Use max to ensure segmentation faults are 100% invoked
        # in case there are any bad indices
        max(metadata["num_a_per_sample"])
        n_a_per_sample = max(metadata["num_a_per_sample"])
        assert (
            n_a_per_sample * batch_size == metadata["num_a"]
        ), "Invalid (batched) number of residues"
        n_protein_patches = min(self.protein_cfg.n_patches, n_a_per_sample)
        batch["metadata"]["n_prot_patches_per_sample"] = n_protein_patches

        # Uniform segmentation
        res_idx_in_batch = torch.arange(metadata["num_a"], device=device)
        batch["indexer"]["gather_idx_a_pid"] = (
            res_idx_in_batch // n_a_per_sample
        ) * n_protein_patches + (
            ((res_idx_in_batch % n_a_per_sample) * n_protein_patches) // n_a_per_sample
        )

        if randomize_anchors:
            # Random down-sampling, assigning residues to the patch grid
            # This maps grid row/column idx to sampled residue idx
            batch["indexer"]["gather_idx_pid_a"] = segment_argmin(
                batch["features"]["res_type"].new_zeros(n_a_per_sample * batch_size),
                indexer["gather_idx_a_pid"],
                n_protein_patches * batch_size,
                randomize=True,
            )
        else:
            batch["indexer"]["gather_idx_pid_a"] = segment_mean(
                res_idx_in_batch,
                indexer["gather_idx_a_pid"],
                n_protein_patches * batch_size,
            ).long()

        return batch

    def prepare_protein_backbone_indexers(
        self, batch: MODEL_BATCH, **kwargs: Dict[str, Any]
    ) -> MODEL_BATCH:
        """Prepare protein backbone indexers for the batch.

        :param batch: A batch dictionary.
        :param kwargs: Additional keyword arguments.
        :return: Batch dictionary with protein backbone indexers.
        """
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        device = features["res_type"].device

        protatm_coords_padded = features["input_protein_coords"]
        batch_size = metadata["num_structid"]

        assert self.global_cfg.single_protein_batch, "Only single protein batch is supported."
        num_res_per_struct = max(metadata["num_a_per_sample"])
        # Check that the samples are clones of the same complex
        assert (
            batch_size * num_res_per_struct == protatm_coords_padded.shape[0]
        ), "Invalid number of residues."

        input_prot_coords_folded = protatm_coords_padded.unflatten(
            0, (batch_size, num_res_per_struct)
        )
        single_struct_chain_id = indexer["gather_idx_a_chainid"][:num_res_per_struct]
        single_struct_res_id = features["residue_index"][:num_res_per_struct]
        ca_ca_dist = (
            input_prot_coords_folded[:, :, None, 1] - input_prot_coords_folded[:, None, :, 1]
        ).norm(dim=-1)
        ca_ca_knn_mask = topk_edge_mask_from_logits(
            -ca_ca_dist / self.CONTACT_SCALE,
            self.protein_cfg.max_residue_degree,
            randomize=True,
        )
        chain_mask = single_struct_chain_id[None, :, None] == single_struct_chain_id[None, None, :]
        sequence_dist = single_struct_res_id[None, :, None] - single_struct_res_id[None, None, :]
        sequence_proximity_mask = (torch.abs(sequence_dist) <= 4) & chain_mask
        prot_res_res_edge_mask = ca_ca_knn_mask | sequence_proximity_mask

        dense_row_idx_3D = (
            torch.arange(batch_size * num_res_per_struct, device=device)
            .view(batch_size, num_res_per_struct)[:, :, None]
            .expand(-1, -1, num_res_per_struct)
        ).contiguous()
        dense_col_idx_3D = dense_row_idx_3D.transpose(1, 2).contiguous()
        batch["metadata"]["num_prot_res"] = metadata["num_a"]
        batch["indexer"]["gather_idx_ab_a"] = dense_row_idx_3D[prot_res_res_edge_mask]
        batch["indexer"]["gather_idx_ab_b"] = dense_col_idx_3D[prot_res_res_edge_mask]
        batch["indexer"]["gather_idx_ab_structid"] = indexer["gather_idx_a_structid"][
            indexer["gather_idx_ab_a"]
        ]
        batch["metadata"]["num_ab"] = batch["indexer"]["gather_idx_ab_a"].shape[0]

        if self.global_cfg.constrained_inpainting:
            # Diversified spherical cropping scheme
            assert self.global_cfg.single_protein_batch, "Only single protein batch is supported."
            batch_size = metadata["num_structid"]
            # Assert single ligand samples
            assert batch_size == metadata["num_molid"], "Invalid number of ligands."
            ligand_coords = batch["features"]["sdf_coordinates"].reshape(batch_size, -1, 3)
            ligand_centroids = torch.mean(ligand_coords, dim=1)
            if kwargs["training"]:
                # 3A perturbations around the ligand centroid
                perturbed_centroids = ligand_centroids + torch.rand_like(ligand_centroids) * 1.73
                site_radius = torch.amax(
                    torch.norm(ligand_coords - perturbed_centroids[:, None, :], dim=-1),
                    dim=1,
                )
                perturbed_site_radius = (
                    site_radius + (0.5 + torch.rand_like(site_radius)) * self.BINDING_SITE_CUTOFF
                )
            else:
                perturbed_centroids = ligand_centroids
                site_radius = torch.amax(
                    torch.norm(ligand_coords - perturbed_centroids[:, None, :], dim=-1),
                    dim=1,
                )
                perturbed_site_radius = site_radius + self.BINDING_SITE_CUTOFF
            centroid_ca_dist = (
                batch["features"]["res_atom_positions"][:, 1].contiguous().view(batch_size, -1, 3)
                - perturbed_centroids[:, None, :]
            ).norm(dim=-1)
            binding_site_mask = (centroid_ca_dist < perturbed_site_radius[:, None]).flatten(0, 1)
            batch["features"]["binding_site_mask"] = binding_site_mask
            batch["features"]["template_alignment_mask"] = (~binding_site_mask) & batch[
                "features"
            ]["template_alignment_mask"].bool()

        return batch

    def initialize_protein_embeddings(self, batch: MODEL_BATCH):
        """Initialize protein embeddings.

        :param batch: A batch dictionary.
        """
        features = batch["features"]

        # Protein residue and residue-pair embeddings
        if "res_embedding_in" not in features:
            if self.protein_cfg.use_esm_embedding:
                assert (
                    self.global_cfg.single_protein_batch
                ), "Only single protein batch is supported."
                features["res_embedding_in"] = self.plm_adapter(
                    batch["features"]["apo_lm_embeddings"]
                    if "apo_lm_embeddings" in batch["features"]
                    else batch["features"]["lm_embeddings"]
                )
                assert (
                    features["res_embedding_in"].shape[0] == features["res_atom_types"].shape[0]
                ), "Invalid number of residues."
            else:
                features["res_embedding_in"] = self.res_in_projector(
                    F.one_hot(
                        features["res_type"].long(),
                        num_classes=self.protein_cfg.n_aa_types,
                    ).float()
                )

    def initialize_protatm_indexer_and_embeddings(self, batch: MODEL_BATCH):
        """Assign coordinate-independent edges and protein features from PIFormer.

        :param batch: A batch dictionary.
        """
        features = batch["features"]
        device = features["res_type"].device
        assert self.global_cfg.single_protein_batch, "Only single protein batch is supported."
        self.standard_aa_molgraph_featset = inplace_to_device(
            self.standard_aa_molgraph_featset, device
        )
        self.standard_aa_template_featset = inplace_to_device(
            self.standard_aa_template_featset, device
        )
        self.standard_aa_molgraph_featset = self.ligand_encoder(self.standard_aa_molgraph_featset)
        with torch.no_grad():
            assert (
                self.standard_aa_molgraph_featset["metadata"]["num_i"] == 167
            ), "Invalid number of atoms."
            template_atom_idx_in_batch_padded = torch.full(
                (20, 37), fill_value=-1, dtype=torch.long, device=device
            )
            template_atom37_mask = self.standard_aa_template_featset["features"][
                "res_atom_mask"
            ].bool()
            template_atom_idx_in_batch_padded[template_atom37_mask] = torch.arange(
                167, device=device
            )
            atom_idx_in_batch_to_restype_idx = (
                torch.arange(20, device=device)[:, None]
                .expand(-1, 37)
                .contiguous()[template_atom37_mask]
            )
            atom_idx_in_batch_to_atom37_idx = (
                torch.arange(37, device=device)[None, :]
                .expand(20, -1)
                .contiguous()[template_atom37_mask]
            )
            template_padded_edge_mask_per_aa = torch.zeros(
                (20, 37, 37), dtype=torch.bool, device=device
            )
            template_aa_graph_indexer = self.standard_aa_molgraph_featset["indexer"]
            template_padded_edge_mask_per_aa[
                atom_idx_in_batch_to_restype_idx[template_aa_graph_indexer["gather_idx_uv_u"]],
                atom_idx_in_batch_to_atom37_idx[template_aa_graph_indexer["gather_idx_uv_u"]],
                atom_idx_in_batch_to_atom37_idx[template_aa_graph_indexer["gather_idx_uv_v"]],
            ] = True

            # Gather adjacency matrix to the input protein
            features = batch["features"]
            metadata = batch["metadata"]
            # Prepare intra-residue protein atom - protein atom indexers
            n_res_first = max(metadata["num_a_per_sample"])
            batch["features"]["res_atom_mask"] = features["res_atom_mask"].bool()
            protatm_padding_mask = batch["features"]["res_atom_mask"][:n_res_first]
            n_protatm_first = int(protatm_padding_mask.sum())
            protatm_res_idx_res_first = (
                torch.arange(n_res_first, device=device)[:, None]
                .expand(-1, 37)
                .contiguous()[protatm_padding_mask]
            )
            protatm_to_atom37_idx_first = (
                torch.arange(37, device=device)[None, :]
                .expand(n_res_first, -1)
                .contiguous()[protatm_padding_mask]
            )
            same_residue_mask = (
                protatm_res_idx_res_first[:, None] == protatm_res_idx_res_first[None, :]
            ).contiguous()
            aa_graph_edge_mask = torch.zeros(
                (n_protatm_first, n_protatm_first), dtype=torch.bool, device=device
            )
            src_idx_sameres = (
                torch.arange(n_protatm_first, device=device)[:, None]
                .expand(-1, n_protatm_first)
                .contiguous()[same_residue_mask]
            )
            dst_idx_sameres = (
                torch.arange(n_protatm_first, device=device)[None, :]
                .expand(n_protatm_first, -1)
                .contiguous()[same_residue_mask]
            )
            aa_graph_edge_mask[
                src_idx_sameres, dst_idx_sameres
            ] = template_padded_edge_mask_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first[src_idx_sameres]],
                protatm_to_atom37_idx_first[src_idx_sameres],
                protatm_to_atom37_idx_first[dst_idx_sameres],
            ]
            src_idx_first = (
                torch.arange(n_protatm_first, device=device)[:, None]
                .expand(-1, n_protatm_first)
                .contiguous()[aa_graph_edge_mask]
            )
            dst_idx_first = (
                torch.arange(n_protatm_first, device=device)[None, :]
                .expand(n_protatm_first, -1)
                .contiguous()[aa_graph_edge_mask]
            )
            batch_size = metadata["num_structid"]
            src_idx = (
                (
                    src_idx_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=device)[:, None] * n_protatm_first
                )
                .contiguous()
                .flatten()
            )
            dst_idx = (
                (
                    dst_idx_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=device)[:, None] * n_protatm_first
                )
                .contiguous()
                .flatten()
            )
            batch["metadata"]["num_protatm_per_sample"] = n_protatm_first
            batch["indexer"]["protatm_protatm_idx_src"] = src_idx
            batch["indexer"]["protatm_protatm_idx_dst"] = dst_idx
            batch["metadata"]["num_prot_atm"] = n_protatm_first * batch_size
            batch["indexer"]["protatm_res_idx_res"] = (
                (
                    protatm_res_idx_res_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=device)[:, None] * n_res_first
                )
                .contiguous()
                .flatten()
            )
            batch["indexer"]["protatm_res_idx_protatm"] = torch.arange(
                batch["metadata"]["num_prot_atm"], device=device
            )
        # Gather graph features to the protein feature set
        template_padded_node_feat_per_aa = torch.zeros(
            (20, 37, self.protein_cfg.residue_dim), device=device
        )
        template_padded_node_feat_per_aa[template_atom37_mask] = self.molgraph_single_projector(
            self.standard_aa_molgraph_featset["features"]["lig_atom_attr"]
        )
        protatm_padding_mask = batch["features"]["res_atom_mask"]
        protatm_to_atom37_idx = (
            protatm_to_atom37_idx_first[None, :].expand(batch_size, -1).contiguous().flatten(0, 1)
        )
        batch["features"]["protatm_to_atom37_index"] = protatm_to_atom37_idx
        batch["features"]["protatm_to_atomic_number"] = features["res_atom_types"].long()[
            protatm_padding_mask
        ]
        batch["features"]["prot_atom_attr_projected"] = (
            template_padded_node_feat_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first],
                protatm_to_atom37_idx_first,
            ][None, :]
            .expand(batch_size, -1, -1)
            .contiguous()
            .flatten(0, 1)
        )
        template_padded_edge_feat_per_aa = torch.zeros(
            (20, 37, 37, self.protein_cfg.pair_dim), device=device
        )
        template_padded_edge_feat_per_aa[
            atom_idx_in_batch_to_restype_idx[template_aa_graph_indexer["gather_idx_uv_u"]],
            atom_idx_in_batch_to_atom37_idx[template_aa_graph_indexer["gather_idx_uv_u"]],
            atom_idx_in_batch_to_atom37_idx[template_aa_graph_indexer["gather_idx_uv_v"]],
        ] = self.molgraph_pair_projector(
            self.standard_aa_molgraph_featset["features"]["lig_atom_pair_attr"]
        )

        batch["features"]["prot_atom_pair_attr_projected"] = (
            template_padded_edge_feat_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first[src_idx_first]],
                protatm_to_atom37_idx_first[src_idx_first],
                protatm_to_atom37_idx_first[dst_idx_first],
            ][None, :, :]
            .expand(batch_size, -1, -1)
            .contiguous()
            .flatten(0, 1)
        )
        return batch

    def initialize_ligand_embeddings(self, batch: MODEL_BATCH, **kwargs: Dict[str, Any]):
        """Initialize ligand embeddings.

        :param batch: A batch dictionary.
        :param kwargs: Additional keyword arguments.
        """
        metadata = batch["metadata"]
        batch["features"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]

        # Ligand atom, frame and pair embeddings
        if kwargs["training"]:
            masking_rate = random.uniform(0, self.lig_masking_rate)  # nosec
        else:
            masking_rate = 0
        batch = self.ligand_encoder(batch, masking_rate=masking_rate)
        batch["features"]["lig_atom_attr_projected"] = self.molgraph_single_projector(
            batch["features"]["lig_atom_attr"]
        )
        # Downsampled ligand frames
        batch["features"]["lig_trp_attr_projected"] = self.molgraph_single_projector(
            batch["features"]["lig_trp_attr"]
        )
        batch["features"]["lig_atom_pair_attr_projected"] = self.molgraph_pair_projector(
            batch["features"]["lig_atom_pair_attr"]
        )
        lig_af_pair_attr_flat_ = self.molgraph_pair_projector(
            batch["features"]["lig_af_pair_attr"]
        )
        batch["features"]["lig_af_pair_attr_projected"] = lig_af_pair_attr_flat_

        if self.global_cfg.single_protein_batch:
            lig_af_pair_attr = lig_af_pair_attr_flat_.new_zeros(
                batch_size,
                max(metadata["num_U_per_sample"]),
                max(metadata["num_I_per_sample"]),
                self.protein_cfg.pair_dim,
            )
            n_U_first = max(metadata["num_U_per_sample"])
            n_I_first = max(metadata["num_I_per_sample"])
            lig_af_pair_attr[
                indexer["gather_idx_UI_U"] // n_U_first,
                indexer["gather_idx_UI_U"] % n_U_first,
                indexer["gather_idx_UI_I"] % n_I_first,
            ] = lig_af_pair_attr_flat_

            batch["features"]["lig_af_grid_attr_projected"] = lig_af_pair_attr
        else:
            raise NotImplementedError("Only single protein batch is supported.")

    def run_encoder_stack(
        self,
        batch: MODEL_BATCH,
        **kwargs: Dict[str, Any],
    ) -> MODEL_BATCH:
        """Run the encoder stack.

        :param lit_module: A LightningModule instance.
        :param batch: A batch dictionary.
        :param training: Whether the model is in training mode.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary.
        """
        with torch.no_grad():
            batch = self.prepare_protein_patch_indexers(
                batch, randomize_anchors=kwargs["training"]
            )
            self.prepare_protein_backbone_indexers(batch, **kwargs)
            self.initialize_protein_embeddings(batch)
            self.initialize_protatm_indexer_and_embeddings(batch)

        batch = self.protein_encoder(
            batch,
            in_attr_suffix="",
            out_attr_suffix="_projected",
            **kwargs,
        )
        if batch["misc"]["protein_only"]:
            return batch

        # NOTE: here, we are assuming a static ligand graph
        if "lig_atom_attr" not in batch["features"]:
            self.initialize_ligand_embeddings(batch, **kwargs)
        return batch

    def run_contact_map_stack(
        self,
        batch: MODEL_BATCH,
        iter_id: Union[int, str],
        observed_block_contacts: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> MODEL_BATCH:
        """Run the contact map stack.

        :param batch: A batch dictionary.
        :param iter_id: The current iteration ID.
        :param observed_block_contacts: Optional observed block contacts.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary.
        """
        features = batch["features"]
        device = features["res_type"].device
        if observed_block_contacts is not None:
            # Merge into 8AA blocks and gather to patches
            patch8_idx = (
                torch.arange(
                    observed_block_contacts.shape[1],
                    device=device,
                )
                // 8
            )
            merged_contacts_reswise = (
                segment_sum(
                    observed_block_contacts.transpose(0, 1).contiguous(),
                    patch8_idx,
                    max(patch8_idx) + 1,
                )
                .bool()[patch8_idx]
                .transpose(0, 1)
                .contiguous()
            )
            merged_contacts_gathered = (
                merged_contacts_reswise.flatten(0, 1)[batch["indexer"]["gather_idx_pid_a"]]
                .contiguous()
                .view(
                    observed_block_contacts.shape[0],
                    -1,
                    observed_block_contacts.shape[2],
                )
            )
            block_contact_embedding = self.contact_code_embed(merged_contacts_gathered.long())
        else:
            block_contact_embedding = None
        batch = self.pl_contact_stack(
            batch,
            in_attr_suffix="_projected",
            out_attr_suffix=f"_out_{iter_id}",
            observed_block_contacts=block_contact_embedding,
        )

        if batch["misc"]["protein_only"]:
            return batch

        metadata = batch["metadata"]
        batch_size = metadata["num_structid"]
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_I_per_sample = metadata["n_lig_patches_per_sample"]
        res_lig_pair_attr = batch["features"][f"res_trp_pair_attr_flat_out_{iter_id}"]
        raw_dgram_logits = self.dgram_head(res_lig_pair_attr).view(
            batch_size, n_a_per_sample, n_I_per_sample, 32
        )
        batch["outputs"][f"res_lig_distogram_out_{iter_id}"] = F.log_softmax(
            raw_dgram_logits, dim=-1
        )
        return batch

    def infer_geometry_prior(
        self,
        batch: MODEL_BATCH,
        cached_block_contacts: Optional[torch.Tensor] = None,
        binding_site_mask: Optional[torch.Tensor] = None,
        logit_clamp_value: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ):
        """Infer a geometry prior.

        :param batch: A batch dictionary.
        :param cached_block_contacts: Cached block contacts.
        :param binding_site_mask: Binding site mask.
        :param logit_clamp_value: Logit clamp value.
        """
        # Parse self.task_cfg.block_contact_decoding_scheme
        assert (
            self.global_cfg.block_contact_decoding_scheme == "beam"
        ), "Only beam search is supported."
        n_lig_frames = max(batch["metadata"]["num_I_per_sample"])
        # Autoregressive block-contact sampling
        if cached_block_contacts is None:
            # Start from the prior distribution
            sampled_block_contacts = None
            last_distogram = batch["outputs"]["res_lig_distogram_out_0"]
            for iter_id in tqdm.tqdm(range(n_lig_frames), desc="Block contact sampling"):
                last_contact_map = distogram_to_gaussian_contact_logits(
                    last_distogram, self.dist_bins, self.CONTACT_SCALE
                )
                sampled_block_contacts = sample_reslig_contact_matrix(
                    batch, last_contact_map, last=sampled_block_contacts
                ).detach()
                self.run_contact_map_stack(
                    batch, iter_id, observed_block_contacts=sampled_block_contacts
                )
                last_distogram = batch["outputs"][f"res_lig_distogram_out_{iter_id}"]
            batch["outputs"]["sampled_block_contacts_last"] = sampled_block_contacts
            # Check that all ligands are assigned to one protein chain segment
            num_assigned_per_lig = segment_sum(
                torch.sum(sampled_block_contacts, dim=1).contiguous().flatten(0, 1),
                batch["indexer"]["gather_idx_I_molid"],
                batch["metadata"]["num_molid"],
            )
            assert torch.all(num_assigned_per_lig >= 1)
        else:
            sampled_block_contacts = cached_block_contacts

        # Use the cached contacts and only sample once
        self.run_contact_map_stack(
            batch, n_lig_frames, observed_block_contacts=sampled_block_contacts
        )
        last_distogram = batch["outputs"][f"res_lig_distogram_out_{n_lig_frames}"]
        res_lig_contact_logit_pred = distogram_to_gaussian_contact_logits(
            last_distogram, self.dist_bins, self.CONTACT_SCALE
        )

        if binding_site_mask is not None:
            res_lig_contact_logit_pred = res_lig_contact_logit_pred - (
                ~binding_site_mask[:, :, None] * 1e9
            )
        if not kwargs["training"] and logit_clamp_value is not None:
            res_lig_contact_logit_pred = (
                res_lig_contact_logit_pred - (res_lig_contact_logit_pred < logit_clamp_value) * 1e9
            )
        batch["outputs"]["geometry_prior_L"] = res_lig_contact_logit_pred.flatten()

    def init_randexp_kNN_edges_and_covmask(
        self, batch: MODEL_BATCH, detect_covalent: bool = False
    ):
        """Initialize random expansion kNN edges and covalent mask.

        :param batch: A batch dictionary.
        :param detect_covalent: Whether to detect covalent bonds.
        """
        device = batch["features"]["res_type"].device
        batch_size = batch["metadata"]["num_structid"]
        protatm_padding_mask = batch["features"]["res_atom_mask"]
        prot_atm_coords_padded = batch["features"]["input_protein_coords"]
        protatm_coords = prot_atm_coords_padded[protatm_padding_mask].contiguous()
        n_protatm_per_sample = batch["metadata"]["num_protatm_per_sample"]
        protatm_coords = protatm_coords.view(batch_size, n_protatm_per_sample, 3)
        if not batch["misc"]["protein_only"]:
            n_ligatm_per_sample = max(batch["metadata"]["num_i_per_sample"])
            ligatm_coords = batch["features"]["input_ligand_coords"]
            ligatm_coords = ligatm_coords.view(batch_size, n_ligatm_per_sample, 3)
            atm_coords = torch.cat([protatm_coords, ligatm_coords], dim=1)
        else:
            atm_coords = protatm_coords
        distance_mat = torch.norm(atm_coords[:, :, None] - atm_coords[:, None, :], dim=-1)
        distance_mat[distance_mat == 0] = 1e9
        knn_edge_mask = topk_edge_mask_from_logits(
            -distance_mat / self.CONTACT_SCALE,
            self.cfg.score_head.max_atom_degree,
            randomize=True,
        )
        if (not batch["misc"]["protein_only"]) and detect_covalent:
            prot_atomic_numbers = batch["features"]["protatm_to_atomic_number"].view(
                batch_size, n_protatm_per_sample
            )
            lig_atomic_numbers = (
                batch["features"]["atomic_numbers"].long().view(batch_size, n_ligatm_per_sample)
            )
            atom_vdw = self.atnum2vdw[torch.cat([prot_atomic_numbers, lig_atomic_numbers], dim=1)]
            average_vdw = (atom_vdw[:, :, None] + atom_vdw[:, None, :]) / 2
            intermol_iscov_mask = distance_mat < average_vdw * 1.3
            intermol_iscov_mask[:, :n_protatm_per_sample, :n_protatm_per_sample] = False
            gather_idx_i_molid = batch["indexer"]["gather_idx_i_molid"].view(
                batch_size, n_ligatm_per_sample
            )
            lig_samemol_mask = gather_idx_i_molid[:, :, None] == gather_idx_i_molid[:, None, :]
            intermol_iscov_mask[
                :, n_protatm_per_sample:, n_protatm_per_sample:
            ] = intermol_iscov_mask[:, n_protatm_per_sample:, n_protatm_per_sample:] & (
                ~lig_samemol_mask
            )
            knn_edge_mask = knn_edge_mask | intermol_iscov_mask
        else:
            intermol_iscov_mask = torch.zeros_like(distance_mat, dtype=torch.bool)
        p_idx = torch.arange(batch_size * n_protatm_per_sample, device=device).view(
            batch_size, n_protatm_per_sample
        )
        pp_edge_mask = knn_edge_mask[:, :n_protatm_per_sample, :n_protatm_per_sample]
        batch["indexer"]["knn_idx_protatm_protatm_src"] = (
            p_idx[:, :, None].expand(-1, -1, n_protatm_per_sample).contiguous()[pp_edge_mask]
        )
        batch["indexer"]["knn_idx_protatm_protatm_dst"] = (
            p_idx[:, None, :].expand(-1, n_protatm_per_sample, -1).contiguous()[pp_edge_mask]
        )
        batch["features"]["knn_feat_protatm_protatm"] = self.covalent_embed(
            intermol_iscov_mask[:, :n_protatm_per_sample, :n_protatm_per_sample][
                pp_edge_mask
            ].long()
        )
        if not batch["misc"]["protein_only"]:
            l_idx = torch.arange(batch_size * n_ligatm_per_sample, device=device).view(
                batch_size, n_ligatm_per_sample
            )
            pl_edge_mask = knn_edge_mask[:, :n_protatm_per_sample, n_protatm_per_sample:]
            batch["indexer"]["knn_idx_protatm_ligatm_src"] = (
                p_idx[:, :, None].expand(-1, -1, n_ligatm_per_sample).contiguous()[pl_edge_mask]
            )
            batch["indexer"]["knn_idx_protatm_ligatm_dst"] = (
                l_idx[:, None, :].expand(-1, n_protatm_per_sample, -1).contiguous()[pl_edge_mask]
            )
            batch["features"]["knn_feat_protatm_ligatm"] = self.covalent_embed(
                intermol_iscov_mask[:, :n_protatm_per_sample, n_protatm_per_sample:][
                    pl_edge_mask
                ].long()
            )
            lp_edge_mask = knn_edge_mask[:, n_protatm_per_sample:, :n_protatm_per_sample]
            batch["indexer"]["knn_idx_ligatm_protatm_src"] = (
                l_idx[:, :, None].expand(-1, -1, n_protatm_per_sample).contiguous()[lp_edge_mask]
            )
            batch["indexer"]["knn_idx_ligatm_protatm_dst"] = (
                p_idx[:, None, :].expand(-1, n_ligatm_per_sample, -1).contiguous()[lp_edge_mask]
            )
            batch["features"]["knn_feat_ligatm_protatm"] = self.covalent_embed(
                intermol_iscov_mask[:, n_protatm_per_sample:, :n_protatm_per_sample][
                    lp_edge_mask
                ].long()
            )
            ll_edge_mask = knn_edge_mask[:, n_protatm_per_sample:, n_protatm_per_sample:]
            batch["indexer"]["knn_idx_ligatm_ligatm_src"] = (
                l_idx[:, :, None].expand(-1, -1, n_ligatm_per_sample).contiguous()[ll_edge_mask]
            )
            batch["indexer"]["knn_idx_ligatm_ligatm_dst"] = (
                l_idx[:, None, :].expand(-1, n_ligatm_per_sample, -1).contiguous()[ll_edge_mask]
            )
            batch["features"]["knn_feat_ligatm_ligatm"] = self.covalent_embed(
                intermol_iscov_mask[:, n_protatm_per_sample:, n_protatm_per_sample:][
                    ll_edge_mask
                ].long()
            )

    def init_esdm_inputs(
        self, batch: MODEL_BATCH, embedding_iter_id: Union[int, str]
    ) -> MODEL_BATCH:
        """Initialize the inputs for the ESDM.

        :param batch: A batch dictionary.
        :param embedding_iter_id: The embedding iteration ID.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary.
        """
        with torch.no_grad():
            self.init_randexp_kNN_edges_and_covmask(
                batch,
                detect_covalent=self.global_cfg.detect_covalent,
            )
        batch["features"]["rec_res_attr_decin"] = batch["features"][
            f"rec_res_attr_out_{embedding_iter_id}"
        ]
        batch["features"]["res_res_pair_attr_decin"] = batch["features"][
            f"res_res_pair_attr_out_{embedding_iter_id}"
        ]
        batch["features"]["res_res_grid_attr_flat_decin"] = batch["features"][
            f"res_res_grid_attr_flat_out_{embedding_iter_id}"
        ]
        if batch["misc"]["protein_only"]:
            return batch
        batch["features"]["lig_trp_attr_decin"] = batch["features"][
            f"lig_trp_attr_out_{embedding_iter_id}"
        ]
        # Use protein-ligand edges from the contact predictor
        batch["features"]["res_trp_grid_attr_flat_decin"] = batch["features"][
            f"res_trp_grid_attr_flat_out_{embedding_iter_id}"
        ]
        batch["features"]["res_trp_pair_attr_flat_decin"] = batch["features"][
            f"res_trp_pair_attr_flat_out_{embedding_iter_id}"
        ]
        batch["features"]["trp_trp_grid_attr_flat_decin"] = batch["features"][
            f"trp_trp_grid_attr_flat_out_{embedding_iter_id}"
        ]
        return batch

    def run_score_head(
        self,
        batch: MODEL_BATCH,
        embedding_iter_id: Union[int, str],
        frozen_lig: Optional[bool] = None,
        frozen_prot: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> MODEL_BATCH:
        """Run the score head.

        :param batch: A batch dictionary.
        :param embedding_iter_id: The embedding iteration ID.
        :param frozen_lig: Whether to freeze the ligand backbone.
        :param frozen_prot: Whether to freeze the protein backbone.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary with the score head output.
        """
        batch = self.init_esdm_inputs(batch, embedding_iter_id)
        return self.score_head(
            batch,
            frozen_lig=frozen_lig
            if frozen_lig is not None
            else self.global_cfg.frozen_ligand_backbone,
            frozen_prot=frozen_prot
            if frozen_prot is not None
            else self.global_cfg.frozen_protein_backbone,
            **kwargs,
        )

    def run_confidence_head(self, batch: MODEL_BATCH, **kwargs: Dict[str, Any]) -> MODEL_BATCH:
        """Run the confidence head.

        :param batch: A batch dictionary.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary with the confidence head output.
        """
        return self.confidence_head(batch, frozen_lig=False, frozen_prot=False, **kwargs)

    def run_affinity_head(self, batch: MODEL_BATCH, **kwargs: Dict[str, Any]) -> MODEL_BATCH:
        """Run the affinity head.

        :param batch: A batch dictionary.
        :param kwargs: Additional keyword arguments.
        :return: Affinity head output.
        """
        aff_out = self.affinity_head(batch, frozen_lig=False, frozen_prot=False, **kwargs)
        aff_pooled = self.ligand_pooling(
            aff_out["final_embedding_lig_atom"][:, 0],
            batch["indexer"]["gather_idx_i_molid"],
            batch["metadata"]["num_molid"],
        )
        return self.affinity_proj_head(aff_pooled).squeeze(1)

    def run_auxiliary_estimation(
        self,
        batch: MODEL_BATCH,
        struct: MODEL_BATCH,
        return_avg_stats: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[MODEL_BATCH, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Run auxiliary estimations.

        :param batch: A batch dictionary.
        :param struct: A batch dictionary.
        :param return_avg_stats: Whether to return average statistics.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary or a tuple of average (optional) statistics.
        """
        batch_size = batch["metadata"]["num_structid"]
        batch["features"]["input_protein_coords"] = struct["receptor_padded"].clone()
        if struct["ligands"] is not None:
            batch["features"]["input_ligand_coords"] = struct["ligands"].clone()
        else:
            batch["features"]["input_ligand_coords"] = None
        self.assign_timestep_encodings(batch, 0.0)
        batch = self.run_encoder_stack(batch, use_template=False, use_plddt=False, **kwargs)
        self.run_contact_map_stack(batch, iter_id="auxiliary")
        batch = self.init_esdm_inputs(batch, "auxiliary")
        if self.affinity_cfg.enabled:
            batch["outputs"]["affinity_logits"] = self.run_affinity_head(batch)
        if not self.confidence_cfg.enabled:
            return batch
        conf_out = self.run_confidence_head(batch)
        conf_rep = (
            conf_out["final_embedding_prot_res"][:, 0]
            .contiguous()
            .view(batch_size, -1, self.cfg.confidence.fiber_dim)
        )
        if struct["ligands"] is not None:
            conf_rep_lig = (
                conf_out["final_embedding_lig_atom"][:, 0]
                .contiguous()
                .view(batch_size, -1, self.cfg.confidence.fiber_dim)
            )
            conf_rep = torch.cat([conf_rep, conf_rep_lig], dim=1)
        plddt_logits = F.log_softmax(self.plddt_gram_head(conf_rep), dim=-1)
        batch["outputs"]["plddt_logits"] = plddt_logits
        plddt_gram = torch.exp(plddt_logits)
        batch["outputs"]["plddt"] = torch.cumsum(plddt_gram[:, :, :4], dim=-1).mean(dim=-1)

        if return_avg_stats:
            plddt_avg = (batch["outputs"]["plddt"].view(batch_size, -1).mean(dim=1)).detach()
            if struct["ligands"] is not None:
                plddt_avg_lig = (
                    batch["outputs"]["plddt"]
                    .view(batch_size, -1)[:, batch["metadata"]["num_a_per_sample"][0] :]
                    .mean(dim=1)
                    .detach()
                )
                plddt_avg_ligs = segment_mean(
                    batch["outputs"]["plddt"]
                    .view(batch_size, -1)[:, batch["metadata"]["num_a_per_sample"][0] :]
                    .reshape(-1),
                    batch["indexer"]["gather_idx_i_molid"],
                    batch["metadata"]["num_molid"],
                ).detach()
            else:
                plddt_avg_lig = None
                plddt_avg_ligs = None
            return plddt_avg, plddt_avg_lig, plddt_avg_ligs

        return batch

    @staticmethod
    def reverse_interp_ode_step(
        x0_hat: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Reverse process sampling using an Euler ODE solver.

        :param x0_hat: The denoised state.
        :param xt: The intermediate (noisy) state.
        :param t: The current timestep.
        :param s: The next timestep after `t`.
        :return: The interpolated state.
        """
        step_size = t - s
        return xt + step_size * x0_hat

    @staticmethod
    def reverse_interp_vdode_step(
        x0_hat: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """Reverse process sampling using an Euler Variance Diminishing (VD) ODE solver.

        Note that the LHS and RHS time step scaling factors are clamped to the range
        `[1e-6, 1 - 1e-6]` by default.

        :param x0_hat: The denoised state.
        :param xt: The intermediate (noisy) state.
        :param t: The current timestep.
        :param s: The next timestep after `t`.
        :param eta: The variance diminishing factor to employ.
        :return: The interpolated state.
        """
        return clamp_tensor(1 - ((s / t) * eta)) * x0_hat + clamp_tensor((s / t) * eta) * xt

    def reverse_interp_plcomplex_latinp(
        self,
        batch: MODEL_BATCH,
        t: torch.Tensor,
        s: torch.Tensor,
        latent_converter: LatentCoordinateConverter,
        score_converter: LatentCoordinateConverter,
        sampler_step_fn: Callable,
        umeyama_correction: bool = True,
        use_template: bool = False,
    ) -> MODEL_BATCH:
        """Reverse-interpolate protein-ligand complex latent internal coordinates.

        :param batch: A batch dictionary.
        :param t: The current timestep.
        :param s: The next timestep after `t`.
        :param latent_converter: The latent coordinate converter.
        :param score_converter: The latent score converter.
        :param sampler_step_fn: The sampling step function to use.
        :param umeyama_correction: Whether to apply the Umeyama correction.
        :param use_template: Whether to use a given protein structure template.
        :return: A batch dictionary.
        """
        batch_size = batch["metadata"]["num_structid"]

        # derive dimension-less internal coordinates
        x_int_t = latent_converter.to_latent(batch)
        (
            x_int_t_ca_lat,
            x_int_t_apo_ca_lat,
            x_int_t_cother_lat,
            x_int_t_apo_cother_lat,
            x_int_t_ca_lat_centroid_coords,
            x_int_t_apo_ca_lat_centroid_coords,
            x_int_t_lig_lat,
        ) = torch.split(
            x_int_t,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_int_t_ = torch.cat(
            [
                x_int_t_ca_lat,
                x_int_t_cother_lat,
                x_int_t_lig_lat,
            ],
            dim=1,
        )
        self.assign_timestep_encodings(batch, t)

        batch = self.forward(
            batch,
            iter_id=score_converter.iter_id,
            contact_prediction=True,
            score=True,
            observed_block_contacts=score_converter.sampled_block_contacts,
            use_template=use_template,
            training=False,
        )
        if umeyama_correction:
            _last_pred_ca_trace = (
                batch["outputs"]["denoised_prediction"]["final_coords_prot_atom_padded"][:, 1]
                .view(batch_size, -1, 3)
                .detach()
            )
            if score_converter._last_pred_ca_trace is not None:
                similarity_transform = corresponding_points_alignment(
                    _last_pred_ca_trace,
                    score_converter._last_pred_ca_trace,
                    estimate_scale=False,
                )
                _last_pred_ca_trace = apply_similarity_transform(
                    _last_pred_ca_trace, *similarity_transform
                )
                protatm_padding_mask = batch["features"]["res_atom_mask"]
                pred_protatm_coords = (
                    batch["outputs"]["denoised_prediction"]["final_coords_prot_atom_padded"][
                        protatm_padding_mask
                    ]
                    .contiguous()
                    .view(batch_size, -1, 3)
                )
                aligned_pred_protatm_coords = (
                    apply_similarity_transform(pred_protatm_coords, *similarity_transform)
                    .contiguous()
                    .flatten(0, 1)
                )
                batch["outputs"]["denoised_prediction"]["final_coords_prot_atom_padded"][
                    protatm_padding_mask
                ] = aligned_pred_protatm_coords
                batch["outputs"]["denoised_prediction"][
                    "final_coords_prot_atom"
                ] = aligned_pred_protatm_coords
                if not batch["misc"]["protein_only"]:
                    pred_ligatm_coords = batch["outputs"]["denoised_prediction"][
                        "final_coords_lig_atom"
                    ].view(batch_size, -1, 3)
                    aligned_pred_ligatm_coords = (
                        apply_similarity_transform(pred_ligatm_coords, *similarity_transform)
                        .contiguous()
                        .flatten(0, 1)
                    )
                    batch["outputs"]["denoised_prediction"][
                        "final_coords_lig_atom"
                    ] = aligned_pred_ligatm_coords
            score_converter._last_pred_ca_trace = _last_pred_ca_trace

        # interpolate in the latent (reduced) coordinates space
        x_int_hat_t = score_converter.to_latent(batch)
        (
            x_int_hat_t_ca_lat,
            _,
            x_int_hat_t_cother_lat,
            _,
            _,
            _,
            x_int_hat_t_lig_lat,
        ) = torch.split(
            x_int_hat_t,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_int_hat_t_ = torch.cat(
            [
                x_int_hat_t_ca_lat,
                x_int_hat_t_cother_lat,
                x_int_hat_t_lig_lat,
            ],
            dim=1,
        )

        # only interpolate using Ca atom, non-Ca atom, and ligand atom coordinates
        x_int_tm = sampler_step_fn(x_int_hat_t_, x_int_t_, t, s)

        # reassemble outputs
        (
            x_int_tm_ca_lat,
            x_int_tm_cother_lat,
            x_int_tm_lig_lat,
        ) = torch.split(
            x_int_tm,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_int_tm_ = torch.cat(
            [
                x_int_tm_ca_lat,
                x_int_t_apo_ca_lat,
                x_int_tm_cother_lat,
                x_int_t_apo_cother_lat,
                x_int_t_ca_lat_centroid_coords,
                x_int_t_apo_ca_lat_centroid_coords,
                x_int_tm_lig_lat,
            ],
            dim=1,
        )
        return latent_converter.assign_to_batch(batch, x_int_tm_)

    def sample_pl_complex_structures(
        self,
        batch: MODEL_BATCH,
        num_steps: int = 100,
        return_summary_stats: int = False,
        return_all_states: bool = False,
        sampler: Literal["ODE", "VDODE"] = "VDODE",
        sampler_eta: float = 1.0,
        umeyama_correction: bool = True,
        start_time: float = 1.0,
        exact_prior: bool = False,
        eval_input_protein: bool = False,
        align_to_ground_truth: bool = True,
        use_template: Optional[bool] = None,
        **kwargs,
    ) -> MODEL_BATCH:
        """Sample protein-ligand complex structures.

        :param batch: A batch dictionary.
        :param num_steps: The number of steps.
        :param return_summary_stats: Whether to return summary statistics.
        :param return_all_states: Whether to return all states along with sampling metrics.
        :param sampler: The reverse process sampler to use.
        :param sampler_eta: The variance diminishing factor to employ for the `VDODE` sampler,
            which offers a trade-off between exploration (1.0) and exploitation (> 1.0).
        :param umeyama_correction: Apply optimal alignment between the denoised structure and
            previous step outputs.
        :param start_time: The start time.
        :param exact_prior: Whether to use the exact prior.
        :param eval_input_protein: Whether to evaluate the input protein structure.
        :param align_to_ground_truth: Whether to align to the ground truth.
        :param use_template: Whether to use a given protein structure template.
        :param kwargs: Additional keyword arguments.
        :return: A batch dictionary.
        """
        assert num_steps > 0, "Invalid number of steps."
        assert 0.0 <= start_time <= 1.0, "Invalid start time."
        if use_template is None:
            use_template = self.global_cfg.use_template

        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        res_atom_mask = batch["features"]["res_atom_mask"].bool()
        device = features["res_type"].device
        batch_size = metadata["num_structid"]

        if "num_molid" in batch["metadata"].keys() and batch["metadata"]["num_molid"] > 0:
            batch["misc"]["protein_only"] = False
        else:
            batch["misc"]["protein_only"] = True

        forward_lat_converter = self.resolve_latent_converter(
            [
                ("features", "res_atom_positions"),
                ("features", "input_protein_coords"),
            ],
            [("features", "sdf_coordinates"), ("features", "input_ligand_coords")],
        )
        reverse_lat_converter = self.resolve_latent_converter(
            [
                ("features", "input_protein_coords"),
                ("features", "input_protein_coords"),
            ],
            [
                ("features", "input_ligand_coords"),
                ("features", "input_ligand_coords"),
            ],
        )
        reverse_score_converter = self.resolve_latent_converter(
            [
                (
                    "outputs",
                    "denoised_prediction",
                    "final_coords_prot_atom_padded",
                ),
                None,
            ],
            [
                (
                    "outputs",
                    "denoised_prediction",
                    "final_coords_lig_atom",
                ),
                None,
            ],
        )

        with torch.no_grad():
            if not batch["misc"]["protein_only"]:
                # Autoregressive block contact map prior
                if exact_prior:
                    batch = self.prepare_protein_patch_indexers(batch)
                    _, contact_logit_matrix = eval_true_contact_maps(
                        batch, self.CONTACT_SCALE, **kwargs
                    )
                else:
                    batch = self.forward_interp_plcomplex_latinp(
                        batch,
                        start_time,
                        forward_lat_converter,
                        erase_data=(start_time >= 1.0),
                    )
                    self.assign_timestep_encodings(batch, start_time)
                    # Sample the categorical contact encodings under the hood
                    batch = self.forward(
                        batch,
                        contact_prediction=True,
                        infer_geometry_prior=True,
                        use_template=use_template,
                        training=False,
                    )
                    # Sample initial ligand coordinates from the geometry prior
                    contact_logit_matrix = batch["outputs"]["geometry_prior_L"]

                sampled_lig_res_anchor_mask = sample_res_rowmask_from_contacts(
                    batch, contact_logit_matrix, self.global_cfg.single_protein_batch
                )
                num_cont_to_sample = max(metadata["num_I_per_sample"])
                sampled_block_contacts = None
                for _ in range(num_cont_to_sample):
                    sampled_block_contacts = sample_reslig_contact_matrix(
                        batch, contact_logit_matrix, last=sampled_block_contacts
                    )
                forward_lat_converter.lig_res_anchor_mask = sampled_lig_res_anchor_mask
                reverse_lat_converter.lig_res_anchor_mask = sampled_lig_res_anchor_mask
                reverse_score_converter.lig_res_anchor_mask = sampled_lig_res_anchor_mask
                reverse_score_converter.iter_id = num_cont_to_sample
                reverse_score_converter.sampled_block_contacts = sampled_block_contacts
            else:
                reverse_score_converter.iter_id = 0
                reverse_score_converter.sampled_block_contacts = None

        if sampler == "ODE":
            sampler_step_fn = self.reverse_interp_ode_step
        elif sampler == "VDODE":
            sampler_step_fn = partial(self.reverse_interp_vdode_step, eta=sampler_eta)
        else:
            raise NotImplementedError(f"Reverse process sampler {sampler} not implemented.")

        with torch.no_grad():
            # NOTE: Here, we assume the predicted contacts are robust to a resampling of geometric noise
            batch = self.forward_interp_plcomplex_latinp(
                batch,
                start_time,
                forward_lat_converter,
                erase_data=(start_time >= 1.0),
            )

            if return_all_states:
                all_frames = [
                    {
                        "ligands": batch["features"]["input_ligand_coords"].cpu(),
                        "receptor": batch["features"]["input_protein_coords"][res_atom_mask].cpu(),
                        "receptor_padded": batch["features"]["input_protein_coords"].cpu(),
                    }
                ]
            if eval_input_protein:
                protein_fape_input, _ = compute_fape_from_atom37(
                    batch,
                    device,
                    batch["features"]["input_protein_coords"],
                    batch["features"]["res_atom_positions"],
                )
                tm_lbound_input = compute_TMscore_lbound(
                    batch,
                    batch["features"]["input_protein_coords"],
                    batch["features"]["res_atom_positions"],
                )
                tm_lbound_mirrored_input = compute_TMscore_lbound(
                    batch,
                    -batch["features"]["input_protein_coords"],
                    batch["features"]["res_atom_positions"],
                )
                tm_aligned_ca_input = compute_TMscore_raw(
                    batch,
                    batch["features"]["input_protein_coords"][:, 1],
                    batch["features"]["res_atom_positions"][:, 1],
                )
                lddt_ca_input = compute_lddt_ca(
                    batch,
                    batch["features"]["input_protein_coords"],
                    batch["features"]["res_atom_positions"],
                )
                input_ret = {
                    "FAPE_protein_input": protein_fape_input,
                    "TM_aligned_ca_input": tm_aligned_ca_input,
                    "TM_lbound_input": tm_lbound_input,
                    "TM_lbound_mirrored_input": tm_lbound_mirrored_input,
                    "lDDT-Ca_input": lddt_ca_input,
                }
            # NOTE: We follow https://arxiv.org/pdf/2402.04845.pdf for all symbolic conventions
            schedule = torch.linspace(start_time, 0, num_steps + 1, device=device)
            for t, s in tqdm.tqdm(
                zip(schedule[:-1], schedule[1:]), desc=f"Structure generation using {sampler}"
            ):
                batch = self.reverse_interp_plcomplex_latinp(
                    batch,
                    t[None, None],
                    s[None, None],
                    reverse_lat_converter,
                    reverse_score_converter,
                    sampler_step_fn,
                    umeyama_correction=umeyama_correction,
                    use_template=use_template,
                )
                if return_all_states:
                    # all_frames.append(
                    #     {
                    #         "ligands": batch["features"]["input_ligand_coords"],
                    #         "receptor": batch["features"]["input_protein_coords"][
                    #             res_atom_mask
                    #         ],
                    #         "receptor_padded": batch["features"][
                    #             "input_protein_coords"
                    #         ],
                    #     }
                    # )
                    all_frames.append(
                        {
                            "ligands": batch["outputs"]["denoised_prediction"][
                                "final_coords_lig_atom"
                            ].cpu(),
                            "receptor": batch["outputs"]["denoised_prediction"][
                                "final_coords_prot_atom"
                            ].cpu(),
                            "receptor_padded": batch["outputs"]["denoised_prediction"][
                                "final_coords_prot_atom_padded"
                            ].cpu(),
                        }
                    )

            mean_x1 = batch["outputs"]["denoised_prediction"]["final_coords_lig_atom"]
            mean_x2_padded = batch["outputs"]["denoised_prediction"][
                "final_coords_prot_atom_padded"
            ]
            protatm_padding_mask = batch["features"]["res_atom_mask"]
            mean_x2 = mean_x2_padded[protatm_padding_mask]
            if align_to_ground_truth:
                similarity_transform = corresponding_points_alignment(
                    mean_x2_padded[:, 1].view(batch_size, -1, 3),
                    batch["features"]["res_atom_positions"][:, 1].view(batch_size, -1, 3),
                    estimate_scale=False,
                )
                mean_x2 = (
                    apply_similarity_transform(
                        mean_x2.view(batch_size, -1, 3), *similarity_transform
                    )
                    .contiguous()
                    .flatten(0, 1)
                )
                mean_x2_padded[protatm_padding_mask] = mean_x2
                if mean_x1 is not None:
                    mean_x1 = (
                        apply_similarity_transform(
                            mean_x1.view(batch_size, -1, 3), *similarity_transform
                        )
                        .contiguous()
                        .flatten(0, 1)
                    )

            if return_all_states:
                all_frames.append(
                    {
                        "ligands": mean_x1.cpu(),
                        "receptor": mean_x2.cpu(),
                        "receptor_padded": mean_x2_padded.cpu(),
                    }
                )
            protein_fape, _ = compute_fape_from_atom37(
                batch,
                device,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_lbound = compute_TMscore_lbound(
                batch,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_lbound_mirrored = compute_TMscore_lbound(
                batch,
                -mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_aligned_ca = compute_TMscore_raw(
                batch,
                mean_x2_padded[:, 1],
                batch["features"]["res_atom_positions"][:, 1],
            )
            lddt_ca = compute_lddt_ca(
                batch,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            ret = {
                "FAPE_protein": protein_fape,
                "TM_aligned_ca": tm_aligned_ca,
                "TM_lbound": tm_lbound,
                "TM_lbound_mirrored": tm_lbound_mirrored,
                "lDDT-Ca": lddt_ca,
            }
            if eval_input_protein:
                ret.update(input_ret)
            if mean_x1 is not None:
                n_I_per_sample = max(metadata["num_I_per_sample"])
                lig_frame_atm_idx = torch.stack(
                    [
                        indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
                        indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
                        indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]][:n_I_per_sample],
                    ],
                    dim=0,
                )
                _, lig_fape, _ = compute_fape_from_atom37(
                    batch,
                    device,
                    mean_x2_padded,
                    batch["features"]["res_atom_positions"],
                    pred_lig_coords=mean_x1,
                    target_lig_coords=batch["features"]["sdf_coordinates"],
                    lig_frame_atm_idx=lig_frame_atm_idx,
                    split_pl_views=True,
                )
                coords_pred_prot = mean_x2_padded[res_atom_mask].view(
                    metadata["num_structid"], -1, 3
                )
                coords_ref_prot = batch["features"]["res_atom_positions"][res_atom_mask].view(
                    metadata["num_structid"], -1, 3
                )
                coords_pred_lig = mean_x1.view(metadata["num_structid"], -1, 3)
                coords_ref_lig = batch["features"]["sdf_coordinates"].view(
                    metadata["num_structid"], -1, 3
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
                lig_centroid_distance = (
                    segment_mean(
                        (coords_pred_lig - coords_pred_prot.mean(dim=1, keepdim=True)).flatten(
                            0, 1
                        ),
                        indexer["gather_idx_i_molid"],
                        metadata["num_molid"],
                    )
                    - segment_mean(
                        (coords_ref_lig - coords_ref_prot.mean(dim=1, keepdim=True)).flatten(0, 1),
                        indexer["gather_idx_i_molid"],
                        metadata["num_molid"],
                    )
                ).norm(dim=-1)
                lig_hit_score_1A = (lig_rmsd < 1.0).float()
                lig_hit_score_2A = (lig_rmsd < 2.0).float()
                lig_hit_score_4A = (lig_rmsd < 4.0).float()
                lddt_pli = compute_lddt_pli(
                    batch,
                    mean_x2_padded,
                    batch["features"]["res_atom_positions"],
                    mean_x1,
                    batch["features"]["sdf_coordinates"],
                )
                ret.update(
                    {
                        "ligand_RMSD": lig_rmsd,
                        "ligand_centroid_distance": lig_centroid_distance,
                        "lDDT-pli": lddt_pli,
                        "FAPE_ligview": lig_fape,
                        "ligand_hit_score_1A": lig_hit_score_1A,
                        "ligand_hit_score_2A": lig_hit_score_2A,
                        "ligand_hit_score_4A": lig_hit_score_4A,
                    }
                )

        if return_summary_stats:
            return ret

        if return_all_states:
            ret.update({"all_frames": all_frames})

        ret.update(
            {
                "ligands": mean_x1,
                "receptor": mean_x2,
                "receptor_padded": mean_x2_padded,
            }
        )
        return ret

    def forward(
        self,
        batch: MODEL_BATCH,
        iter_id: Union[int, str] = 0,
        observed_block_contacts: Optional[torch.Tensor] = None,
        contact_prediction: bool = True,
        infer_geometry_prior: bool = False,
        score: bool = False,
        use_template: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[MODEL_BATCH, torch.Tensor]:
        """Perform a forward pass through the model.

        :param batch: A batch dictionary.
        :param training: Whether the model is in training mode.
        :param iter_id: The current iteration ID.
        :param observed_block_contacts: Observed block contacts.
        :param contact_prediction: Whether to predict contacts.
        :param infer_geometry_prior: Whether to predict using a geometry prior.
        :param score: Whether to predict a denoised complex structure.
        :param use_template: Whether to use a template protein structure.
        :param kwargs: Additional keyword arguments.
        :return: Batch dictionary with outputs or ligand binding affinity.
        """
        prepare_batch(batch)

        batch = self.run_encoder_stack(
            batch,
            use_template=use_template,
            use_plddt=self.global_cfg.use_plddt,
            **kwargs,
        )

        if contact_prediction:
            self.run_contact_map_stack(
                batch,
                iter_id,
                observed_block_contacts=observed_block_contacts,
                **kwargs,
            )

        if infer_geometry_prior:
            assert (
                batch["misc"]["protein_only"] is False
            ), "Only protein-ligand complexes are supported for a geometry prior."
            self.infer_geometry_prior(batch, **kwargs)

        if score:
            batch["outputs"]["denoised_prediction"] = self.run_score_head(
                batch, embedding_iter_id=iter_id, **kwargs
            )

        return batch


if __name__ == "__main__":
    _ = FlowDock()
