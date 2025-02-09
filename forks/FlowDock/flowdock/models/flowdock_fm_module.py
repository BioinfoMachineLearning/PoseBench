import os

import esm
import numpy as np
import rootutils
import torch
from beartype.typing import Any, Dict, Literal, Optional, Union
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    spearman_corrcoef,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.models.components.losses import (
    eval_auxiliary_estimation_losses,
    eval_structure_prediction_losses,
)
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import pdb_filepath_to_protein, prepare_batch
from flowdock.utils.model_utils import extract_esm_embeddings
from flowdock.utils.sampling_utils import multi_pose_sampling
from flowdock.utils.visualization_utils import (
    construct_prot_lig_pairs,
    write_prot_lig_pairs_to_pdb_file,
)

MODEL_BATCH = Dict[str, Any]
MODEL_STAGE = Literal["train", "val", "test", "predict"]
LOSS_MODES_LIST = [
    "structure_prediction",
    "auxiliary_estimation",
    "auxiliary_estimation_without_structure_prediction",
]
LOSS_MODES = Literal[
    "structure_prediction",
    "auxiliary_estimation",
    "auxiliary_estimation_without_structure_prediction",
]
AUX_ESTIMATION_STAGES = ["train", "val", "test"]

log = RankedLogger(__name__, rank_zero_only=True)


class FlowDockFMLitModule(LightningModule):
    """A `LightningModule` for geometric flow matching (FM) with FlowDock.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        cfg: DictConfig,
        **kwargs: Dict[str, Any],
    ):
        """Initialize a `FlowDockFMLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model before training.
        :param cfg: The model configuration.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        # the model along with its hyperparameters
        self.net = net(cfg)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # for validating input arguments
        if self.hparams.cfg.task.loss_mode not in LOSS_MODES_LIST:
            raise ValueError(
                f"Invalid loss mode: {self.hparams.cfg.task.loss_mode}. Must be one of {LOSS_MODES}."
            )

        # for inspecting the model's outputs during validation and testing
        (
            self.training_step_outputs,
            self.validation_step_outputs,
            self.test_step_outputs,
            self.predict_step_outputs,
        ) = (
            [],
            [],
            [],
            [],
        )

    def forward(
        self,
        batch: MODEL_BATCH,
        iter_id: Union[int, str] = 0,
        observed_block_contacts: Optional[torch.Tensor] = None,
        contact_prediction: bool = True,
        infer_geometry_prior: bool = False,
        score: bool = False,
        affinity: bool = True,
        use_template: bool = False,
        **kwargs: Dict[str, Any],
    ) -> MODEL_BATCH:
        """Perform a forward pass through the model.

        :param batch: A batch dictionary.
        :param iter_id: The current iteration ID.
        :param observed_block_contacts: Observed block contacts.
        :param contact_prediction: Whether to predict contacts.
        :param infer_geometry_prior: Whether to predict using a geometry prior.
        :param score: Whether to predict a denoised complex structure.
        :param affinity: Whether to predict ligand binding affinity.
        :param use_template: Whether to use a template protein structure.
        :param kwargs: Additional keyword arguments.
        :return: Batch dictionary with outputs.
        """
        return self.net(
            batch,
            iter_id=iter_id,
            observed_block_contacts=observed_block_contacts,
            contact_prediction=contact_prediction,
            infer_geometry_prior=infer_geometry_prior,
            score=score,
            affinity=affinity,
            use_template=use_template,
            training=self.training,
            **kwargs,
        )

    def model_step(
        self,
        batch: MODEL_BATCH,
        batch_idx: int,
        stage: MODEL_STAGE,
        loss_mode: Optional[LOSS_MODES] = None,
    ) -> MODEL_BATCH:
        """Perform a single model step on a batch of data.

        :param batch: A batch dictionary.
        :param batch_idx: The index of the current batch.
        :param stage: The current model stage (i.e., `train`, `val`, `test`, or `predict`).
        :param loss_mode: The loss mode to use for training.
        :return: Batch dictionary with losses.
        """
        prepare_batch(batch)
        predicting_aux_outputs = (
            self.hparams.cfg.confidence.enabled or self.hparams.cfg.affinity.enabled
        )
        is_aux_loss_stage = stage in AUX_ESTIMATION_STAGES
        is_aux_batch = batch_idx % self.hparams.cfg.task.aux_batch_freq == 0
        struct_pred_loss_mode_requested = (
            loss_mode is not None and loss_mode == "structure_prediction"
        )
        should_eval_aux_loss = (
            predicting_aux_outputs
            and is_aux_loss_stage
            and is_aux_batch
            and not struct_pred_loss_mode_requested
            and (
                not self.hparams.cfg.task.freeze_confidence
                or (
                    not self.hparams.cfg.task.freeze_affinity
                    and batch["features"]["affinity"].any().item()
                )
            )
        )
        eval_aux_loss_mode_requested = (
            predicting_aux_outputs
            and loss_mode is not None
            and "auxiliary_estimation" in loss_mode
        )
        if should_eval_aux_loss or eval_aux_loss_mode_requested:
            return eval_auxiliary_estimation_losses(
                self, batch, stage, loss_mode, training=self.training
            )
        loss_fn = eval_structure_prediction_losses
        return loss_fn(self, batch, batch_idx, self.device, stage, t_1=1.0)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        pass

    def training_step(self, batch: MODEL_BATCH, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch dictionary.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.hparams.cfg.task.overfitting_example_name is not None and not all(
            name == self.hparams.cfg.task.overfitting_example_name
            for name in batch["metadata"]["sample_ID_per_sample"]
        ):
            return None

        try:
            batch = self.model_step(batch, batch_idx, "train")
        except Exception as e:
            log.error(
                f"Failed to perform training step for batch index {batch_idx} due to: {e}. Skipping example."
            )
            return None

        if self.hparams.cfg.affinity.enabled and "affinity_logits" in batch["outputs"]:
            training_outputs = {
                "affinity_logits": batch["outputs"]["affinity_logits"],
                "affinity": batch["features"]["affinity"],
            }
            self.training_step_outputs.append(training_outputs)

        # return loss or backpropagation will fail
        return batch["outputs"]["loss"]

    def on_train_epoch_end(self):
        """Lightning hook that is called when a training epoch ends."""
        if self.hparams.cfg.affinity.enabled and any(
            "affinity_logits" in output for output in self.training_step_outputs
        ):
            affinity_logits = torch.cat(
                [
                    output["affinity_logits"]
                    for output in self.training_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity = torch.cat(
                [
                    output["affinity"]
                    for output in self.training_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity_logits = affinity_logits[~affinity.isnan()]
            affinity = affinity[~affinity.isnan()]
            if affinity.numel() > 1:
                # NOTE: there must be at least two affinity batches to properly score the affinity predictions
                aff_rmse = torch.sqrt(mean_squared_error(affinity_logits, affinity))
                aff_mae = mean_absolute_error(affinity_logits, affinity)
                aff_pearson = pearson_corrcoef(affinity_logits, affinity)
                aff_spearman = spearman_corrcoef(affinity_logits, affinity)
                self.log(
                    "train_affinity/RMSE",
                    aff_rmse.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=False,
                )
                self.log(
                    "train_affinity/MAE",
                    aff_mae.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=False,
                )
                self.log(
                    "train_affinity/Pearson",
                    aff_pearson.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=False,
                )
                self.log(
                    "train_affinity/Spearman",
                    aff_spearman.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=False,
                )
        self.training_step_outputs.clear()  # free memory

    def on_validation_start(self):
        """Lightning hook that is called when validation begins."""
        # create a directory to store model outputs from each validation epoch
        os.makedirs(
            os.path.join(self.trainer.default_root_dir, "validation_epoch_outputs"), exist_ok=True
        )

    def validation_step(self, batch: MODEL_BATCH, batch_idx: int, dataloader_idx: int = 0):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch dictionary.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        if self.hparams.cfg.task.overfitting_example_name is not None and not all(
            name == self.hparams.cfg.task.overfitting_example_name
            for name in batch["metadata"]["sample_ID_per_sample"]
        ):
            return None

        try:
            prepare_batch(batch)
            sampling_stats = self.net.sample_pl_complex_structures(
                batch,
                sampler="VDODE",
                sampler_eta=1.0,
                num_steps=10,
                start_time=1.0,
                exact_prior=False,
                return_all_states=True,
                eval_input_protein=True,
            )
            all_frames = sampling_stats["all_frames"]
            del sampling_stats["all_frames"]
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            sampling_stats = self.net.sample_pl_complex_structures(
                batch,
                sampler="VDODE",
                sampler_eta=1.0,
                num_steps=10,
                start_time=1.0,
                exact_prior=False,
                use_template=False,
            )
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling_notemplate/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            sampling_stats = self.net.sample_pl_complex_structures(
                batch,
                sampler="VDODE",
                sampler_eta=1.0,
                num_steps=10,
                start_time=1.0,
                return_summary_stats=True,
                exact_prior=True,
            )
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling_trueprior/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            batch = self.model_step(batch, batch_idx, "val")
        except Exception as e:
            log.error(
                f"Failed to perform validation step for batch index {batch_idx} of dataloader {dataloader_idx} due to: {e}. Skipping example."
            )
            return None

        # store model outputs for inspection
        validation_outputs = {}
        if self.hparams.cfg.task.visualize_generated_samples:
            validation_outputs = {
                "name": batch["metadata"]["sample_ID_per_sample"],
                "batch_size": batch["metadata"]["num_structid"],
                "aatype": batch["features"]["res_type"].long().cpu().numpy(),
                "res_atom_mask": batch["features"]["res_atom_mask"].cpu().numpy(),
                "protein_coordinates_list": [
                    frame["receptor_padded"].cpu().numpy() for frame in all_frames
                ],
                "ligand_coordinates_list": [
                    frame["ligands"].cpu().numpy() for frame in all_frames
                ],
                "ligand_mol": batch["metadata"]["mol_per_sample"],
                "protein_batch_indexer": batch["indexer"]["gather_idx_a_structid"].cpu().numpy(),
                "ligand_batch_indexer": batch["indexer"]["gather_idx_i_structid"].cpu().numpy(),
                "gt_protein_coordinates": batch["features"]["res_atom_positions"].cpu().numpy(),
                "gt_ligand_coordinates": batch["features"]["sdf_coordinates"].cpu().numpy(),
                "dataloader_idx": dataloader_idx,
            }
        if self.hparams.cfg.affinity.enabled and "affinity_logits" in batch["outputs"]:
            validation_outputs.update(
                {
                    "affinity_logits": batch["outputs"]["affinity_logits"],
                    "affinity": batch["features"]["affinity"],
                    "dataloader_idx": dataloader_idx,
                }
            )
        if validation_outputs:
            self.validation_step_outputs.append(validation_outputs)

    def on_validation_epoch_end(self):
        "Lightning hook that is called when a validation epoch ends."
        if self.hparams.cfg.task.visualize_generated_samples:
            for i, outputs in enumerate(self.validation_step_outputs):
                for batch_index in range(outputs["batch_size"]):
                    prot_lig_pairs = construct_prot_lig_pairs(outputs, batch_index)
                    write_prot_lig_pairs_to_pdb_file(
                        prot_lig_pairs,
                        os.path.join(
                            self.trainer.default_root_dir,
                            "validation_epoch_outputs",
                            f"{outputs['name'][batch_index]}_validation_epoch_{self.current_epoch}_global_step_{self.global_step}_output_{i}_batch_{batch_index}_dataloader_{outputs['dataloader_idx']}.pdb",
                        ),
                    )
        if self.hparams.cfg.affinity.enabled and any(
            "affinity_logits" in output for output in self.validation_step_outputs
        ):
            affinity_logits = torch.cat(
                [
                    output["affinity_logits"]
                    for output in self.validation_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity = torch.cat(
                [
                    output["affinity"]
                    for output in self.validation_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity_logits = affinity_logits[~affinity.isnan()]
            affinity = affinity[~affinity.isnan()]
            if affinity.numel() > 1:
                # NOTE: there must be at least two affinity batches to properly score the affinity predictions
                aff_rmse = torch.sqrt(mean_squared_error(affinity_logits, affinity))
                aff_mae = mean_absolute_error(affinity_logits, affinity)
                aff_pearson = pearson_corrcoef(affinity_logits, affinity)
                aff_spearman = spearman_corrcoef(affinity_logits, affinity)
                self.log(
                    "val_affinity/RMSE",
                    aff_rmse.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "val_affinity/MAE",
                    aff_mae.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "val_affinity/Pearson",
                    aff_pearson.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "val_affinity/Spearman",
                    aff_spearman.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
        self.validation_step_outputs.clear()  # free memory

    def on_test_start(self):
        """Lightning hook that is called when testing begins."""
        # create a directory to store model outputs from each test epoch
        os.makedirs(
            os.path.join(self.trainer.default_root_dir, "test_epoch_outputs"), exist_ok=True
        )

    def test_step(self, batch: MODEL_BATCH, batch_idx: int, dataloader_idx: int = 0):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch dictionary.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        if self.hparams.cfg.task.overfitting_example_name is not None and not all(
            name == self.hparams.cfg.task.overfitting_example_name
            for name in batch["metadata"]["sample_ID_per_sample"]
        ):
            return None

        try:
            prepare_batch(batch)
            if self.hparams.cfg.task.eval_structure_prediction:
                sampling_stats = self.net.sample_pl_complex_structures(
                    batch,
                    sampler=self.hparams.cfg.task.sampler,
                    sampler_eta=self.hparams.cfg.task.sampler_eta,
                    num_steps=self.hparams.cfg.task.num_steps,
                    start_time=self.hparams.cfg.task.start_time,
                    exact_prior=False,
                    return_all_states=True,
                    eval_input_protein=True,
                )
                all_frames = sampling_stats["all_frames"]
                del sampling_stats["all_frames"]
                for metric_name in sampling_stats.keys():
                    log_stat = sampling_stats[metric_name].mean().detach()
                    batch_size = sampling_stats[metric_name].shape[0]
                    self.log(
                        f"test_sampling/{metric_name}",
                        log_stat,
                        on_step=True,
                        on_epoch=True,
                        batch_size=batch_size,
                    )
                sampling_stats = self.net.sample_pl_complex_structures(
                    batch,
                    sampler=self.hparams.cfg.task.sampler,
                    sampler_eta=self.hparams.cfg.task.sampler_eta,
                    num_steps=self.hparams.cfg.task.num_steps,
                    start_time=self.hparams.cfg.task.start_time,
                    exact_prior=False,
                    use_template=False,
                )
                for metric_name in sampling_stats.keys():
                    log_stat = sampling_stats[metric_name].mean().detach()
                    batch_size = sampling_stats[metric_name].shape[0]
                    self.log(
                        f"test_sampling_notemplate/{metric_name}",
                        log_stat,
                        on_step=True,
                        on_epoch=True,
                        batch_size=batch_size,
                    )
                sampling_stats = self.net.sample_pl_complex_structures(
                    batch,
                    sampler=self.hparams.cfg.task.sampler,
                    sampler_eta=self.hparams.cfg.task.sampler_eta,
                    num_steps=self.hparams.cfg.task.num_steps,
                    start_time=self.hparams.cfg.task.start_time,
                    return_summary_stats=True,
                    exact_prior=True,
                )
                for metric_name in sampling_stats.keys():
                    log_stat = sampling_stats[metric_name].mean().detach()
                    batch_size = sampling_stats[metric_name].shape[0]
                    self.log(
                        f"test_sampling_trueprior/{metric_name}",
                        log_stat,
                        on_step=True,
                        on_epoch=True,
                        batch_size=batch_size,
                    )
            batch = self.model_step(
                batch, batch_idx, "test", loss_mode=self.hparams.cfg.task.loss_mode
            )
        except Exception as e:
            log.error(
                f"Failed to perform test step for {batch['metadata']['sample_ID_per_sample']} with batch index {batch_idx} of dataloader {dataloader_idx} due to: {e}."
            )
            raise e

        # store model outputs for inspection
        test_outputs = {}
        if (
            self.hparams.cfg.task.visualize_generated_samples
            and self.hparams.cfg.task.eval_structure_prediction
        ):
            test_outputs.update(
                {
                    "name": batch["metadata"]["sample_ID_per_sample"],
                    "batch_size": batch["metadata"]["num_structid"],
                    "aatype": batch["features"]["res_type"].long().cpu().numpy(),
                    "res_atom_mask": batch["features"]["res_atom_mask"].cpu().numpy(),
                    "protein_coordinates_list": [
                        frame["receptor_padded"].cpu().numpy() for frame in all_frames
                    ],
                    "ligand_coordinates_list": [
                        frame["ligands"].cpu().numpy() for frame in all_frames
                    ],
                    "ligand_mol": batch["metadata"]["mol_per_sample"],
                    "protein_batch_indexer": batch["indexer"]["gather_idx_a_structid"]
                    .cpu()
                    .numpy(),
                    "ligand_batch_indexer": batch["indexer"]["gather_idx_i_structid"]
                    .cpu()
                    .numpy(),
                    "gt_protein_coordinates": batch["features"]["res_atom_positions"]
                    .cpu()
                    .numpy(),
                    "gt_ligand_coordinates": batch["features"]["sdf_coordinates"].cpu().numpy(),
                    "dataloader_idx": dataloader_idx,
                }
            )
        if self.hparams.cfg.affinity.enabled and "affinity_logits" in batch["outputs"]:
            test_outputs.update(
                {
                    "affinity_logits": batch["outputs"]["affinity_logits"],
                    "affinity": batch["features"]["affinity"],
                    "dataloader_idx": dataloader_idx,
                }
            )
        if test_outputs:
            self.test_step_outputs.append(test_outputs)

    def on_test_epoch_end(self):
        """Lightning hook that is called when a test epoch ends."""
        if (
            self.hparams.cfg.task.visualize_generated_samples
            and self.hparams.cfg.task.eval_structure_prediction
        ):
            for i, outputs in enumerate(self.test_step_outputs):
                for batch_index in range(outputs["batch_size"]):
                    prot_lig_pairs = construct_prot_lig_pairs(outputs, batch_index)
                    write_prot_lig_pairs_to_pdb_file(
                        prot_lig_pairs,
                        os.path.join(
                            self.trainer.default_root_dir,
                            "test_epoch_outputs",
                            f"{outputs['name'][batch_index]}_test_epoch_{self.current_epoch}_global_step_{self.global_step}_output_{i}_batch_{batch_index}_dataloader_{outputs['dataloader_idx']}.pdb",
                        ),
                    )
        if self.hparams.cfg.affinity.enabled and any(
            "affinity_logits" in output for output in self.test_step_outputs
        ):
            affinity_logits = torch.cat(
                [
                    output["affinity_logits"]
                    for output in self.test_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity = torch.cat(
                [
                    output["affinity"]
                    for output in self.test_step_outputs
                    if "affinity_logits" in output
                ]
            )
            affinity_logits = affinity_logits[~affinity.isnan()]
            affinity = affinity[~affinity.isnan()]
            if affinity.numel() > 1:
                # NOTE: there must be at least two affinity batches to properly score the affinity predictions
                aff_rmse = torch.sqrt(mean_squared_error(affinity_logits, affinity))
                aff_mae = mean_absolute_error(affinity_logits, affinity)
                aff_pearson = pearson_corrcoef(affinity_logits, affinity)
                aff_spearman = spearman_corrcoef(affinity_logits, affinity)
                self.log(
                    "test_affinity/RMSE",
                    aff_rmse.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "test_affinity/MAE",
                    aff_mae.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "test_affinity/Pearson",
                    aff_pearson.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
                self.log(
                    "test_affinity/Spearman",
                    aff_spearman.detach(),
                    on_epoch=True,
                    batch_size=len(affinity),
                    sync_dist=True,
                )
        self.test_step_outputs.clear()  # free memory

    def on_predict_start(self):
        """Lightning hook that is called when testing begins."""
        # create a directory to store model outputs from each predict epoch
        os.makedirs(
            os.path.join(self.trainer.default_root_dir, "predict_epoch_outputs"), exist_ok=True
        )

        log.info("Loading pretrained ESM model...")
        esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_hub(
            self.hparams.cfg.model.cfg.protein_encoder.esm_version
        )
        self.esm_model = esm_model.eval().float()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.cpu()

        skip_loading_esmfold_weights = (
            # skip loading ESMFold weights if the template protein structure for a single complex input is provided
            self.hparams.cfg.task.csv_path is None
            and self.hparams.cfg.task.input_template is not None
            and os.path.exists(self.hparams.cfg.task.input_template)
        )
        if not skip_loading_esmfold_weights:
            log.info("Loading pretrained ESMFold model...")
            esmfold_model = esm.pretrained.esmfold_v1()
            self.esmfold_model = esmfold_model.eval().float()
            self.esmfold_model.set_chunk_size(self.hparams.cfg.esmfold_chunk_size)
            self.esmfold_model.cpu()

    def predict_step(self, batch: MODEL_BATCH, batch_idx: int, dataloader_idx: int = 0):
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch dictionary.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        rec_path = batch["rec_path"][0]
        ligand_paths = list(
            path[0] for path in batch["lig_paths"]
        )  # unpack a list of (batched) single-element string tuples
        sample_id = batch["sample_id"][0] if "sample_id" in batch else "sample"
        input_template = batch["input_template"][0] if "input_template" in batch else None

        # generate ESM embeddings for the protein
        protein = pdb_filepath_to_protein(rec_path)
        sequences = [
            "".join(np.array(list(chain_seq))[chain_mask])
            for (_, chain_seq, chain_mask) in protein.letter_sequences
        ]
        esm_embeddings = extract_esm_embeddings(
            self.esm_model,
            self.esm_alphabet,
            self.esm_batch_converter,
            sequences,
            device="cpu",
            esm_repr_layer=self.hparams.cfg.model.cfg.protein_encoder.esm_repr_layer,
        )
        sequences_to_embeddings = {
            f"{seq}:{i}": esm_embeddings[i].cpu().numpy() for i, seq in enumerate(sequences)
        }

        # generate initial ESMFold-predicted structure for the protein if a template is not provided
        apo_rec_path = None
        if input_template and os.path.exists(input_template):
            apo_protein = pdb_filepath_to_protein(input_template)
            apo_chain_seq_masked = "".join(
                "".join(np.array(list(chain_seq))[chain_mask])
                for (_, chain_seq, chain_mask) in apo_protein.letter_sequences
            )
            chain_seq_masked = "".join(
                "".join(np.array(list(chain_seq))[chain_mask])
                for (_, chain_seq, chain_mask) in protein.letter_sequences
            )
            if apo_chain_seq_masked != chain_seq_masked:
                log.error(
                    f"Provided template protein structure {input_template} does not match the input protein sequence within {rec_path}. Skipping example {sample_id} at batch index {batch_idx} of dataloader {dataloader_idx}."
                )
                return None
            log.info(f"Starting from provided template protein structure: {input_template}")
            apo_rec_path = input_template
        if apo_rec_path is None and self.hparams.cfg.prior_type == "esmfold":
            esmfold_sequence = ":".join(sequences)
            apo_rec_path = rec_path.replace(".pdb", "_apo.pdb")
            with torch.no_grad():
                esmfold_pdb_output = self.esmfold_model.infer_pdb(esmfold_sequence)
            with open(apo_rec_path, "w") as f:
                f.write(esmfold_pdb_output)

        _, _, _, _, _, all_frames, batch_all, b_factors, plddt_rankings = multi_pose_sampling(
            rec_path,
            ligand_paths,
            self.hparams.cfg,
            self,
            self.hparams.cfg.out_path,
            separate_pdb=self.hparams.cfg.separate_pdb,
            apo_receptor_path=apo_rec_path,
            sample_id=sample_id,
            protein=protein,
            sequences_to_embeddings=sequences_to_embeddings,
            return_all_states=self.hparams.cfg.task.visualize_generated_samples,
            auxiliary_estimation_only=self.hparams.cfg.task.auxiliary_estimation_only,
        )
        # store model outputs for inspection
        if self.hparams.cfg.task.visualize_generated_samples:
            predict_outputs = {
                "name": batch_all["metadata"]["sample_ID_per_sample"],
                "batch_size": batch_all["metadata"]["num_structid"],
                "aatype": batch_all["features"]["res_type"].long().cpu().numpy(),
                "res_atom_mask": batch_all["features"]["res_atom_mask"].cpu().numpy(),
                "protein_coordinates_list": [
                    frame["receptor_padded"].cpu().numpy() for frame in all_frames
                ],
                "ligand_coordinates_list": [
                    frame["ligands"].cpu().numpy() for frame in all_frames
                ],
                "ligand_mol": batch_all["metadata"]["mol_per_sample"],
                "protein_batch_indexer": batch_all["indexer"]["gather_idx_a_structid"]
                .cpu()
                .numpy(),
                "ligand_batch_indexer": batch_all["indexer"]["gather_idx_i_structid"]
                .cpu()
                .numpy(),
                "b_factors": b_factors,
                "plddt_rankings": plddt_rankings,
            }
            self.predict_step_outputs.append(predict_outputs)

    def on_predict_epoch_end(self):
        """Lightning hook that is called when a predict epoch ends."""
        if self.hparams.cfg.task.visualize_generated_samples:
            for i, outputs in enumerate(self.predict_step_outputs):
                for batch_index in range(outputs["batch_size"]):
                    prot_lig_pairs = construct_prot_lig_pairs(outputs, batch_index)
                    ranking = (
                        outputs["plddt_rankings"][batch_index]
                        if "plddt_rankings" in outputs
                        else None
                    )
                    write_prot_lig_pairs_to_pdb_file(
                        prot_lig_pairs,
                        os.path.join(
                            self.hparams.cfg.out_path,
                            "predict_epoch_outputs",
                            f"{outputs['name'][batch_index]}{f'_rank{ranking + 1}' if ranking is not None else ''}_predict_epoch_{self.current_epoch}_global_step_{self.global_step}_output_{i}_batch_{batch_index}.pdb",
                        ),
                    )
        self.predict_step_outputs.clear()  # free memory

    def on_after_backward(self):
        """Skip updates in case of unstable gradients.

        Reference: https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for _, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break
        if not valid_gradients:
            log.warning(
                "Detected `inf` or `nan` values in gradients. Not updating model parameters."
            )
            self.zero_grad()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        """Override the optimizer step to dynamically update the learning rate.

        :param epoch: The current epoch.
        :param batch_idx: The index of the current batch.
        :param optimizer: The optimizer to use for training.
        :param optimizer_closure: The optimizer closure.
        """
        # update params
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)

        # warm up learning rate
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                # NOTE: `self.hparams.optimizer.keywords["lr"]` refers to the optimizer's initial learning rate
                pg["lr"] = lr_scale * self.hparams.optimizer.keywords["lr"]

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        try:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        except TypeError:
            # NOTE: strategies such as DeepSpeed require `params` to instead be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FlowDockFMLitModule(None, None, None, None)
