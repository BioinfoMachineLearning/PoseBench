import numpy as np
import rootutils
import torch
from beartype.typing import Any, Dict, List, Literal, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.combined_dataset import CombinedDataset
from flowdock.data.components.moad_dataset import BindingMOADDataset
from flowdock.data.components.mol_features import collate_samples
from flowdock.data.components.pdbbind_dataset import PDBBindDataset
from flowdock.data.components.pdbsidechain_dataset import PDBSidechainDataset
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    parse_moad_binding_affinity_data_file,
    parse_pdbbind_binding_affinity_data_file,
)

log = RankedLogger(__name__, rank_zero_only=True)

DATA_PHASE = Literal["train", "val", "test"]
VALID_TRAIN_DATASETS = ["pdbbind", "moad", "pdbsidechain"]
VALID_TEST_DATASETS = ["pdbbind", "moad"]


class CombinedDataModule(LightningDataModule):
    """`LightningDataModule` for combined training and test datasets.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_datasets: List[str] = ["pdbbind", "moad", "pdbsidechain"],
        test_datasets: List[str] = ["pdbbind", "moad"],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        stage: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `CombinedDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_datasets: The datasets to use for training. Defaults to `["pdbbind", "moad", "pdbsidechain"]`.
        :param test_datasets: The datasets to use for testing. Defaults to `["pdbbind", "moad"]`.
        :param batch_size: The batch size. Defaults to `16`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        super().__init__()
        load_train_val_datasets = stage is None or (
            stage is not None and stage in ["fit", "validate"]
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # parse binding affinity values
        self.pdbbind_binding_affinity_values_dict = (
            parse_pdbbind_binding_affinity_data_file(
                self.hparams.pdbbind_binding_affinity_values_path
            )
            if "pdbbind" in train_datasets
            else None
        )
        self.moad_binding_affinity_values_dict = (
            parse_moad_binding_affinity_data_file(self.hparams.moad_binding_affinity_values_path)
            if "moad" in train_datasets
            and self.hparams.moad_map_binding_affinities_to_superligands
            else None
        )

        # prepare for dataset(s) to be loaded into rank-shared memory (e.g., when setting `trainer.strategy = "ddp_spawn"`)
        invalid_trainsets = [
            dataset
            for dataset in self.hparams.train_datasets
            if dataset not in VALID_TRAIN_DATASETS
        ]
        if invalid_trainsets:
            raise ValueError(f"Invalid dataset(s) specified for training: {invalid_trainsets}")
        if any(dataset not in VALID_TEST_DATASETS for dataset in self.hparams.test_datasets):
            raise ValueError(
                f"Invalid dataset(s) specified for testing: {self.hparams.test_datasets}"
            )

        if load_train_val_datasets and "pdbbind" in self.hparams.train_datasets:
            pdbbind_train = PDBBindDataset(
                root=self.hparams.pdbbind_dir,
                cache_path=self.hparams.cache_path,
                split_path=self.hparams.split_train,
                esm_embeddings_path=self.hparams.pdbbind_esm_embeddings_path,
                apo_protein_structure_dir=self.hparams.pdbbind_apo_protein_structure_dir,
                protein_file=self.hparams.protein_file,
                require_ligand=self.hparams.overfitting_example_name is not None,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                max_lig_size=self.hparams.max_lig_size,
                num_workers=self.hparams.num_workers,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                a2h_assessment_csv_filepath=self.hparams.pdbbind_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.pdbbind_filter_using_a2h_assessment,
                a2h_min_tmscore=self.hparams.pdbbind_a2h_min_tmscore,
                a2h_max_rmsd=self.hparams.pdbbind_a2h_max_rmsd,
                a2h_min_protein_length=self.hparams.pdbbind_a2h_min_protein_length,
                a2h_max_protein_length=self.hparams.pdbbind_a2h_max_protein_length,
                a2h_min_ligand_length=self.hparams.pdbbind_a2h_min_ligand_length,
                a2h_max_ligand_length=self.hparams.pdbbind_a2h_max_ligand_length,
                binding_affinity_values_dict=self.pdbbind_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
            pdbbind_val = PDBBindDataset(
                root=self.hparams.pdbbind_dir,
                cache_path=self.hparams.cache_path,
                split_path=self.hparams.split_val,
                esm_embeddings_path=self.hparams.pdbbind_esm_embeddings_path,
                apo_protein_structure_dir=self.hparams.pdbbind_apo_protein_structure_dir,
                protein_file=self.hparams.protein_file,
                require_ligand=True,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                max_lig_size=self.hparams.max_lig_size,
                num_workers=self.hparams.num_workers,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                a2h_assessment_csv_filepath=self.hparams.pdbbind_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.pdbbind_filter_using_a2h_assessment,
                a2h_min_tmscore=self.hparams.pdbbind_a2h_min_tmscore,
                a2h_max_rmsd=self.hparams.pdbbind_a2h_max_rmsd,
                a2h_min_protein_length=self.hparams.pdbbind_a2h_min_protein_length,
                a2h_max_protein_length=self.hparams.pdbbind_a2h_max_protein_length,
                a2h_min_ligand_length=self.hparams.pdbbind_a2h_min_ligand_length,
                a2h_max_ligand_length=self.hparams.pdbbind_a2h_max_ligand_length,
                binding_affinity_values_dict=self.pdbbind_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
        if load_train_val_datasets and "moad" in self.hparams.train_datasets:
            moad_train = BindingMOADDataset(
                root=self.hparams.moad_dir,
                dockgen_dir=self.hparams.moad_dockgen_dir,
                clusters_filepath=self.hparams.moad_clusters_filepath,
                cache_path=self.hparams.cache_path,
                split="train",
                max_receptor_size=self.hparams.max_receptor_size,
                remove_promiscuous_targets=self.hparams.remove_promiscuous_targets,
                min_ligand_size=self.hparams.min_ligand_size,
                multiplicity=self.hparams.train_multiplicity,
                unroll_clusters=self.hparams.unroll_clusters,
                esm_embeddings_sequences_path=self.hparams.moad_esm_embeddings_sequences_path,
                esm_embeddings_path=self.hparams.moad_esm_embeddings_path,
                apo_protein_structure_dir=self.hparams.moad_apo_protein_structure_dir,
                require_ligand=self.hparams.overfitting_example_name is not None,
                enforce_timesplit=self.hparams.enforce_timesplit,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                max_lig_size=self.hparams.max_lig_size,
                min_multi_lig_distance=self.hparams.min_multi_lig_distance,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                pdbbind_split_train=self.hparams.pdbbind_split_train,
                pdbbind_split_val=self.hparams.pdbbind_split_val,
                remove_pdbbind=self.hparams.remove_pdbbind,
                split_time=self.hparams.split_time,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                a2h_assessment_csv_filepath=self.hparams.moad_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.moad_filter_using_a2h_assessment,
                a2h_min_tmscore=self.hparams.moad_a2h_min_tmscore,
                a2h_max_rmsd=self.hparams.moad_a2h_max_rmsd,
                a2h_min_protein_length=self.hparams.moad_a2h_min_protein_length,
                a2h_max_protein_length=self.hparams.moad_a2h_max_protein_length,
                a2h_min_ligand_length=self.hparams.moad_a2h_min_ligand_length,
                a2h_max_ligand_length=self.hparams.moad_a2h_max_ligand_length,
                binding_affinity_values_dict=self.moad_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
            moad_val = BindingMOADDataset(
                root=self.hparams.moad_dir,
                dockgen_dir=self.hparams.moad_dockgen_dir,
                clusters_filepath=self.hparams.moad_clusters_filepath,
                cache_path=self.hparams.cache_path,
                split="val",
                multiplicity=self.hparams.val_multiplicity,
                remove_promiscuous_targets=self.hparams.remove_promiscuous_targets,
                dockgen_esm_embeddings_path=self.hparams.moad_dockgen_esm_embeddings_path,
                dockgen_esm_embeddings_sequences_path=self.hparams.moad_dockgen_esm_embeddings_sequences_path,
                dockgen_apo_protein_structure_dir=self.hparams.moad_dockgen_apo_protein_structure_dir,
                unroll_clusters=self.hparams.unroll_clusters,
                require_ligand=True,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                num_workers=self.hparams.num_workers,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                pdbbind_split_train=self.hparams.pdbbind_split_train,
                pdbbind_split_val=self.hparams.pdbbind_split_val,
                remove_pdbbind=self.hparams.remove_pdbbind,
                split_time=self.hparams.split_time,
                dockgen_a2h_assessment_csv_filepath=self.hparams.moad_dockgen_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.moad_filter_using_a2h_assessment,
                # NOTE: the following `a2h` parameters were deduced directly from the DockGen dataset's `a2h` assessment plots
                a2h_min_tmscore=0.6,
                a2h_max_rmsd=6.0,
                a2h_min_protein_length=50,
                a2h_max_protein_length=2000,
                a2h_min_ligand_length=1,
                a2h_max_ligand_length=300,
                binding_affinity_values_dict=self.moad_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
        if load_train_val_datasets and "pdbsidechain" in self.hparams.train_datasets:
            pdbsidechain_train = PDBSidechainDataset(
                root=self.hparams.pdbsidechain_dir,
                cache_path=self.hparams.cache_path,
                split="train",
                esm_embeddings_path=self.hparams.pdbsidechain_esm_embeddings_path,
                esm_embeddings_sequences_path=self.hparams.pdbsidechain_esm_embeddings_sequences_path,
                apo_protein_structure_dir=self.hparams.pdbsidechain_apo_protein_structure_dir,
                pdbsidechain_metadata_dir=self.hparams.pdbsidechain_metadata_dir,
                multiplicity=self.hparams.train_multiplicity,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                num_workers=self.hparams.num_workers,
                vandermers_max_dist=self.hparams.vandermers_max_dist,
                vandermers_buffer_residue_num=self.hparams.vandermers_buffer_residue_num,
                vandermers_min_contacts=self.hparams.vandermers_min_contacts,
                vandermers_max_surrogate_binding_affinity=self.hparams.vandermers_max_surrogate_binding_affinity,
                vandermers_second_ligand_max_closeness=self.hparams.vandermers_second_ligand_max_closeness,
                vandermers_extract_second_ligand=self.hparams.vandermers_extract_second_ligand,
                merge_clusters=self.hparams.merge_clusters,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                vandermers_use_prob_as_surrogate_binding_affinity=self.hparams.vandermers_use_prob_as_surrogate_binding_affinity,
                a2h_assessment_csv_filepath=self.hparams.pdbsidechain_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.pdbsidechain_filter_using_a2h_assessment,
                a2h_min_tmscore=self.hparams.pdbsidechain_a2h_min_tmscore,
                a2h_max_rmsd=self.hparams.pdbsidechain_a2h_max_rmsd,
                a2h_min_protein_length=self.hparams.pdbsidechain_a2h_min_protein_length,
                a2h_max_protein_length=self.hparams.pdbsidechain_a2h_max_protein_length,
                postprocess_min_protein_length=self.hparams.pdbsidechain_postprocess_min_protein_length,
                postprocess_max_protein_length=self.hparams.pdbsidechain_postprocess_max_protein_length,
                n_lig_patches=self.hparams.n_lig_patches,
            )
            pdbsidechain_val = PDBSidechainDataset(
                root=self.hparams.pdbsidechain_dir,
                cache_path=self.hparams.cache_path,
                split="val",
                esm_embeddings_path=self.hparams.pdbsidechain_esm_embeddings_path,
                esm_embeddings_sequences_path=self.hparams.pdbsidechain_esm_embeddings_sequences_path,
                apo_protein_structure_dir=self.hparams.pdbsidechain_apo_protein_structure_dir,
                pdbsidechain_metadata_dir=self.hparams.pdbsidechain_metadata_dir,
                multiplicity=self.hparams.val_multiplicity,
                limit_complexes=self.hparams.limit_complexes,
                remove_hs=self.hparams.remove_hs,
                num_workers=self.hparams.num_workers,
                vandermers_max_dist=self.hparams.vandermers_max_dist,
                vandermers_buffer_residue_num=self.hparams.vandermers_buffer_residue_num,
                vandermers_min_contacts=self.hparams.vandermers_min_contacts,
                vandermers_max_surrogate_binding_affinity=self.hparams.vandermers_max_surrogate_binding_affinity,
                vandermers_second_ligand_max_closeness=self.hparams.vandermers_second_ligand_max_closeness,
                vandermers_extract_second_ligand=self.hparams.vandermers_extract_second_ligand,
                merge_clusters=self.hparams.merge_clusters,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                vandermers_use_prob_as_surrogate_binding_affinity=self.hparams.vandermers_use_prob_as_surrogate_binding_affinity,
                a2h_assessment_csv_filepath=self.hparams.pdbsidechain_a2h_assessment_csv_filepath,
                filter_using_a2h_assessment=self.hparams.pdbsidechain_filter_using_a2h_assessment,
                a2h_min_tmscore=self.hparams.pdbsidechain_a2h_min_tmscore,
                a2h_max_rmsd=self.hparams.pdbsidechain_a2h_max_rmsd,
                a2h_min_protein_length=self.hparams.pdbsidechain_a2h_min_protein_length,
                a2h_max_protein_length=self.hparams.pdbsidechain_a2h_max_protein_length,
                postprocess_min_protein_length=self.hparams.pdbsidechain_postprocess_min_protein_length,
                postprocess_max_protein_length=self.hparams.pdbsidechain_postprocess_max_protein_length,
                n_lig_patches=self.hparams.n_lig_patches,
            )
        if "pdbbind" in self.hparams.test_datasets:
            pdbbind_test = PDBBindDataset(
                root=self.hparams.pdbbind_dir,
                limit_complexes=self.hparams.limit_complexes,
                cache_path=self.hparams.cache_path,
                split_path=self.hparams.split_test,
                remove_hs=self.hparams.remove_hs,
                max_lig_size=None,
                esm_embeddings_path=self.hparams.pdbbind_esm_embeddings_path,
                apo_protein_structure_dir=self.hparams.pdbbind_apo_protein_structure_dir,
                require_ligand=True,
                num_workers=self.hparams.num_workers,
                protein_file=self.hparams.protein_file,
                ligand_file=self.hparams.ligand_file,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                min_protein_length=self.hparams.min_protein_length,
                max_protein_length=self.hparams.max_protein_length,
                is_test_dataset=True,
                binding_affinity_values_dict=self.pdbbind_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
        if "moad" in self.hparams.test_datasets:
            moad_test = BindingMOADDataset(
                root=self.hparams.moad_dir,
                dockgen_dir=self.hparams.moad_dockgen_dir,
                clusters_filepath=self.hparams.moad_clusters_filepath,
                limit_complexes=self.hparams.limit_complexes,
                cache_path=self.hparams.cache_path,
                split="test",
                remove_hs=self.hparams.remove_hs,
                dockgen_esm_embeddings_path=self.hparams.moad_dockgen_esm_embeddings_path,
                dockgen_esm_embeddings_sequences_path=self.hparams.moad_dockgen_esm_embeddings_sequences_path,
                dockgen_apo_protein_structure_dir=self.hparams.moad_dockgen_apo_protein_structure_dir,
                require_ligand=True,
                num_workers=self.hparams.num_workers,
                include_miscellaneous_atoms=self.hparams.include_miscellaneous_atoms,
                pdbbind_split_train=self.hparams.pdbbind_split_train,
                pdbbind_split_val=self.hparams.pdbbind_split_val,
                split_time=self.hparams.split_time,
                unroll_clusters=True,
                remove_pdbbind=self.hparams.remove_pdbbind,
                remove_promiscuous_targets=self.hparams.remove_promiscuous_targets,
                no_randomness=True,
                dockgen_a2h_assessment_csv_filepath=self.hparams.moad_dockgen_a2h_assessment_csv_filepath,
                is_test_dataset=True,
                binding_affinity_values_dict=self.moad_binding_affinity_values_dict,
                n_lig_patches=self.hparams.n_lig_patches,
            )
        if not ("pdbbind" in self.hparams.test_datasets or "moad" in self.hparams.test_datasets):
            raise ValueError(f"Invalid test dataset(s) specified: {self.hparams.test_datasets}")

        trainset, valset, valset_secondary = None, None, None
        if load_train_val_datasets and self.hparams.overfitting_example_name is not None:
            log.info(
                "Overfitting on a single batch. Establishing a fixed (uncombined) training set."
            )
            if "moad" in self.hparams.train_datasets:
                trainset = moad_train
                valset = moad_train
                log.info("Overfitting using Binding MOAD training set.")
            elif "pdbbind" in self.hparams.train_datasets:
                trainset = pdbbind_train
                valset = pdbbind_train
                log.info("Overfitting using PDBBind training set.")
            elif "pdbsidechain" in self.hparams.train_datasets:
                trainset = pdbsidechain_train
                valset = pdbsidechain_train
                log.info("Overfitting using van der Mers training set.")

        elif load_train_val_datasets and len(self.hparams.train_datasets) == 3:
            trainset1 = pdbbind_train
            trainset2 = moad_train
            trainset3 = pdbsidechain_train
            trainset = CombinedDataset(CombinedDataset(trainset2, trainset1), trainset3)
            valset = moad_val
            if self.hparams.double_val:
                valset_secondary = pdbbind_val

        elif load_train_val_datasets and len(self.hparams.train_datasets) == 2:
            if "pdbbind" in self.hparams.train_datasets and "moad" in self.hparams.train_datasets:
                trainset = CombinedDataset(
                    moad_train,
                    pdbbind_train,
                )
                valset = moad_val
                if self.hparams.double_val:
                    valset_secondary = pdbbind_val

            elif (
                load_train_val_datasets
                and "pdbbind" in self.hparams.train_datasets
                and "pdbsidechain" in self.hparams.train_datasets
            ):
                trainset = CombinedDataset(
                    pdbbind_train,
                    pdbsidechain_train,
                )
                valset = pdbbind_val
                if self.hparams.double_val:
                    valset_secondary = pdbsidechain_val

            elif (
                load_train_val_datasets
                and "moad" in self.hparams.train_datasets
                and "pdbsidechain" in self.hparams.train_datasets
            ):
                trainset = CombinedDataset(
                    moad_train,
                    pdbsidechain_train,
                )
                valset = moad_val
                if self.hparams.double_val:
                    valset_secondary = pdbsidechain_val

        elif load_train_val_datasets and len(self.hparams.train_datasets) == 1:
            if "pdbbind" in self.hparams.train_datasets:
                trainset = pdbbind_train
                valset = pdbbind_val
            elif "moad" in self.hparams.train_datasets:
                trainset = moad_train
                valset = moad_val
            elif "pdbsidechain" in self.hparams.train_datasets:
                trainset = pdbsidechain_train
                valset = pdbsidechain_val

        elif load_train_val_datasets:
            raise ValueError("No datasets specified for training.")

        testset, testset_secondary = None, None
        if "pdbbind" in self.hparams.test_datasets:
            testset = pdbbind_test
        if "moad" in self.hparams.test_datasets:
            testset_secondary = moad_test

        (
            self.data_train,
            self.data_val,
            self.data_val_secondary,
            self.data_test,
            self.data_test_secondary,
        ) = (
            trainset,
            valset,
            valset_secondary,
            testset,
            testset_secondary,
        )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        pass

    @staticmethod
    def dynamic_batching_by_max_edge_count(
        x: Dict[str, Any], max_n_edges: int, max_batch_size: int
    ) -> Any:
        """Dynamically batch by maximum edge count.

        :param x: The input graph data. If `None`, skip the current batch.
        :param max_n_edges: The maximum number of edges.
        :param max_batch_size: The maximum batch size.
        :return: The batched data.
        """
        if "num_u" in x["metadata"].keys():
            num_edges_upperbound = (
                x["metadata"]["num_a"] * 128 + x["metadata"]["num_i"] * 8 + 160**2
            )
        else:
            num_edges_upperbound = x["metadata"]["num_a"] * 128 + 160**2
        batch_size = max(1, min(max_n_edges // num_edges_upperbound, max_batch_size))
        return collate_samples([x] * batch_size)

    def get_dataloader(
        self,
        phase: DATA_PHASE,
        dataset: Dataset,
        **kwargs: Dict[str, Any],
    ) -> DataLoader[Any]:
        """Create a dataloader from a dataset.

        :param phase: The phase of the dataset. Either `"train"`, `"val"`, or `"test"`.
        :param dataset: The dataset.
        :param kwargs: Additional keyword arguments to pass to the dataloader.
        :return: The dataloader.
        """
        if phase == "train":
            batch_size = self.hparams.batch_size
            epoch_frac = self.hparams.epoch_frac
        else:
            batch_size = 1
            epoch_frac = 1
        sampled_indices = np.random.choice(
            len(dataset),
            int(len(dataset) * epoch_frac),
            replace=False,
        )
        if phase == "val":
            sampled_indices = np.repeat(sampled_indices, self.trainer.world_size)
        subdataset = torch.utils.data.Subset(dataset, sampled_indices)
        return DataLoader(
            subdataset,
            batch_size=None,
            collate_fn=lambda x: self.dynamic_batching_by_max_edge_count(
                x, self.hparams.edge_crop_size, batch_size
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.get_dataloader(
            "train",
            dataset=self.data_train,
            shuffle=True,
        )

    def val_dataloader(self) -> List[DataLoader[Any]]:
        """Create and return the validation dataloaders.

        :return: The validation dataloaders.
        """
        val_dataloaders = []
        if self.data_val is not None:
            val_dataloaders.append(
                self.get_dataloader(
                    "val",
                    dataset=self.data_val,
                    shuffle=False,
                )
            )
        if self.data_val_secondary is not None:
            val_dataloaders.append(
                self.get_dataloader(
                    "val",
                    dataset=self.data_val_secondary,
                    shuffle=False,
                )
            )
        assert len(val_dataloaders) > 0, "No validation datasets to be loaded."
        return val_dataloaders

    def test_dataloader(self) -> List[DataLoader[Any]]:
        """Create and return the test dataloaders.

        :return: The test dataloaders.
        """
        test_dataloaders = []
        if self.data_test is not None:
            test_dataloaders.append(
                self.get_dataloader(
                    "test",
                    dataset=self.data_test,
                    shuffle=False,
                )
            )
        if self.data_test_secondary is not None:
            test_dataloaders.append(
                self.get_dataloader(
                    "test",
                    dataset=self.data_test_secondary,
                    shuffle=False,
                )
            )
        assert len(test_dataloaders) > 0, "No test datasets to be loaded."
        return test_dataloaders

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CombinedDataModule()
