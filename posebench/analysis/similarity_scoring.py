# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from runs-n-poses: (https://github.com/plinder-org/runs-n-poses)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rootutils
from plinder.core import get_config
from plinder.core.scores import query_index
from plinder.data.utils.annotations.aggregate_annotations import Entry
from plinder.data.utils.annotations.get_similarity_scores import Scorer
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, rdShapeAlign, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PLINDER_DIR = Path(get_config().data.plinder_dir)
MAIN_DIRECTORY = Path("scoring")
MAIN_DIRECTORY.mkdir(exist_ok=True, parents=True)
DIRECTORIES = {
    "db": MAIN_DIRECTORY / "db",
    "search": MAIN_DIRECTORY / "search",
    "scores": MAIN_DIRECTORY / "scores",
}


def align_molecules_crippen(mol_ref, mol_probe, iterations=100):
    """Align two molecules using the Crippen O3A method.

    :param mol_ref: The reference molecule.
    :param mol_probe: The probe molecule.
    :param iterations: The number of iterations for the alignment.
    """
    crippenO3A = Chem.rdMolAlign.GetCrippenO3A(mol_probe, mol_ref, maxIters=iterations)
    crippenO3A.Align()


def align_molecules(
    reference: Chem.Mol,
    mobile: Chem.Mol,
    max_preiters: int = 100,
    max_postiters: int = 100,
) -> Tuple[float, float, np.ndarray]:
    """Align two molecules and return the RMSD, Tanimoto, and aligned
    coordinates.

    :param reference: The reference molecule.
    :param mobile: The mobile molecule.
    :param max_preiters: The maximum number of pre-alignment iterations.
    :param max_postiters: The maximum number of post-alignment
        iterations.
    :return: A tuple containing the RMSD, Tanimoto, and aligned
        coordinates.
    """
    align_molecules_crippen(reference, mobile, iterations=max_preiters)
    return rdShapeAlign.AlignMol(
        reference,
        mobile,
        max_preiters=max_preiters,
        max_postiters=max_postiters,
    )


# Adapted from https://github.com/susanhleung/SuCOS
# Initialize feature factory for pharmacophore scoring
FDEF = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))

# Feature map parameters
FEAT_MAP_PARAMS = {k: FeatMaps.FeatMapParams() for k in FDEF.GetFeatureFamilies()}

# Feature types to keep for pharmacophore scoring
PHARMACOPHORE_FEATURES = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)


def get_feature_map_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    """Calculate the feature map score between two molecules.

    :param mol_1: The first molecule.
    :param mol_2: The second molecule.
    :param score_mode: The scoring mode to use.
    :return: The feature map score.
    """
    feat_lists = []
    for molecule in [mol_1, mol_2]:
        raw_feats = FDEF.GetFeaturesForMol(molecule)
        feat_lists.append([f for f in raw_feats if f.GetFamily() in PHARMACOPHORE_FEATURES])

    feat_maps = [
        FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=FEAT_MAP_PARAMS) for x in feat_lists
    ]
    feat_maps[0].scoreMode = score_mode

    score = feat_maps[0].ScoreFeats(feat_lists[1])
    return score / min(feat_maps[0].GetNumFeatures(), len(feat_lists[1]))


def get_sucos_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    """Calculate the SuCOS similarity score of two molecules.

    :param mol_1: The first molecule.
    :param mol_2: The second molecule.
    :param score_mode: The scoring mode to use.
    :return: The SuCOS similarity score.
    """
    fm_score = get_feature_map_score(mol_1, mol_2, score_mode)
    fm_score = np.clip(fm_score, 0, 1)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(mol_1, mol_2, allowReordering=False)
    protrude_dist = np.clip(protrude_dist, 0, 1)

    return 0.5 * fm_score + 0.5 * (1 - protrude_dist)


def get_random_conformer(
    molecule: Chem.Mol,
    random_seed: int = 42,
    max_iterations: Optional[int] = None,
) -> Optional[Chem.Conformer]:
    """Generate a random conformer using ETKDGv3.

    :param mol: Input molecule
    :param random_seed: Random seed for reproducibility
    :param max_iterations: Maximum number of iterations for conformer
        generation
    :return: Generated conformer or None if generation fails
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if max_iterations is not None:
        params.maxIterations = max_iterations

    mol_copy = Chem.Mol(molecule)
    try:
        conformer_id = AllChem.EmbedMolecule(mol_copy, params)
        if conformer_id >= 0:
            return mol_copy
        return None
    except ValueError:
        return None


class SimilarityScorer:
    """Class for calculating similarity scores between systems."""

    def __init__(self):
        self.db_dir = DIRECTORIES["scores"]
        self.training_cutoff = pd.to_datetime("2021-09-30")
        self.output_dir = self.db_dir / "all_scores"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open("new_pdb_ids.txt") as f:
            self.new_pdb_ids = set(f.read().split(","))
        self.fpgen = AllChem.GetRDKitFPGenerator()

    def score_system_plinder(self, pdb_id: str) -> None:
        """Score a system using PLINDER.

        :param pdb_id: The PDB ID of the system to score.
        """
        filename = self.db_dir / f"holo_foldseek/aln/{pdb_id}.parquet"
        if not filename.exists():
            logger.info(f"No Foldseek file for {pdb_id}")
            return None
        df = pd.read_parquet(filename)
        target_ids = set(df["target_pdb_id"])
        entries = {}
        for p in target_ids.intersection(self.new_pdb_ids):
            try:
                entries[p] = Entry.from_json(
                    PLINDER_DIR / "systems" / p[-3:-1] / f"{p}.json"
                ).prune(
                    clear_non_pocket_residues=True,
                    load_for_scoring=True,
                    max_protein_chains=20,
                    max_ligand_chains=20,
                )
            except Exception as e:
                logger.error(f"Failed to load entry for {p} due to: {e}")
                continue
        scorer = Scorer(
            entries=entries,
            source_to_full_db_file={},
            db_dir=self.db_dir,
            scores_dir=Path("scores"),
            minimum_threshold=0.0,
        )
        scorer.get_score_df(PLINDER_DIR, pdb_id=pdb_id, search_db="holo", overwrite=False)

    def score_system_ligand(self, pdb_id: str):
        """Score a system using ligand-based methods.

        :param pdb_id: The PDB ID of the system to score.
        """
        plinder_score_file = self.db_dir / f"search_db=holo/{pdb_id}.parquet"
        scores_df = pd.read_parquet(plinder_score_file)
        system_id_pairs = set(zip(scores_df["query_system"], scores_df["target_system"]))
        all_systems_to_load = set(scores_df["query_system"]).union(set(scores_df["target_system"]))
        plindex = query_index(
            columns=[
                "system_id",
                "system_biounit_id",
                "ligand_instance_chain",
                "ligand_ccd_code",
                "ligand_rdkit_canonical_smiles",
                "system_protein_chains_auth_id",
            ],
            filters=[
                ("system_id", "in", all_systems_to_load),
                ("ligand_is_proper", "==", True),
            ],
            splits=["*"],
        ).drop(columns=["split"])
        fps = defaultdict(dict)
        mols = defaultdict(dict)
        for system_id, ligand_instance_chain, smiles in tqdm(
            zip(
                plindex["system_id"],
                plindex["ligand_instance_chain"],
                plindex["ligand_rdkit_canonical_smiles"],
            ),
            total=len(plindex),
            desc="getting fingerprints",
        ):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mols[system_id][ligand_instance_chain] = mol
                fps[system_id][ligand_instance_chain] = self.fpgen.GetFingerprint(mol)
            except Exception as e:
                logger.info(f"Failed to get fingerprint for {system_id} due to: {e}")
                continue
        system_id_to_protein_chains = defaultdict(set)
        system_id_to_ligand_instance_chain = defaultdict(set)
        for system_id, protein_chains, ligand_instance_chain in zip(
            plindex["system_id"],
            plindex["system_protein_chains_auth_id"],
            plindex["ligand_instance_chain"],
        ):
            system_id_to_protein_chains[system_id].update(set(protein_chains))
            system_id_to_ligand_instance_chain[system_id].add(ligand_instance_chain)
        score_dict = defaultdict(lambda: defaultdict(list))
        errors = []
        for system_1, system_2 in tqdm(system_id_pairs):
            if system_1.split("__")[1] != "1":
                continue
            if (
                system_1 not in system_id_to_protein_chains
                or system_2 not in system_id_to_protein_chains
            ):
                continue
            if (
                system_1 not in system_id_to_ligand_instance_chain
                or system_2 not in system_id_to_ligand_instance_chain
            ):
                continue
            foldseek_results = (
                pd.read_parquet(
                    self.db_dir / f"holo_foldseek/aln/{system_1[:4]}.parquet",
                    filters=[
                        ("target_pdb_id", "==", system_2[:4]),
                        ("query_chain", "in", system_id_to_protein_chains[system_1]),
                        ("target_chain", "in", system_id_to_protein_chains[system_2]),
                    ],
                    columns=["u", "t", "lddt"],
                )
                .sort_values(by="lddt", ascending=False)
                .head(1)
            )
            if foldseek_results.empty:
                continue
            rotation = np.array(list(map(float, foldseek_results.u[0].split(","))))
            translation = np.array(list(map(float, foldseek_results.t[0].split(","))))
            for ligand_instance_chain_1 in system_id_to_ligand_instance_chain[system_1]:
                key = (system_1, ligand_instance_chain_1)
                try:
                    score_dict["tanimoto"][(key, system_2)] = [
                        max(
                            [
                                DataStructs.TanimotoSimilarity(
                                    fps[system_1][ligand_instance_chain_1],
                                    fps[system_2][ligand_instance_chain_2],
                                )
                                for ligand_instance_chain_2 in system_id_to_ligand_instance_chain[
                                    system_2
                                ]
                            ]
                        )
                    ]
                except Exception as e:
                    errors.append((key, system_2, "tanimoto", e))
                for ligand_instance_chain_2 in system_id_to_ligand_instance_chain[system_2]:
                    sdf_file_1 = (
                        PLINDER_DIR
                        / "systems"
                        / system_1
                        / "ligand_files"
                        / f"{ligand_instance_chain_1}.sdf"
                    )
                    sdf_file_2 = (
                        PLINDER_DIR
                        / "systems"
                        / system_2
                        / "ligand_files"
                        / f"{ligand_instance_chain_2}.sdf"
                    )
                    if not sdf_file_1.exists() or not sdf_file_2.exists():
                        continue
                    try:
                        # ALIGN USING FOLDSEEK
                        mol_1 = Chem.MolFromMolFile(str(sdf_file_1))
                        mol_2 = Chem.MolFromMolFile(str(sdf_file_2))
                        conf = mol_2.GetConformer()
                        coords = np.array(
                            [list(conf.GetAtomPosition(i)) for i in range(mol_2.GetNumAtoms())]
                        )
                        rotated_coords = coords @ rotation.reshape(3, 3).T + translation
                        for i in range(mol_2.GetNumAtoms()):
                            conf.SetAtomPosition(i, rotated_coords[i])
                        score_dict["sucos_protein"][(key, system_2)].append(
                            get_sucos_score(mol_1, mol_2)
                        )
                    except Exception as e:
                        errors.append((key, system_2, "sucos_protein", e))
                    # ALIGN USING SHAPE
                    try:
                        mol_1 = Chem.MolFromMolFile(str(sdf_file_1))
                        mol_2 = Chem.MolFromMolFile(str(sdf_file_2))
                    except Exception:
                        try:
                            mol_1 = Chem.MolFromMolFile(
                                str(sdf_file_1), sanitize=False, strictParsing=False
                            )
                            mol_2 = Chem.MolFromMolFile(
                                str(sdf_file_2), sanitize=False, strictParsing=False
                            )
                        except Exception as e:
                            errors.append((key, system_2, "sucos_shape", e))
                            continue
                    try:
                        shape_similarity, color_similarity = align_molecules(mol_1, mol_2)
                    except Exception as e:
                        errors.append((key, system_2, "shape", e))
                        continue
                    try:
                        score_dict["sucos_shape"][(key, system_2)].append(
                            get_sucos_score(mol_1, mol_2)
                        )
                    except Exception:
                        score_dict["sucos_shape"][(key, system_2)].append(0)
                    score_dict["shape"][(key, system_2)].append(shape_similarity)
                    score_dict["color"][(key, system_2)].append(color_similarity)

        dfs = []
        for metric in score_dict:
            df = pd.DataFrame(
                [
                    {
                        "query_system": key[0],
                        "query_ligand_instance_chain": key[1],
                        "target_system": system_2,
                        "metric": metric,
                        "similarity": (
                            np.nanmax(score_dict[metric][(key, system_2)])
                            if len(score_dict[metric][(key, system_2)]) > 0
                            else np.nan
                        ),
                    }
                    for key, system_2 in score_dict[metric]
                ]
            )
            dfs.append(df)
        if len(dfs) > 0:
            df = (
                pd.concat(dfs)
                .pivot_table(
                    index=[
                        "query_system",
                        "query_ligand_instance_chain",
                        "target_system",
                    ],
                    columns="metric",
                    values="similarity",
                    observed=False,
                )
                .reset_index()
            )
            scores_df = scores_df.pivot_table(
                index=["query_system", "target_system"],
                columns="metric",
                values="similarity",
                observed=False,
            ).reset_index()
            scores_df = (
                pd.merge(scores_df, df, on=["query_system", "target_system"], how="outer")
                .reset_index(drop=True)
                .drop_duplicates()
            )
            scores_df.to_parquet(self.output_dir / f"{pdb_id}.parquet", index=False)
        if len(errors):
            with open(self.output_dir / f"{pdb_id}.errors", "w") as f:
                for line in errors:
                    f.write("\t".join([str(x) for x in line]) + "\n")


def main():
    """Execute similarity scoring."""
    import sys

    pdb_id = sys.argv[1]
    scorer = SimilarityScorer()

    filename = DIRECTORIES["scores"] / f"holo_foldseek/aln/{pdb_id}.parquet"
    plinder_score_file = DIRECTORIES["scores"] / f"search_db=holo/{pdb_id}.parquet"
    output_file = DIRECTORIES["scores"] / f"scores/all_scores/{pdb_id}.parquet"
    if filename.exists():
        if not plinder_score_file.exists():
            scorer.score_system_plinder(pdb_id)
        if plinder_score_file.exists() and not output_file.exists():
            scorer.score_system_ligand(pdb_id)


if __name__ == "__main__":
    main()
