#!/usr/bin/env python3

"""This script parses predictions and validates them against the target.

From: https://git.scicore.unibas.ch/schwede/casp15_ligand
"""

import argparse
import logging
import os
import re
from glob import glob

# Local imports
from posebench.analysis.casp15_ligand_scoring import casp_parser, helper

HETERO_TARGET_IDS = [  # 'H1114' removed
    "H1135",
    "H1171v1",
    "H1171v2",
    "H1172v1",
    "H1172v2",
    "H1172v3",
    "H1172v4",
]
RNA_TARGET_IDS = (
    []
)  # 'R1117' removed because bad smiles? Also removed 'R1117v2' since RNA is not supported in this benchmark
# 'R1126', 'R1136' removed
HOMO_TARGET_IDS = [
    # 'T1118', 'T1118v1', `T1127`, 'T1170' removed
    "T1124",
    "T1127v2",
    "T1146",  # 'T1105v1' removed... why?
    "T1152",
    "T1158v1",
    "T1158v2",
    "T1158v3",
    "T1158v4",
    "T1181",
    "T1186",
    "T1187",
    "T1188",
]
TARGET_IDS = HETERO_TARGET_IDS + RNA_TARGET_IDS + HOMO_TARGET_IDS

logging.basicConfig()
logger = logging.getLogger("parse_predictions")

frmt = "{mdl},{pose},{lig},{trg_lig_num},{trg_lig_compound},{relevant},{trg_formula},{mdl_formula},{target},{submission}"


def get_parser():
    """Parse options."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        choices=("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
        default="INFO",
        help="Verbosity/Log level. Defaults to INFO",
    )
    parser.add_argument(
        "-d",
        "--targets-dir",
        "--targets_dir",
        default="../targets",
        help="Folder containing the targets.",
    )
    parser.add_argument(
        "-p", "--predictions", default="../predictions", help="Folder containing the predictions."
    )
    parser.add_argument(
        "-t", "--targets", nargs="*", default=TARGET_IDS, help="IDs of the targets to process."
    )
    parser.add_argument(
        "-m",
        "--mol-plots-dir",
        "--mol_plots_dir",
        default="../mol_plots",
        help="Folder containing generated molecule plots.",
    )
    parser.add_argument(
        "-s",
        "--pdb-save-dir",
        "--pdb_save_dir",
        default="../pdb_save",
        help="Folder containing extracted prediction PDB files.",
    )
    return parser


def main(args):
    # Counts
    num_targets = num_target_ligands = num_prediction_files = num_parsable_prediction_files = (
        num_valid_prediction_files
    ) = num_models = num_poses = num_ligands = 0
    participant_groups = set()
    predictions = set()
    prediction_filename_re = re.compile(r"([TRH]\d+(v\d)?(LG\d+))_\d")

    print(
        frmt.format(
            mdl="model_num",
            pose="pose_num",
            lig="ligand",
            trg_lig_num="trg_ligand_num",
            trg_lig_compound="trg_ligand_num",
            relevant="relevant",
            trg_formula="trg_formula",
            mdl_formula="mdl_formula",
            target="target",
            submission="submission",
        )
    )

    for target_id in args.targets:
        num_targets += 1
        target = casp_parser.Target(target_id, args.targets_dir)
        target.validate()
        num_target_ligands += len(target.target_smiles)
        target.save_images()

        prediction_files = sorted(glob(os.path.join(args.predictions, target_id, "*")))

        for prediction_file in prediction_files:
            num_prediction_files += 1
            prediction_filename = os.path.basename(prediction_file)
            # Extract group
            matches = prediction_filename_re.match(prediction_filename)
            group = matches.group(3)
            participant_groups.add(group)
            prediction = matches.group(1)
            predictions.add(prediction)

            logger.info(prediction_filename)
            with open(prediction_file) as fd:
                prediction = fd.read()
            parser = casp_parser.CaspParser(prediction)
            try:
                parser.parse()
            except casp_parser.CaspParserError as err:
                logger.error("Parse error in %s: %s", prediction_filename, err)
            else:
                # Validate
                num_parsable_prediction_files += 1
                try:
                    parser.validate(target)
                except casp_parser.CaspValidationError as err:
                    if "formula" in str(err):
                        logger.error("Error with formula, saving images for inspection")
                        plots_path = os.path.join(args.mol_plots_dir, target_id)
                        os.makedirs(plots_path, exist_ok=True)
                        plots_prefix = os.path.join(plots_path, prediction_filename)
                        parser.save_images(plots_prefix)
                        # Save PDB...
                        pdb_path = os.path.join(args.pdb_save_dir, target_id)
                        os.makedirs(pdb_path, exist_ok=True)
                        pdb_prefix = os.path.join(pdb_path, prediction_filename)
                        parser.save_pdb(pdb_prefix)
                        logger.error("Validation error in %s: %s", prediction_filename, err)
                else:
                    # Save plots...
                    logger.error("Saving images for inspection")
                    plots_path = os.path.join(args.mol_plots_dir, target_id)
                    os.makedirs(plots_path, exist_ok=True)
                    plots_prefix = os.path.join(plots_path, prediction_filename)
                    parser.save_images(plots_prefix)
                    # Save PDB...
                    pdb_path = os.path.join(args.pdb_save_dir, target_id)
                    os.makedirs(pdb_path, exist_ok=True)
                    pdb_prefix = os.path.join(pdb_path, prediction_filename)
                    parser.save_pdb(pdb_prefix)

                    num_valid_prediction_files += 1
                    logger.info("%s parsed and validated successfully" % prediction_filename)
                    # Write output
                    for model_id, poses in parser.ligands.items():
                        num_models += 1
                        for pose_id, ligands in poses.items():
                            num_poses += 1
                            for ligand in ligands:
                                num_ligands += 1
                                ligand_id, compound_id = ligand.name.split()
                                ligand_id = int(ligand_id)
                                target_ligand = target.target_smiles[ligand_id - 1]
                                print(
                                    frmt.format(
                                        mdl=model_id,
                                        pose=pose_id,
                                        lig=ligand.name,
                                        trg_lig_num=target_ligand.ligand_id,
                                        trg_lig_compound=target_ligand.compound_id,
                                        relevant=target_ligand.relevant,
                                        trg_formula=target_ligand.smiles_obmol.GetFormula(),
                                        mdl_formula=ligand.obmol.GetFormula(),
                                        target=target_id,
                                        submission=prediction_filename,
                                    )
                                )
    # Print
    logging.info("num_targets: %d", num_targets)
    logging.info("num_target_ligands: %d", num_target_ligands)
    logging.info("num_prediction_files: %d", num_prediction_files)
    logging.info("num_predictions: %d", len(predictions))
    logging.info("predictions: %s", str(predictions))
    logging.info("num_parsable_prediction_files: %d", num_parsable_prediction_files)
    logging.info("num_valid_prediction_files: %d", num_valid_prediction_files)
    logging.info("num_models: %d", num_models)
    logging.info("num_poses: %d", num_poses)
    logging.info("num_ligands: %d", num_ligands)
    logging.info("num_groups: %d", len(participant_groups))
    logging.info("groups: %s", str(participant_groups))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    helper.set_verbosity(args.verbosity)
    with helper.capture_ost_logs():
        main(args)
