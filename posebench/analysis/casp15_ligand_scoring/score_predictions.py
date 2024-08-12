#!/usr/bin/env python3

"""This script scores CASP predictions against the targets.

It generates validation output on the standard error.

For valid predictions, it outputs ligand score son the standard output.

Adapted from: https://git.scicore.unibas.ch/schwede/casp15_ligand


Symmetry-corrected RMSD with OpenBabel:
https://gist.github.com/baoilleach/974477
"""

import argparse
import logging
import os
import time
from glob import glob

import numpy
import ost.mol
import pandas as pd
from ost.mol.alg import ligand_scoring

# Local imports
from posebench.analysis.casp15_ligand_scoring import casp_parser, helper


# This is needed if the script is run with `python` instead of `ost`.
def _InitRuleBasedProcessor():
    compound_lib_path = os.path.join(ost.GetSharedDataPath(), "compounds.chemlib")
    if os.path.exists(compound_lib_path):
        compound_lib = ost.conop.CompoundLib.Load(compound_lib_path)
        ost.conop.SetDefaultLib(compound_lib)


_InitRuleBasedProcessor()


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

# NOTE: The excluded targets above were removed to ensure
# (1) no target is an RNA-primary target and
# (2) each remaining target is associated with a high-quality predicted protein structure of perfectly-matching sequence length w.r.t. the experimental protein structure

logging.basicConfig()
logger = logging.getLogger("score_predictions")

frmt = '{target},{submission},{mdl},{pose},{ref_lig},{ref_lig_num},{ref_lig_compound},{ref_formula},{ref_formula_simplified},{mdl_lig},{mdl_lig_name},{mdl_formula},{mdl_formula_simplified},{lddt_pli},{rmsd},{lddt_pli_rmsd},{lddt_pli_n_contacts},"{chain_mapping}",{lddt_bs},{bb_rmsd},{bs_num_res},{bs_num_overlap_res},{bs_radius},{lddt_radius},{lddt_lp_radius},{substructure_match}'


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
        "-o",
        "--output-csv-dir",
        "--output_csv_dir",
        default="../output_csv",
        help="Folder containing the results CSV outputs.",
    )
    parser.add_argument(
        "-g",
        "--glob",
        "--predictions-glob",
        dest="predictions_glob",
        default="*",
        help="A glob for prediction files.",
    )
    parser.add_argument(
        "-l",
        "--lddt-radius",
        dest="lddt_radius",
        type=float,
        default=6.0,
        help="Inclusion radius for lDDT.",
    )
    parser.add_argument(
        "-b",
        "--bs-radius",
        "--binding-site-radius",
        dest="bs_radius",
        type=float,
        default=4.0,
        help="Inclusion radius for the binding site residues.",
    )
    parser.add_argument(
        "--lddt-lp-radius",
        dest="lddt_lp_radius",
        type=float,
        default=10.0,
        help="Inclusion radius lDDT-BS.",
    )
    parser.add_argument(
        "-f", "--fault-tolerant", action="store_true", help="Ignore target validation errors."
    )
    return parser


def main(args):
    print(
        frmt.format(
            target="target_id",
            submission="submission_file",
            mdl="model_num",
            pose="pose_num",
            ref_lig="ref_lig",
            ref_lig_num="ref_lig_num",
            ref_lig_compound="ref_lig_compound",
            ref_formula="ref_lig_formula",
            ref_formula_simplified="ref_lig_formula_simplified",
            relevant="relevant",
            mdl_lig="mdl_lig",
            mdl_lig_name="mdl_lig_name",
            mdl_formula="mdl_lig_formula",
            mdl_formula_simplified="mdl_lig_formula_simplified",
            lddt_pli="lddt_pli",
            rmsd="rmsd",
            lddt_pli_rmsd="lddt_pli_rmsd",
            lddt_pli_n_contacts="lddt_pli_n_contacts",
            chain_mapping="chain_mapping",
            lddt_bs="lddt_lp",
            bb_rmsd="bs_bb_rmsd",
            bs_num_res="bs_ref_res_mapped",
            bs_num_overlap_res="bs_mdl_res_mapped",
            bs_radius="bs_radius",
            lddt_radius="lddt_radius",
            lddt_lp_radius="lddt_lp_radius",
            substructure_match="substructure_match",
        )
    )

    prediction_metrics = []
    for target_id in args.targets:
        logger.info("Target: %s", target_id)
        target = casp_parser.Target(target_id, args.targets_dir)
        try:
            target.validate()
        except casp_parser.CaspValidationError as err:
            if args.fault_tolerant:
                logger.error("Target validation error for %s: %s", target_id, err)
            else:
                raise

        prediction_files = sorted(
            glob(os.path.join(args.predictions, target_id, args.predictions_glob))
        )

        for prediction_file in prediction_files:
            prediction_filename = os.path.basename(prediction_file)
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
                try:
                    parser.validate(target)
                except casp_parser.CaspValidationError as err:
                    logger.error("Validation error in %s: %s", prediction_filename, err)
                else:
                    logger.info("%s parsed and validated successfully" % prediction_filename)

                    for model_id, model in parser.models.items():
                        for pose_id, model_ligands in parser.ligands[model_id].items():
                            sc = ligand_scoring.LigandScorer(
                                model=model,
                                target=target.struct_ost_ent,
                                model_ligands=[l.ost_residue for l in model_ligands],
                                target_ligands=[
                                    l.struct_ost_residue for l in target.target_ligands
                                ],
                                radius=args.bs_radius,
                                lddt_pli_radius=args.lddt_radius,
                                lddt_lp_radius=args.lddt_lp_radius,
                                substructure_match=numpy.any(
                                    [l.substructure_match for l in target.target_ligands]
                                ),
                            )

                            for chain_name, value in sc.lddt_pli_details.items():
                                for resnum_tuple, lddt_pli_result in value.items():
                                    for ref_ligand in target.target_ligands:
                                        if (
                                            ref_ligand.struct_ost_residue.qualified_name
                                            == lddt_pli_result["target_ligand"].qualified_name
                                        ):
                                            break
                                    else:
                                        raise RuntimeError(
                                            "Didn't find ref_ligand %s"
                                            % lddt_pli_result["target_ligand"].qualified_name
                                        )

                                    for model_ligand in model_ligands:
                                        if (
                                            model_ligand.ost_residue.qualified_name
                                            == lddt_pli_result["model_ligand"].qualified_name
                                        ):
                                            break
                                    else:
                                        raise RuntimeError(
                                            "Didn't find model_ligand %s"
                                            % lddt_pli_result["model_ligand"].qualified_name
                                        )

                                    try:
                                        rmsd = sc.rmsd[chain_name][resnum_tuple]
                                    except KeyError:
                                        # Didn't find the ligand in the RMSD - however a different mapping may exist.
                                        # This can easily be the case if we have more modelled than target ligands
                                        # (eg. T1187 or the T1170 and H1171-H1172 series)
                                        # TODO: for a future CASP this must be improved upon
                                        logger.warning(
                                            "Didn't find ligand %s in RMSD mapping. Setting to Infinity, "
                                            "however this might be too pessimistic and a different mapping may exist."
                                            " Refer to lddt_pli_rmsd or use a different assignment.",
                                            str(model_ligand),
                                        )
                                        rmsd = float("inf")

                                    if (
                                        lddt_pli_result["substructure_match"]
                                        and not ref_ligand.substructure_match
                                    ):
                                        raise RuntimeError(
                                            "Found substructure match for %s, not allowed for target %s"
                                            % (model_ligand, ref_ligand)
                                        )

                                    prediction_metrics.append(
                                        {
                                            "target": target_id,
                                            "submission": prediction_filename,
                                            "mdl": model_id,
                                            "pose": pose_id,
                                            "ref_lig": str(ref_ligand),
                                            "ref_lig_num": ref_ligand.ligand_id,
                                            "ref_lig_compound": ref_ligand.compound_id,
                                            "ref_formula": ref_ligand.formula,
                                            "ref_formula_simplified": ref_ligand.simplified_formula,
                                            "relevant": ref_ligand.relevant,
                                            "mdl_lig": str(model_ligand),
                                            "mdl_lig_name": model_ligand.name,
                                            "mdl_formula": model_ligand.formula,
                                            "mdl_formula_simplified": model_ligand.simplified_formula,
                                            "lddt_pli": lddt_pli_result["lddt_pli"],
                                            "rmsd": rmsd,
                                            "lddt_pli_rmsd": lddt_pli_result["rmsd"],
                                            "lddt_pli_n_contacts": lddt_pli_result[
                                                "lddt_pli_n_contacts"
                                            ],
                                            "chain_mapping": str(lddt_pli_result["chain_mapping"]),
                                            "lddt_bs": lddt_pli_result["lddt_lp"],
                                            "bb_rmsd": lddt_pli_result["bb_rmsd"],
                                            "bs_num_res": str(
                                                lddt_pli_result["bs_ref_res_mapped"]
                                            ),
                                            "bs_num_overlap_res": str(
                                                lddt_pli_result["bs_mdl_res_mapped"]
                                            ),
                                            "bs_radius": args.bs_radius,
                                            "lddt_radius": args.lddt_radius,
                                            "lddt_lp_radius": args.lddt_lp_radius,
                                            "substructure_match": lddt_pli_result[
                                                "substructure_match"
                                            ],
                                        }
                                    )
                                    print(
                                        frmt.format(
                                            target=target_id,
                                            submission=prediction_filename,
                                            mdl=model_id,
                                            pose=pose_id,
                                            ref_lig=str(ref_ligand),
                                            ref_lig_num=ref_ligand.ligand_id,
                                            ref_lig_compound=ref_ligand.compound_id,
                                            ref_formula=ref_ligand.formula,
                                            ref_formula_simplified=ref_ligand.simplified_formula,
                                            relevant=ref_ligand.relevant,
                                            mdl_lig=str(model_ligand),
                                            mdl_lig_name=model_ligand.name,
                                            mdl_formula=model_ligand.formula,
                                            mdl_formula_simplified=model_ligand.simplified_formula,
                                            lddt_pli=lddt_pli_result["lddt_pli"],
                                            rmsd=rmsd,
                                            lddt_pli_rmsd=lddt_pli_result[
                                                "rmsd"
                                            ],  # The RMSD corresponding to the best lDDT-PLI - may differ from best RMSD,
                                            lddt_pli_n_contacts=lddt_pli_result[
                                                "lddt_pli_n_contacts"
                                            ],
                                            chain_mapping=str(lddt_pli_result["chain_mapping"]),
                                            lddt_bs=lddt_pli_result["lddt_lp"],
                                            bb_rmsd=lddt_pli_result[
                                                "bb_rmsd"
                                            ],  # backbone RMSD of binding site
                                            bs_num_res=lddt_pli_result[
                                                "bs_ref_res_mapped"
                                            ],  # Number of residues in BS
                                            bs_num_overlap_res=lddt_pli_result[
                                                "bs_mdl_res_mapped"
                                            ],  # Number of overlapping residues in BS
                                            bs_radius=args.bs_radius,
                                            lddt_radius=args.lddt_radius,
                                            lddt_lp_radius=args.lddt_lp_radius,
                                            substructure_match=lddt_pli_result[
                                                "substructure_match"
                                            ],
                                        )
                                    )

    # Write output CSV
    os.makedirs(args.output_csv_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_csv_dir, "scoring_results.csv")
    output_csv_df = pd.DataFrame(prediction_metrics)
    output_csv_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    helper.set_verbosity(args.verbosity)
    logger.debug("Arguments: %s", str(args))
    start_t = time.time()
    with helper.capture_ost_logs():
        main(args)
    end_t = time.time()
    logger.info("Elapsed time: %.1fs", end_t - start_t)
