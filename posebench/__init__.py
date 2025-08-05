# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import importlib
import os
from typing import Any

from omegaconf import OmegaConf

METHOD_TITLE_MAPPING = {
    "diffdock": "DiffDock",
    "fabind": "FABind",
    "dynamicbind": "DynamicBind",
    "neuralplexer": "NeuralPLexer",
    "flowdock": "FlowDock",
    "rfaa": "RoseTTAFold-All-Atom",
    "chai-lab": "chai-lab",
    "boltz": "boltz",
    "alphafold3": "alphafold3",
    "vina": "Vina",
    "tulip": "TULIP",
    "p2rank": "P2Rank",
    "consensus_ensemble": "Ensemble (Con)",
}

STANDARDIZED_DIR_METHODS = ["diffdock", "fabind"]


def resolve_omegaconf_variable(variable_path: str) -> Any:
    """Resolve an OmegaConf variable path to its value."""
    # split the string into parts using the dot separator
    parts = variable_path.rsplit(".", 1)

    # get the module name from the first part of the path
    module_name = parts[0]

    # dynamically import the module using the module name
    try:
        module = importlib.import_module(module_name)
        # use the imported module to get the requested attribute value
        attribute = getattr(module, parts[1])
    except Exception:
        module = importlib.import_module(".".join(module_name.split(".")[:-1]))
        inner_module = ".".join(module_name.split(".")[-1:])
        # use the imported module to get the requested attribute value
        attribute = getattr(getattr(module, inner_module), parts[1])

    return attribute


def resolve_method_title(method: str) -> str:
    """Resolve the method title for a given method.

    :param method: The method name.
    :return: The method title for the given method.
    """
    return METHOD_TITLE_MAPPING.get(method, method.removesuffix("_ss"))


def resolve_method_protein_dir(
    method: str,
    dataset: str,
    repeat_index: int,
    pocket_only_baseline: bool,
    single_seq_baseline: bool = False,
) -> str:
    """Resolve the protein directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param repeat_index: The repeat index for the method.
    :param pocket_only_baseline: Whether to return protein files for a
        pocket-only baseline.
    :param single_seq_baseline: Whether to return protein files for a
        single-sequence baseline.
    :return: The protein directory for the given method.
    """
    pocket_suffix = "_bs_cropped" if pocket_only_baseline else ""
    pocket_only_suffix = "_pocket_only" if pocket_only_baseline else ""
    single_seq_suffix = "_ss" if single_seq_baseline or method.endswith("_ss") else ""
    if single_seq_suffix:
        method = method.removesuffix("_ss")
    if method in STANDARDIZED_DIR_METHODS or method in ["vina", "tulip"]:
        return (
            os.path.join(
                "data",
                f"{dataset}_set",
                f"{dataset}_holo_aligned_predicted_structures{pocket_suffix}",
            )
            if os.path.exists(
                os.path.join(
                    "data",
                    f"{dataset}_set",
                    f"{dataset}_holo_aligned_predicted_structures{pocket_suffix}",
                )
            )
            else os.path.join(
                "data",
                f"{dataset}_set",
                (
                    "predicted_structures"
                    if dataset == "casp15"
                    else f"{dataset}_predicted_structures"
                ),
            )
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            f"{dataset}{pocket_only_suffix}",
        )
    elif method in ["neuralplexer", "flowdock", "rfaa", "chai-lab", "boltz", "alphafold3"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}{single_seq_suffix}{pocket_only_suffix}_{dataset}_outputs_{repeat_index}",
        )
    elif method == "consensus_ensemble":
        return os.path.join(
            "data",
            "test_cases",
            dataset,
            f"top_consensus{pocket_only_suffix}_ensemble_predictions_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_ligand_dir(
    method: str,
    dataset: str,
    vina_binding_site_method: str,
    repeat_index: int,
    pocket_only_baseline: bool,
    v1_baseline: bool,
    single_seq_baseline: bool = False,
) -> str:
    """Resolve the ligand directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param vina_binding_site_method: The binding site method used for
        Vina.
    :param repeat_index: The repeat index for the method.
    :param pocket_only_baseline: Whether to return ligand files for a
        pocket-only baseline.
    :param v1_baseline: Whether to return ligand files for a V1
        baseline.
    :param single_seq_baseline: Whether to return ligand files for a
        single-sequence baseline.
    :return: The ligand directory for the given method.
    """
    pocket_only_suffix = "_pocket_only" if pocket_only_baseline else ""
    v1_baseline_suffix = "v1" if v1_baseline else ""
    single_seq_suffix = "_ss" if single_seq_baseline or method.endswith("_ss") else ""
    if single_seq_suffix:
        method = method.removesuffix("_ss")
    if method in STANDARDIZED_DIR_METHODS or method in [
        "neuralplexer",
        "flowdock",
        "rfaa",
        "chai-lab",
        "boltz",
        "alphafold3",
        "tulip",
    ]:
        output_suffix = (
            "s"
            if method
            in [
                "neuralplexer",
                "flowdock",
                "rfaa",
                "chai-lab",
                "boltz",
                "alphafold3",
                "tulip",
            ]
            else ""
        )
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method) + v1_baseline_suffix,
            "inference",
            f"{method}{single_seq_suffix}{pocket_only_suffix}_{dataset}_output{output_suffix}_{repeat_index}",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            f"{dataset}{pocket_only_suffix}",
        )
    elif method == "vina":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"vina{pocket_only_suffix}_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        )
    elif method == "consensus_ensemble":
        return os.path.join(
            "data",
            "test_cases",
            dataset,
            f"top_consensus{pocket_only_suffix}_ensemble_predictions_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_output_dir(
    method: str,
    dataset: str,
    vina_binding_site_method: str,
    ensemble_ranking_method: str,
    repeat_index: int,
    pocket_only_baseline: bool,
    v1_baseline: bool,
    single_seq_baseline: bool = False,
) -> str:
    """Resolve the output directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param vina_binding_site_method: The binding site method used for
        Vina.
    :param ensemble_ranking_method: The ranking method used for the
        ensemble method.
    :param repeat_index: The repeat index for the method.
    :param pocket_only_baseline: Whether to output files for a pocket-
        only baseline.
    :param v1_baseline: Whether to output files for a V1 baseline.
    :param single_seq_baseline: Whether to output files for a single-
        sequence baseline.
    :return: The output directory for the given method.
    """
    pocket_only_suffix = "_pocket_only" if pocket_only_baseline else ""
    v1_baseline_suffix = "v1" if v1_baseline else ""
    single_seq_suffix = "_ss" if single_seq_baseline or method.endswith("_ss") else ""
    if single_seq_suffix:
        method = method.removesuffix("_ss")
    if method in STANDARDIZED_DIR_METHODS or method in [
        "neuralplexer",
        "flowdock",
        "rfaa",
        "chai-lab",
        "boltz",
        "alphafold3",
        "tulip",
    ]:
        output_suffix = (
            "s"
            if method
            in [
                "neuralplexer",
                "flowdock",
                "rfaa",
                "chai-lab",
                "boltz",
                "alphafold3",
                "tulip",
            ]
            else ""
        )
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method) + v1_baseline_suffix,
            "inference",
            f"{method}{single_seq_suffix}{pocket_only_suffix}_{dataset}_output{output_suffix}_{repeat_index}",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            f"{dataset}{pocket_only_suffix}_{repeat_index}",
        )
    elif method in ["vina", "p2rank"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"vina{pocket_only_suffix}_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        )
    elif method == "ensemble":
        return os.path.join(
            "data",
            "test_cases",
            dataset,
            f"top_{ensemble_ranking_method}{pocket_only_suffix}_ensemble_predictions_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_input_csv_path(
    method: str, dataset: str, pocket_only_baseline: bool, single_seq_baseline: bool = False
) -> str:
    """Resolve the input CSV path for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param pocket_only_baseline: Whether to return the input CSV path
        for a pocket-only baseline.
    :param single_seq_baseline: Whether to return the input CSV path for
        a single-sequence baseline.
    :return: The input CSV path for the given method.
    """
    pocket_only_suffix = "_pocket_only" if pocket_only_baseline else ""
    single_seq_suffix = "_ss" if single_seq_baseline or method.endswith("_ss") else ""
    if single_seq_suffix:
        method = method.removesuffix("_ss")
    if method in STANDARDIZED_DIR_METHODS or method in [
        "neuralplexer",
        "flowdock",
        "rfaa",
        "chai-lab",
        "boltz",
        "alphafold3",
        "vina",
        "tulip",
    ]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}{single_seq_suffix}{pocket_only_suffix}_{dataset}_inputs.csv",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}{pocket_only_suffix}_{dataset}_inputs",
        )
    elif method == "ensemble":
        return os.path.join(
            "data",
            "test_cases",
            dataset,
            "ensemble_inputs.csv",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver(
        "resolve_variable", lambda variable_path: resolve_omegaconf_variable(variable_path)
    )
    OmegaConf.register_new_resolver(
        "resolve_method_title",
        lambda method: resolve_method_title(method),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_protein_dir",
        lambda method, dataset, repeat_index, pocket_only_baseline: resolve_method_protein_dir(
            method, dataset, repeat_index, pocket_only_baseline
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_ligand_dir",
        lambda method, dataset, vina_binding_site_method, repeat_index, pocket_only_baseline, v1_baseline: resolve_method_ligand_dir(
            method,
            dataset,
            vina_binding_site_method,
            repeat_index,
            pocket_only_baseline,
            v1_baseline,
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_output_dir",
        lambda method, dataset, vina_binding_site_method, ensemble_ranking_method, repeat_index, pocket_only_baseline, v1_baseline: resolve_method_output_dir(
            method,
            dataset,
            vina_binding_site_method,
            ensemble_ranking_method,
            repeat_index,
            pocket_only_baseline,
            v1_baseline=v1_baseline,
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_input_csv_path",
        lambda method, dataset, pocket_only_baseline: resolve_method_input_csv_path(
            method, dataset, pocket_only_baseline
        ),
    )
