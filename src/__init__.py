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
    "rfaa": "RoseTTAFold-All-Atom",
    "vina": "Vina",
    "tulip": "TULIP",
    "p2rank": "P2Rank",
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
    return METHOD_TITLE_MAPPING.get(method, method)


def resolve_method_protein_dir(
    method: str, dataset: str, repeat_index: int, pocket_only_baseline: bool
) -> str:
    """Resolve the protein directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param repeat_index: The repeat index for the method.
    :param pocket_only_baseline: Whether to return protein files for a pocket-only baseline.
    :return: The protein directory for the given method.
    """
    pocket_postfix = "_bs_cropped" if pocket_only_baseline else ""
    if method in STANDARDIZED_DIR_METHODS or method in ["vina", "tulip"]:
        return (
            os.path.join(
                "data",
                f"{dataset}_set",
                f"{dataset}_holo_aligned_esmfold_structures{pocket_postfix}",
            )
            if os.path.exists(
                os.path.join(
                    "data",
                    f"{dataset}_set",
                    f"{dataset}_holo_aligned_esmfold_structures{pocket_postfix}",
                )
            )
            else os.path.join(
                "data",
                f"{dataset}_set",
                "predicted_structures" if dataset == "casp15" else f"{dataset}_esmfold_structures",
            )
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            dataset,
        )
    elif method in ["neuralplexer", "rfaa"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_outputs_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_ligand_dir(
    method: str, dataset: str, vina_binding_site_method: str, repeat_index: int
) -> str:
    """Resolve the ligand directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param vina_binding_site_method: The binding site method used for Vina.
    :param repeat_index: The repeat index for the method.
    :return: The ligand directory for the given method.
    """
    if method in STANDARDIZED_DIR_METHODS or method in ["neuralplexer", "rfaa", "tulip"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_output{'s' if method in ['neuralplexer', 'rfaa', 'tulip'] else ''}_{repeat_index}",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            dataset,
        )
    elif method == "vina":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_output_dir(
    method: str,
    dataset: str,
    vina_binding_site_method: str,
    ensemble_ranking_method: str,
    repeat_index: int,
) -> str:
    """Resolve the output directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param vina_binding_site_method: The binding site method used for Vina.
    :param ensemble_ranking_method: The ranking method used for the ensemble method.
    :param repeat_index: The repeat index for the method.
    :return: The output directory for the given method.
    """
    if method in STANDARDIZED_DIR_METHODS or method in ["neuralplexer", "rfaa", "tulip"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_output{'s' if method in ['neuralplexer', 'rfaa', 'tulip'] else ''}_{repeat_index}",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            "outputs",
            "results",
            f"{dataset}_{repeat_index}",
        )
    elif method in ["vina", "p2rank"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        )
    elif method == "ensemble":
        return os.path.join(
            "data",
            "test_cases",
            dataset,
            f"top_{ensemble_ranking_method}_ensemble_predictions_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_input_csv_path(method: str, dataset: str) -> str:
    """Resolve the input CSV path for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :return: The input CSV path for the given method.
    """
    if method in STANDARDIZED_DIR_METHODS or method in ["neuralplexer", "rfaa", "vina", "tulip"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_inputs.csv",
        )
    elif method == "dynamicbind":
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_inputs",
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
        lambda method, dataset, vina_binding_site_method, repeat_index: resolve_method_ligand_dir(
            method,
            dataset,
            vina_binding_site_method,
            repeat_index,
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_output_dir",
        lambda method, dataset, vina_binding_site_method, ensemble_ranking_method, repeat_index: resolve_method_output_dir(
            method,
            dataset,
            vina_binding_site_method,
            ensemble_ranking_method,
            repeat_index,
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_input_csv_path",
        lambda method, dataset: resolve_method_input_csv_path(method, dataset),
    )
