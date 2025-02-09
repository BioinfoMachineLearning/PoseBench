import importlib
import os

from beartype.typing import Any
from omegaconf import OmegaConf

METHOD_TITLE_MAPPING = {
    "diffdock": "DiffDock",
    "flowdock": "FlowDock",
    "neuralplexer": "NeuralPLexer",
}

STANDARDIZED_DIR_METHODS = ["diffdock"]


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


def resolve_dataset_path_dirname(dataset: str) -> str:
    """Resolve the dataset path directory name based on the dataset's name.

    :param dataset: Name of the dataset.
    :return: Directory name for the dataset path.
    """
    return "DockGen" if dataset == "dockgen" else dataset


def resolve_method_input_csv_path(method: str, dataset: str) -> str:
    """Resolve the input CSV path for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :return: The input CSV path for the given method.
    """
    if method in STANDARDIZED_DIR_METHODS or method in ["flowdock", "neuralplexer"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_inputs.csv",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def resolve_method_title(method: str) -> str:
    """Resolve the method title for a given method.

    :param method: The method name.
    :return: The method title for the given method.
    """
    return METHOD_TITLE_MAPPING.get(method, method)


def resolve_method_output_dir(
    method: str,
    dataset: str,
    repeat_index: int,
) -> str:
    """Resolve the output directory for a given method.

    :param method: The method name.
    :param dataset: The dataset name.
    :param repeat_index: The repeat index for the method.
    :return: The output directory for the given method.
    """
    if method in STANDARDIZED_DIR_METHODS or method in ["flowdock", "neuralplexer"]:
        return os.path.join(
            "forks",
            METHOD_TITLE_MAPPING.get(method, method),
            "inference",
            f"{method}_{dataset}_output{'s' if method in ['flowdock', 'neuralplexer'] else ''}_{repeat_index}",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver(
        "resolve_variable", lambda variable_path: resolve_omegaconf_variable(variable_path)
    )
    OmegaConf.register_new_resolver(
        "resolve_dataset_path_dirname", lambda dataset: resolve_dataset_path_dirname(dataset)
    )
    OmegaConf.register_new_resolver(
        "resolve_method_input_csv_path",
        lambda method, dataset: resolve_method_input_csv_path(method, dataset),
    )
    OmegaConf.register_new_resolver(
        "resolve_method_title", lambda method: resolve_method_title(method)
    )
    OmegaConf.register_new_resolver(
        "resolve_method_output_dir",
        lambda method, dataset, repeat_index: resolve_method_output_dir(
            method, dataset, repeat_index
        ),
    )
    OmegaConf.register_new_resolver(
        "int_divide", lambda dividend, divisor: int(dividend) // int(divisor)
    )
