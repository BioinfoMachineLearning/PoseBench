# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import subprocess  # nosec
import time
from typing import List

import hydra
import psutil
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def assemble_baseline_command(cfg: DictConfig) -> List[str]:
    """Assemble the baseline command.

    :param cfg: The configuration object.
    :return: The baseline command as a list of strings (i.e., command
        segments).
    """
    if cfg.method in [
        "diffdock",
        "dynamicbind",
        "neuralplexer",
        "flowdock",
        "rfaa",
        "chai-lab",
        "boltz",
        "vina",
    ]:
        # NOTE: When running RoseTTAFold-All-Atom (or Chai-1/Boltz), the `RFAA` (`chai-lab`/`boltz`) Conda environment must be activated instead of the `PoseBench` environment
        vina_suffix = f" method={cfg.vina_binding_site_method}" if cfg.method == "vina" else ""
        cuda_device_suffix = (
            "" if cfg.method == "vina" else f" cuda_device_index={cfg.cuda_device_index}"
        )
        rfaa_suffix = " run_inference_directly=true" if cfg.method == "rfaa" else ""
        method = cfg.method.split("-")[0] if cfg.method == "chai-lab" else cfg.method
        return f"python3 posebench/models/{method}_inference.py dataset={cfg.dataset} repeat_index={cfg.repeat_index} max_num_inputs={cfg.max_num_inputs}{vina_suffix}{cuda_device_suffix}{rfaa_suffix}".split()
    else:
        raise ValueError(f"Invalid method: {cfg.method}")


def get_all_processes(proc: psutil.Process) -> List[psutil.Process]:
    """Recursively get all child processes of a given process.

    :param proc: The process object.
    :return: A list of all child processes.
    """
    procs = [proc]
    children = proc.children(recursive=True)
    procs.extend(children)
    return procs


def query_gpu_memory():
    """Query GPU memory usage using nvidia-smi.

    :return: A dictionary mapping process IDs to their GPU memory usage.
    """
    try:
        gpu_memory_info = (
            subprocess.check_output(  # nosec
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,nounits,noheader",
                ]
            )
            .decode()
            .strip()
            .split("\n")
        )
        gpu_memory_usage = {}
        for line in gpu_memory_info:
            if line:
                pid, mem = map(int, line.split(","))
                gpu_memory_usage[pid] = mem
        return gpu_memory_usage
    except Exception as e:
        logger.error(f"Error querying GPU memory usage: {e}")
        return {}


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="benchmark_baseline_compute_resources.yaml",
)
def main(cfg: DictConfig):
    """Benchmark the baseline compute resources (CPU and GPU memory usage)
    required by an external script."""
    start_time = time.time()

    # Run the baseline command
    baseline_command = assemble_baseline_command(cfg)
    process = subprocess.Popen(baseline_command)  # nosec

    # Monitor CPU and GPU memory usage of the subprocess and its children
    process = psutil.Process(process.pid)
    cpu_memory_usage = 0.0
    gpu_memory_usage = 0.0

    try:
        while process.is_running():
            all_procs = get_all_processes(process)
            cpu_memory_info = sum(proc.memory_info().rss for proc in all_procs)
            cpu_memory_usage = max(
                cpu_memory_usage, cpu_memory_info / 1024 / 1024 / 1024
            )  # Convert to GB

            gpu_memory_info = query_gpu_memory()
            for proc in all_procs:
                pid = proc.pid
                if pid in gpu_memory_info:
                    gpu_memory_usage = max(gpu_memory_usage, gpu_memory_info[pid])

            time.sleep(1)

        process.wait()

    except Exception as e:
        logger.error(f"Error monitoring subprocess: {e}")

    # Calculate the runtime
    runtime = time.time() - start_time

    # Calculate the average runtime and peak CPU (GPU) memory usage
    average_runtime = runtime / cfg.max_num_inputs
    average_cpu_memory_usage = cpu_memory_usage
    average_gpu_memory_usage = gpu_memory_usage / 1024  # Convert to GB

    # Print the results
    logger.info(f"Average Runtime: {average_runtime:.2f} seconds")
    logger.info(f"Peak CPU Memory Usage: {average_cpu_memory_usage:.2f} GB")
    logger.info(f"Peak GPU Memory Usage: {average_gpu_memory_usage:.2f} GB")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
