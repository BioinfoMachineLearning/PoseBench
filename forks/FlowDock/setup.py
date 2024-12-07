#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="FlowDock",
    version="0.0.1",
    description="Geometric Flow Matching for Generative Protein-Ligand Docking and Affinity Prediction",
    author="",
    author_email="",
    url="https://github.com/BioinfoMachineLearning/FlowDock",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = flowdock.train:main",
            "eval_command = flowdock.eval:main",
        ]
    },
)
