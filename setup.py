#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="PoseBench",
    version="0.3.0",
    description="Comprehensive benchmarking of protein-ligand structure generation methods",
    author="Alex Morehead",
    author_email="acmwhb@umsystem.edu",
    url="https://github.com/BioinfoMachineLearning/PoseBench",
    install_requires=["hydra-core"],
    packages=find_packages(),
)
