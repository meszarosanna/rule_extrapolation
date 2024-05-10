#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["mamba"]),
    python_requires="~=3.10.12",
)
