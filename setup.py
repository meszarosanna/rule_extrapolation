#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["rule_extrapolation", "mamba"]),
    python_requires="~=3.12.3",
)
