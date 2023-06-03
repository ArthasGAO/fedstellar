#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

PACKAGE_NAME = "fedstellar"
VERSION = "0.1"
AUTHOR = "Enrique Tomás Martínez Beltrán"
AUTHOR_EMAIL = "enriquetomas@um.es"
URL = "https://github.com/enriquetomasmb/fedstellar"
DESCRIPTION = "Framework for dynamic scenario management using DFL approach"
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
LONG_DESC_TYPE = "text/markdown"
LICENSE = "MIT"

INSTALL_REQUIRES = [
    "torch==1.11.0",
    "tensorboard",
    "pytorch-lightning",
    "torchvision",
    "pycryptodome",
]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    license=LICENSE,
    install_requires=INSTALL_REQUIRES,
    package_dir={"fedstellar": "fedstellar"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=[
            "test",
        ],
    ),
    include_package_data=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="test",
)
