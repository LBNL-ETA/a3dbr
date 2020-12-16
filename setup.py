"""
A3DBR
"""

import setuptools
from os import path

root = path.dirname(__file__)

with open(path.join(root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(root, "requirements.txt")) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="A3DBR",
    version="0.1",
    url="https://github.com/avideh/a3dbr",
    download_url="https://github.com/avideh/a3dbr",
    license="MIT",
    maintainer="mwudunn",
    maintainer_email="mwudunn@gmail.com",
    description="Pipeline for 3D Building Reconstruction/Footprint Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: 3D Reconstruction",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)