"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from freeholdforecast import __version__

PACKAGE_REQUIREMENTS = [
    "auto-sklearn==0.14.7",
    "azure-storage-blob==12.13.1",
    "imbalanced-learn==0.8.1",
    "ipywidgets==8.0.2",
    "mlflow==1.28.0",
    "pandas==1.4.4",
    "pandarallel==1.6.3",
    "pyarrow==9.0.0",
    "pyyaml==6.0",
    "scikit-learn==0.24.1"
]

DEV_REQUIREMENTS = [
    # installation & build
    "setuptools",
    "wheel",
    # versions set in accordance with DBR 10.4 ML Runtime
    "delta-spark==1.1.0",
    "pyspark==3.2.1",
    # development & testing tools
    "coverage[toml]",
    "dbx>=0.7,<0.8",
    "matplotlib",
    "jupyter",
    "pytest",
    "pytest-cov",
]

setup(
    name="freeholdforecast",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    entry_points={
        "console_scripts": [
            "etl = freeholdforecast.tasks.sample_etl_task:entrypoint",
            "ml = freeholdforecast.tasks.sample_ml_task:entrypoint",
        ]
    },
    version=__version__,
    description="",
    author="",
)
