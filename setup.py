"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from freeholdforecast import __version__

PACKAGE_REQUIREMENTS = [
    "auto-sklearn",
    "azure-storage-blob",
    "boto3",
    "googlemaps",
    "imbalanced-learn",
    "ipywidgets",
    "mlflow",
    "openpyxl",
    "pandas",
    "psycopg2",
    "pyarrow",
    "pyyaml",
    "scikit-learn",
    "sqlalchemy",
    "xlsxwriter",
]

DEV_REQUIREMENTS = [
    # installation & build
    "setuptools",
    "wheel",
    # versions set in accordance with DBR 10.4 ML Runtime
    "delta-spark",
    "pyspark",
    # development & testing tools
    "coverage[toml]",
    "dbx>=0.8",
    "jupyter",
    "matplotlib",
    "pandarallel",
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
