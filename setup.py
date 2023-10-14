"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from freeholdforecast import __version__

PACKAGE_REQUIREMENTS = [
    "auto-sklearn==0.15.0",
    "azure-storage-blob==12.18.3",
    "boto3==1.28.63",
    "googlemaps==4.10.0",
    "ipywidgets==8.1.1",
    "mlflow==2.7.1",
    "numpy==1.23.1",
    "openpyxl==3.1.2",
    "pandas==2.0.3",
    "psycopg2==2.9.9",
    "pyarrow==13.0.0",
    "pyyaml==6.0.1",
    "sqlalchemy==1.4.47",
    "urllib3==1.26.7",
    "xlsxwriter==3.1.7",
]

DEV_REQUIREMENTS = [
    "black[jupyter]",
    "coverage[toml]",
    "dbx>=0.8",
    "delta-spark",
    "ipympl",
    "jupyter",
    "matplotlib",
    "mypy",
    "pandarallel",
    "pyspark",
    "pytest",
    "pytest-cov",
    "setuptools",
    "wheel",
]

setup(
    name="freeholdforecast",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    version=__version__,
    description="",
    author="",
)
