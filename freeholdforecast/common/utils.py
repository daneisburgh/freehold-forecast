import os
import pandas as pd
import requests
import shutil

from azure.storage.blob import BlobServiceClient

from freeholdforecast.common.task import get_dbutils


def get_container_client(container):
    if os.getenv("APP_ENV") == "local":
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    else:
        from pyspark.context import SparkContext
        from pyspark.sql.session import SparkSession

        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)
        connect_str = get_dbutils(spark).secrets.get(scope="kv-isburghdatabricks", key="secret-stisburghdatabricks")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container)
    return container_client


def copy_file_to_storage(container, file_path_local, container_client=None):
    if os.getenv("APP_ENV") == "local":
        return

    if container_client is None:
        container_client = get_container_client(container)

    file_path_azure = file_path_local.replace(os.path.join("data", container, ""), "")
    blob_client = container_client.get_blob_client(blob=file_path_azure)

    if blob_client.exists():
        blob_client.delete_blob()

    blob_client.create_append_blob()
    chunk_size = 4 * 1024 * 1024  # 4MB

    with open(file_path_local, "rb") as data:
        while True:
            read_data = data.read(chunk_size)

            if read_data:
                blob_client.append_block(read_data)
            else:
                break


def copy_directory_to_storage(container, directory_path):
    if os.getenv("APP_ENV") == "local":
        return

    container_client = get_container_client(container)

    for r, d, f in os.walk(directory_path):
        for file in f:
            copy_file_to_storage(container, os.path.join(r, file), container_client)


def download_file_from_storage(container, file_path):
    container_client = get_container_client(container)

    with open(file=file_path, mode="wb") as download_file:
        download_file.write(container_client.download_blob(file_path).readall())


def date_string(date):
    return date.strftime("%Y-%m-%d")


def download_file_from_source(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as save_file:
            for chunk in response.iter_content(chunk_size=128):
                save_file.write(chunk)

            copy_file_to_storage("etl", save_path)


def file_exists(file_path):
    return os.path.isfile(file_path)


def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def remove_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)


def to_numeric(value):
    return pd.to_numeric(value, errors="coerce")


def diff_month(start_date, end_date):
    import math

    # return int((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)
    return math.floor(((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) / 12)


def round_base(value, base):
    return None if pd.isna(value) else int(round(value / base)) * base
