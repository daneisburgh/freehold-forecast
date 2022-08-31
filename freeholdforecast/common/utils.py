import numpy as np
import os
import pandas as pd
import requests

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
    container_client = get_container_client(container)

    for r, d, f in os.walk(directory_path):
        for file in f:
            copy_file_to_storage(container, os.path.join(r, file), container_client)


def download_file(url, save_path):
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


def to_numeric(value):
    try:
        return pd.to_numeric(value)
    except:
        return np.nan
