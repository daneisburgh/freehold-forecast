import glob
import os
import pandas as pd

from freeholdforecast.common.utils import download_file, make_directory, to_numeric

download_root_url = "https://www.hamiltoncountyauditor.org/download/"


def get_df_hamilton(landing_directory):
    df_new_format = get_hamilton_df_new_format(landing_directory)
    df_old_format = get_hamilton_df_old_format(landing_directory)

    common_columns = list(set(df_new_format.columns).intersection(df_old_format.columns))

    df = pd.concat([df_new_format[common_columns], df_old_format[common_columns]])
    # df.dropna(thresh=len(df) * 0.75, axis=1, inplace=True)
    df["Building Value"] = df["Building Value"].apply(to_numeric)
    df["Land Value"] = df["Land Value"].apply(to_numeric)
    df["Parid"] = df[["Book", "Plat", "Parcel"]].apply(lambda x: "".join(x.astype(str)) + "00", axis=1)
    df = df.loc[df["Year of Sale"] != "YearSale"]
    df["Year of Sale"] = df["Year of Sale"].apply(lambda year: year if year not in [98, 99] else year + 1900)
    df["last_sale_date"] = pd.to_datetime(df["Year of Sale"], format="%Y")
    df.sort_values(by="last_sale_date", ascending=True, inplace=True)
    df.dropna(thresh=len(df) * 0.75, axis=1, inplace=True)
    df.drop_duplicates(subset=["Parid", "Year of Sale"], keep="last", inplace=True)
    # df["County"] = "Hamilton"
    return df


def get_hamilton_df_new_format(landing_directory):
    new_format_directory = os.path.join(landing_directory, "new-format")
    new_format_files = [
        "transfer_ytd_new.csv",
        "transfer_files_ytd_2021.csv",
        "transfer_files_ytd_2020.csv",
        "transfer_files_ytd_2019.csv",
        "transfer_files_ytd_2018.csv",
        "transfer_files_ytd_2017.csv",
        "transfer_files_ytd_2016.csv",
        "transfer_files_ytd_2015.csv",
        "transfer_files_ytd_2014.csv",
        "transfer_files_ytd_2013.csv",
        "transfer_files_ytd_2012.csv",
        "transfer_files_ytd_2011.csv",
        "transfer_files_ytd_2010.csv",
        "transfer_files_ytd_2009.csv",
        "transfer_files_ytd_2008.csv",
        "transfer_files_ytd_2007.csv",
    ]

    make_directory(new_format_directory)

    for file_name in new_format_files:
        download_url = download_root_url + file_name
        download_file_path = os.path.join(new_format_directory, file_name)
        download_file(download_url, download_file_path)

    dfs = []
    all_files = glob.glob(os.path.join(new_format_directory, "*.csv"))

    for filename in all_files:
        header = 0 if filename == "transfer_ytd_new.csv" else None
        df = pd.read_csv(
            filename,
            index_col=None,
            header=header,
            names=[
                "Book",
                "Plat",
                "Parcel",
                "ParcelID",
                "Tax District",
                "Owner Name 1",
                "Owner Name 2",
                "Land Value",
                "Building Value",
                "Property Class",
                "House #",
                "Street Name",
                "Street Suffix",
                "Zip Code",
                "Month of Sale",
                "Day of Sale",
                "Year of Sale",
                "# of Parcels Sold",
                "Sale Price",
                "Valid Sale",
                "Conveyance #",
                "Deed Type",
                "Appraisal Area",
                "PriorOwner",
                "PropertyNumber",
            ],
        )

        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True)


def get_hamilton_df_old_format(landing_directory):
    old_format_directory = os.path.join(landing_directory, "old-format")
    old_format_files = [
        "transfer_files_2006.csv",
        "transfer_files_2005.csv",
        "transfer_files_ytd_2004.csv",
        "transfer_files_ytd_2003.csv",
        "transfer_files_2002.csv",
        "transfer_files_2001.csv",
        "transfer_files_2000.csv",
        "transfer_files_1999.csv",
        "transfer_files_1998.csv",
    ]

    make_directory(old_format_directory)

    for file_name in old_format_files:
        download_url = download_root_url + file_name
        download_file_path = os.path.join(old_format_directory, file_name)
        download_file(download_url, download_file_path)

    dfs = []
    all_files = glob.glob(os.path.join(old_format_directory, "*.csv"))

    for filename in all_files:
        df = pd.read_csv(
            filename,
            index_col=None,
            header=None,
            names=[
                "Book",
                "Plat",
                "Parcel",
                "Multi-owner",
                "Tax District",
                "Owner Name 1",
                "Owner Name 2",
                "Land Value",
                "Building Value",
                "Property Class",
                "House #",
                "Street Name",
                "Street Suffix",
                # "Zip Code",
                "Month of Sale",
                "Day of Sale",
                "Year of Sale",
                "# of Parcels Sold",
                "Sale Price",
                "Valid Sale",
                "Conveyance #",
                "Deed Type",
            ],
        )

        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True)
