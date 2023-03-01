import glob
import numpy as np
import os
import pandas as pd

from freeholdforecast.common.utils import download_file_from_source, make_directory, to_numeric

download_root_url = "https://www.hamiltoncountyauditor.org/download/"
download_format_info = {
    "new-format": {
        "files": [
            "transfer_ytd_new.csv",
            "transfer_files_ytd_2022.csv",
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
        ],
        "columns": [
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
    },
    "old-format": {
        "files": [
            "transfer_files_2006.csv",
            "transfer_files_2005.csv",
            "transfer_files_ytd_2004.csv",
            "transfer_files_ytd_2003.csv",
            "transfer_files_2002.csv",
            "transfer_files_2001.csv",
            "transfer_files_2000.csv",
            "transfer_files_1999.csv",
            "transfer_files_1998.csv",
        ],
        "columns": [
            "Book",
            "Plat",
            "Parcel",
            "ParcelID",  # was Multi-owner
            "Tax District",
            "Owner Name 1",
            "Owner Name 2",
            "Land Value",
            "Building Value",
            "Property Class",
            "House #",
            "Street Name",
            "Street Suffix",
            "Month of Sale",
            "Day of Sale",
            "Year of Sale",
            "# of Parcels Sold",
            "Sale Price",
            "Valid Sale",
            "Conveyance #",
            "Deed Type",
        ],
    },
}


def get_df_hamilton(landing_directory):
    format_dfs = []

    for format_name in download_format_info.keys():
        format_dfs.append(get_df_format(format_name, landing_directory))

    df = pd.concat(format_dfs)
    df = df.loc[~df["Book"].str.contains("book", na=False, case=False)]  # filter out header rows
    df["Book"] = df["Book"].apply(lambda x: str(x).rjust(3, "0"))
    df["Plat"] = df["Plat"].apply(lambda x: str(x).rjust(4, "0"))
    df["Parcel"] = df["Parcel"].apply(lambda x: str(x).rjust(4, "0"))
    df["ParcelID"] = df["ParcelID"].apply(lambda x: str(x).rjust(2, "0"))
    df["Parid"] = df[["Book", "Plat", "Parcel", "ParcelID"]].apply(lambda x: "".join(x.astype(str)), axis=1)
    df["Building Value"] = df["Building Value"].apply(to_numeric)
    df["Land Value"] = df["Land Value"].apply(to_numeric)
    df["Year of Sale"] = df["Year of Sale"].apply(lambda year: year if year not in [98, 99] else year + 1900)
    df["Month of Sale"] = df["Month of Sale"].apply(lambda month: month if len(str(month)) == 2 else "0" + str(month))
    df["last_sale_price"] = pd.to_numeric(df["Sale Price"])
    df["last_sale_date"] = pd.to_datetime(
        df[["Year of Sale", "Month of Sale", "Day of Sale"]].apply(lambda x: "-".join(x.astype(str)), axis=1),
        format="%Y-%m-%d",
    )

    df = pd.concat([df, get_df_historical_sales(landing_directory)])[
        [
            "Parid",
            "Owner Name 1",
            "House #",
            "Street Name",
            "Street Suffix",
            "Deed Type",
            "Valid Sale",
            "Sale Price",
            "Building Value",
            "Land Value",
            "Tax District",
            "Property Class",
            "last_sale_price",
            "last_sale_date",
        ]
    ]

    df.sort_values(by="last_sale_date", ascending=True, inplace=True)
    df = pd.merge(df, get_df_info(landing_directory), on="Parid", how="left")
    return df


def get_df_format(format_name, landing_directory):
    format_directory = os.path.join(landing_directory, format_name)

    make_directory(format_directory)

    # for file_name in download_format_info[format_name]["files"]:
    #     download_url = download_root_url + file_name
    #     download_file_path = os.path.join(format_directory, file_name)
    #     download_file_from_source(download_url, download_file_path)

    dfs = []
    all_files = glob.glob(os.path.join(format_directory, "*.csv"))

    for filename in all_files:
        df = pd.read_csv(
            filename,
            index_col=None,
            header=None,
            names=download_format_info[format_name]["columns"],
        )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_df_historical_sales(landing_directory):
    file_name = "HistoricSalesExport.xlsx"
    download_url = download_root_url + f"revalue/{file_name}"
    download_file_path = os.path.join(landing_directory, file_name)
    # download_file_from_source(download_url, download_file_path)
    df = pd.read_excel(download_file_path)
    df["Parid"] = df.parcel_number
    df["Owner Name 1"] = df.owner_name_1
    df["House #"] = df.house_number
    df["Street Name"] = df.street_name
    df["Street Suffix"] = df.street_suffix
    df["Deed Type"] = df.instrument_type.str.split().str[0]
    df["Valid Sale"] = "Y"
    df["Sale Price"] = df.sale_price
    df["Building Value"] = np.nan
    df["Land Value"] = np.nan
    df["Tax District"] = np.nan
    df["Property Class"] = df.use_code
    df["last_sale_date"] = df.date_of_sale
    df["last_sale_price"] = df.sale_price
    return df


def get_df_info(landing_directory):
    file_name = "bldginfo.xlsx"
    download_url = download_root_url + f"revalue/{file_name}"
    download_file_path = os.path.join(landing_directory, file_name)
    # download_file_from_source(download_url, download_file_path)
    df = pd.read_excel(download_file_path)
    df["Parid"] = df.PARCELID
    df["SQFT_FLR1"] = pd.to_numeric(df.SQFT_FLR1)
    df["SQFT_FLR2"] = pd.to_numeric(df.SQFT_FLR2)
    df["SQFT_FLRH"] = pd.to_numeric(df.SQFT_FLRH)
    df["Livable Sqft"] = df.SQFT_FLR1 + df.SQFT_FLR2 + df.SQFT_FLRH
    df["Stories"] = pd.to_numeric(df.STORYHT)
    df["Year Built"] = pd.to_numeric(df.YEARBUILT)
    return df[["Parid", "Livable Sqft", "Stories", "Year Built"]]
