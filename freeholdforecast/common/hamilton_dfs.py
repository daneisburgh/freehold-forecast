# Functions to load and format data from Hamilton county

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
            "ValidSale",
            "Conveyance #",
            "DeedType",
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
            "ValidSale",
            "Conveyance #",
            "DeedType",
        ],
    },
}


def get_df_hamilton(landing_directory: str) -> pd.DataFrame:
    """Load and format data for Hamilton county

    Args:
        landing_directory (str): Landing directory path

    Returns:
        pd.DataFrame: Formatted data for Hamilton county
    """

    format_dfs = []

    for format_name in download_format_info.keys():
        format_directory = os.path.join(landing_directory, format_name)

        make_directory(format_directory)

        for file_name in download_format_info[format_name]["files"]:
            download_url = download_root_url + file_name
            download_file_path = os.path.join(format_directory, file_name)
            download_file_from_source(download_url, download_file_path)

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

        format_dfs.append(pd.concat(dfs, ignore_index=True))

    df = pd.concat(format_dfs, ignore_index=True)
    df = df.loc[~df["Book"].str.contains("book", na=False, case=False)]  # filter out header rows
    df["Book"] = df["Book"].apply(lambda x: str(x).rjust(3, "0"))
    df["Plat"] = df["Plat"].apply(lambda x: str(x).rjust(4, "0"))
    df["Parcel"] = df["Parcel"].apply(lambda x: str(x).rjust(4, "0"))
    df["ParcelID"] = df["ParcelID"].apply(lambda x: str(x).rjust(2, "0"))
    df["Parid"] = df[["Book", "Plat", "Parcel", "ParcelID"]].apply(lambda x: "".join(x.astype(str)), axis=1)
    df["Parid"] = df.Parid.replace(" ", "", regex=True)
    df["Building Value"] = df["Building Value"].apply(to_numeric)
    df["Land Value"] = df["Land Value"].apply(to_numeric)
    df["Year of Sale"] = df["Year of Sale"].apply(lambda year: year if year not in [98, 99] else year + 1900)
    df["Month of Sale"] = df["Month of Sale"].apply(lambda month: month if len(str(month)) == 2 else "0" + str(month))
    df["last_sale_price"] = pd.to_numeric(df["Sale Price"])
    df["last_sale_date"] = pd.to_datetime(
        df[["Year of Sale", "Month of Sale", "Day of Sale"]].apply(lambda x: "-".join(x.astype(str)), axis=1),
        format="%Y-%m-%d",
    )

    file_name = "HistoricSalesExport.xlsx"
    download_url = download_root_url + f"revalue/{file_name}"
    download_file_path = os.path.join(landing_directory, file_name)
    download_file_from_source(download_url, download_file_path)
    df_historical_sales = pd.read_excel(download_file_path)
    df_historical_sales["Parid"] = df.parcel_number
    df_historical_sales["Owner Name 1"] = df.owner_name_1
    df_historical_sales["House #"] = df.house_number
    df_historical_sales["Street Name"] = df.street_name
    df_historical_sales["Street Suffix"] = df.street_suffix
    df_historical_sales["DeedType"] = df.instrument_type.str.split().str[0]
    df_historical_sales["ValidSale"] = "Y"
    df_historical_sales["Sale Price"] = df.sale_price
    df_historical_sales["Building Value"] = np.nan
    df_historical_sales["Land Value"] = np.nan
    df_historical_sales["Tax District"] = np.nan
    df_historical_sales["Property Class"] = df.use_code
    df_historical_sales["last_sale_date"] = df.date_of_sale
    df_historical_sales["last_sale_price"] = df.sale_price

    df = pd.concat([df, df_historical_sales])[
        [
            "Parid",
            "Owner Name 1",
            "House #",
            "Street Name",
            "Street Suffix",
            "DeedType",
            "ValidSale",
            "Sale Price",
            "Building Value",
            "Land Value",
            "Tax District",
            "Property Class",
            "Year of Sale",
            "Month of Sale",
            "Day of Sale",
            "last_sale_price",
            "last_sale_date",
        ]
    ]

    file_name = "bldginfo.xlsx"
    download_url = download_root_url + f"revalue/{file_name}"
    download_file_path = os.path.join(landing_directory, file_name)
    download_file_from_source(download_url, download_file_path)
    df_info = pd.read_excel(download_file_path)
    df_info["Parid"] = df.PARCELID
    df_info["SQFT_FLR1"] = pd.to_numeric(df.SQFT_FLR1)
    df_info["SQFT_FLR2"] = pd.to_numeric(df.SQFT_FLR2)
    df_info["SQFT_FLRH"] = pd.to_numeric(df.SQFT_FLRH)
    df_info["Livable Sqft"] = df.SQFT_FLR1 + df.SQFT_FLR2 + df.SQFT_FLRH
    df_info["Stories"] = pd.to_numeric(df.STORYHT)
    df_info["Year Built"] = pd.to_numeric(df.YEARBUILT)
    df_info = df[["Parid", "Livable Sqft", "Stories", "Year Built"]]

    df = df.loc[df.last_sale_price > 0]
    df.sort_values(by="last_sale_date", ascending=True, ignore_index=True, inplace=True)
    df.drop_duplicates(subset=["Parid", "last_sale_date"], keep="first", ignore_index=True, inplace=True)
    df = pd.merge(df, df_info, on="Parid", how="right")
    return df
