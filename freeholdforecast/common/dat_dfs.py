import os
import pandas as pd
import subprocess

from datetime import datetime
from freeholdforecast.common.utils import download_file_from_source, make_directory


def get_df_dat(county, landing_directory):
    download_urls = {
        "ohio-butler": "https://www.butlercountyauditor.org/butler_oh_reports/AA407_files.zip",
        "ohio-clermont": "https://www.clermontauditor.org/wp-content/uploads/PublicAccess/AA407_2021_Final.zip",
    }

    download_url = download_urls[county]
    download_file_name = download_url.split("/")[-1]
    download_file_path = os.path.join(landing_directory, download_file_name)

    data_directory = os.path.join(landing_directory, download_file_name.replace(".zip", ""))
    make_directory(data_directory)

    download_file_from_source(download_url, download_file_path)

    subprocess.run(
        f"unzip {download_file_path} -d {data_directory}".split(),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    data_subdirectory = os.path.join(data_directory, os.listdir(data_directory)[0])

    if os.path.isdir(data_subdirectory):
        data_directory = data_subdirectory

    df_parcels = get_df_parcels(data_directory)
    df_asmt = get_df_asmt(data_directory)
    df_legal = get_df_legal(data_directory)
    df_sales = get_df_sales(data_directory)

    df = (
        df_parcels.merge(df_asmt, on="Parid", suffixes=("_parcels0", "_asmt"))
        .merge(df_legal, on="Parid", suffixes=("_parcels1", "_legal"))
        .merge(df_sales, on="Parid", suffixes=("_parcels2", "_sales"))
    )

    rename_columns = {
        "Own1": "Owner Name 1",
        "Adrno_legal": "House #",
        "Adrstr_legal": "Street Name",
        "Adrsuf_legal": "Street Suffix",
        "Saletype": "Deed Type",
        "Saleval": "Valid Sale",
        "Price": "Sale Price",
        "Aprbldg": "Building Value",
        "Aprland": "Land Value",
        "Taxdist": "Tax District",
        "Class_asmt": "Property Class",
    }

    df.rename(columns=rename_columns, inplace=True)
    df["Year of Sale"] = df.last_sale_date.dt.year
    df["Month of Sale"] = df.last_sale_date.dt.month
    df["Day of Sale"] = df.last_sale_date.dt.day

    common_columns = [
        "Parid",
        "Year of Sale",
        "Month of Sale",
        "Day of Sale",
        "last_sale_amount",
        "last_sale_date",
    ] + list(rename_columns.values())

    df = df[common_columns]
    return df


def get_df_asmt(data_directory):
    df_asmt = pd.read_fwf(
        os.path.join(data_directory, "ASMT.DAT"),
        encoding="ISO-8859-1",
        header=None,
        widths=[
            30,
            6,
            5,
            3,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            6,
            9,
            4,
            4,
            4,
            4,
            6,
            40,
            40,
            9,
            9,
            10,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
        ],
        names=[
            "Parid",
            "Jur",
            "Taxyr",
            "Seq",
            "Aprland",
            "Aprbldg",
            "Asmtland",
            "Asmtbldg",
            "Farmland",
            "Forland",
            "Hmsdland",
            "Hmsdbldg",
            "Ppval",
            "Abate",
            "Abateland",
            "Noticval",
            "Specval",
            "Exmpcode",
            "Noticedate",
            "Reascd",
            "Class",
            "Taxclass",
            "Luc",
            "Flag2",
            "Note1",
            "Note2",
            "Deactivat",
            "Wen",
            "Rolltype",
            "Owncode",
            "Effdate",
            "Flg319",
            "Flgerta",
            "Flgappeal",
            "Flglerta",
            "Lucexmp",
        ],
    )

    return df_asmt


def get_df_legal(data_directory):
    df_legal = pd.read_fwf(
        os.path.join(data_directory, "LEGDAT.DAT"),
        encoding="ISO-8859-1",
        header=None,
        widths=[
            30,
            6,
            5,
            4,
            2,
            11,
            6,
            8,
            2,
            30,
            8,
            40,
            5,
            4,
            10,
            10,
            40,
            40,
            5,
            12,
            6,
            10,
            11,
            11,
            11,
            40,
            40,
            15,
            8,
            8,
            1,
            8,
            10,
            60,
            60,
            60,
            40,
            40,
            9,
            9,
            10,
            9,
            5,
            11,
            7,
            7,
            11,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            80,
            80,
            80,
            80,
            80,
        ],
        names=[
            "Parid",
            "Jur",
            "Taxyr",
            "Seq",
            "Adrpre",
            "Adrno",
            "Addradd",
            "Adrsuf",
            "Adrdir",
            "Adrstr",
            "Adrsuf2",
            "Cityname",
            "Zip1",
            "Zip2",
            "Unitdesc",
            "Unitno",
            "Loc2",
            "Rptdashx",
            "Taxdist",
            "Acres",
            "Compct",
            "Convyno",
            "Xcoord",
            "Ycoord",
            "Zcoord",
            "Lotdim",
            "Subdiv",
            "Subdnum",
            "Sublot",
            "Subblck",
            "Subcode",
            "Condbdg",
            "Condunt",
            "Legal1",
            "Legal2",
            "Legal3",
            "Note1",
            "Note2",
            "Deactivat",
            "Wen",
            "Procname",
            "Procdate",
            "Schdist",
            "Sqft",
            "Actfrt",
            "Actdep",
            "Numlot",
            "User1",
            "User2",
            "User3",
            "User4",
            "User5",
            "User6",
            "User7",
            "User8",
            "User9",
            "User10",
            "User11",
            "User12",
            "User13",
            "User14",
            "User15",
            "User16",
            "User17",
            "User18",
            "User19",
            "User20",
            "User21",
            "User22",
            "User23",
            "User24",
            "User25",
        ],
    )

    return df_legal


def get_df_parcels(data_directory):
    df_parcels = pd.read_fwf(
        os.path.join(data_directory, "PARDAT.DAT"),
        encoding="ISO-8859-1",
        header=None,
        widths=[
            30,
            6,
            5,
            4,
            6,
            2,
            3,
            2,
            10,
            11,
            10,
            8,
            8,
            2,
            50,
            50,
            20,
            20,
            40,
            5,
            4,
            8,
            4,
            4,
            1,
            4,
            30,
            2,
            9,
            9,
            10,
            16,
            15,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            1,
            2,
            2,
            2,
            6,
            1,
            21,
            11,
            3,
            11,
            2,
            2,
            40,
            40,
            40,
            40,
            1,
            9,
            2,
            9,
            9,
            9,
            30,
            4,
            4,
            4,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            9,
            10,
            8,
            10,
            6,
            6,
            6,
            11,
            11,
            20,
            20,
            20,
            20,
        ],
        names=[
            "Parid",
            "Jur",
            "Taxyr",
            "Seq",
            "Mappre",
            "Mapsuf",
            "Rtepre",
            "Rtesuf",
            "Adrpre",
            "Adrno",
            "Adradd",
            "Adrsuf",
            "Adrsuf2",
            "Adrdir",
            "Adrstr",
            "Cityname",
            "Unitdesc",
            "Unitno",
            "Loc2",
            "Zip1",
            "Zip2",
            "Nbhd",
            "Class",
            "Luc",
            "Lucmult",
            "Livunit",
            "Tieback",
            "Tiebackcd",
            "Tielandpct",
            "Tiebldgpct",
            "Landisc",
            "Calcacres",
            "Acres",
            "Street1",
            "Street2",
            "Traffic",
            "Topo1",
            "Topo2",
            "Topo3",
            "Location",
            "Fronting",
            "Util1",
            "Util2",
            "Util3",
            "Ofcard",
            "Partial",
            "Bldgros_d",
            "Bldgros_v",
            "Mscbld_n",
            "Mscbld_v",
            "Notecd1",
            "Notecd2",
            "Note1",
            "Note2",
            "Note3",
            "Note4",
            "Rectype",
            "Pctown",
            "Chgrsn",
            "Salekey",
            "Deactivat",
            "Wen",
            "Alt_ID",
            "Muni",
            "Block",
            "Spot",
            "Juris",
            "Parkprox",
            "Parkquanit",
            "Parktype",
            "Restrict1",
            "Restrict2",
            "Restrict3",
            "Zoning",
            "Prefactmscbld",
            "Adjfact",
            "Fldref",
            "Zfar",
            "Pfar",
            "Afar",
            "Pfarsf",
            "Afarsf",
            "User1",
            "User2",
            "User3",
            "User4",
        ],
    )

    return df_parcels


def get_df_sales(data_directory):
    df_sales = pd.read_fwf(
        os.path.join(data_directory, "SALES.DAT"),
        encoding="ISO-8859-1",
        header=None,
        widths=[
            30,
            6,
            20,
            5,
            4,
            5,
            4,
            9,
            8,
            8,
            205,
            205,
            9,
            11,
            1,
            1,
            14,
            8,
            12,
            15,
            3,
            9,
            11,
            2,
            80,
            80,
            80,
            80,
            80,
            80,
            9,
            9,
            3,
            3,
            9,
            1,
            5,
            11,
            9,
            20,
            20,
            20,
            20,
            20,
            10,
        ],
        names=[
            "Parid",
            "Jur",
            "Transno",
            "Oldyr",
            "Oldseq",
            "Newyr",
            "Newseq",
            "Salekey",
            "Book",
            "Page",
            "Oldown",
            "Own1",
            "Saledt",
            "Price",
            "Source",
            "Saletype",
            "Instruno",
            "Instrtyp",
            "Imageno",
            "Adjprice",
            "Adjreas",
            "Recorddt",
            "Aprtot",
            "Saleval",
            "Note1",
            "Note2",
            "Note3",
            "Note4",
            "Note5",
            "Note6",
            "Transdt",
            "Wen",
            "Mktvalid",
            "Steb",
            "Asmt",
            "Stflag",
            "Nopar",
            "Adjamt",
            "Asr",
            "Linkno",
            "User1",
            "User2",
            "User3",
            "User4",
            "Who",
        ],
    )

    df_sales = df_sales.loc[df_sales.Price > 0]

    def update_invalid_years(date):
        if date.year > datetime.now().year:
            date = date.replace(year=(date.year - 100))

        return date

    df_sales["last_sale_amount"] = pd.to_numeric(df_sales.Price)
    df_sales["last_sale_date"] = pd.to_datetime(df_sales.Saledt, format="%d-%b-%y").apply(update_invalid_years)
    df_sales.sort_values(by="last_sale_date", ascending=True, inplace=True)

    df_sales.Saletype.replace(pd.NA, 0, inplace=True)
    df_sales.Saletype.replace("I", 1, inplace=True)

    return df_sales
