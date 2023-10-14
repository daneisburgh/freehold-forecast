import gc
import numpy as np
import os
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import partial
from multiprocessing import Pool

from freeholdforecast.common.dat_dfs import get_df_dat
from freeholdforecast.common.hamilton_dfs import get_df_hamilton
from freeholdforecast.common.static import get_df_additional_data
from freeholdforecast.common.utils import make_directory, round_base, to_numeric


state_counties = {
    "ohio": [
        "ohio-butler",
        "ohio-clermont",
        "ohio-hamilton",
    ],
}


def get_df_county(county, landing_directory):
    if county in ["ohio-butler", "ohio-clermont"]:
        df = get_df_dat(county, landing_directory)
    elif county == "ohio-hamilton":
        df = get_df_hamilton(landing_directory)

    df["State"] = county.split("-")[0].title()
    df["County"] = county.split("-")[1].title()

    return df


def get_df_state(task, landing_directory):
    df_counties = []

    for county in state_counties[task.state]:
        task.logger.info(f"Loading data for {county}")
        county_landing_directory = os.path.join(landing_directory, county)
        make_directory(county_landing_directory)

        df_county = get_df_county(county, county_landing_directory)
        df_counties.append(df_county)

    df = pd.concat(
        df_counties,
        copy=False,
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)

    min_sale_price = 0
    df["last_sale_price"] = to_numeric(df.last_sale_price)
    # df = df.loc[(df.last_sale_price.notna()) & (df.last_sale_price > min_sale_price)]

    task.logger.info("Loading source data")

    def get_df_source(data_path_value, date_column):
        dfs = []
        source_data_path = os.path.join(landing_directory, "source")

        for r, d, f in os.walk(source_data_path):
            if len(f) > 0:
                file_name = f[0].split(".")[0]
                data_path = os.path.join(source_data_path, file_name, file_name + ".txt")

                if data_path_value in data_path:
                    dfs.append(pd.read_csv(data_path, sep="\t", low_memory=False))

        return (
            pd.concat(dfs, copy=False, ignore_index=True)
            .sort_values(by=["source_id", date_column], ascending=True, ignore_index=True)
            .drop_duplicates(subset="source_id", keep="last", ignore_index=True)
        )

    def format_parid(df, parid_column):
        return df[parid_column].replace("(-|\.)", "", regex=True)

    df_assessor = get_df_source("ASSESSOR", "AssrLastUpdated")
    df_assessor["ParidFormatted"] = format_parid(df_assessor, "ParcelNumberRaw")
    df_avm = get_df_source("AVM", "LastUpdateDate")

    assessor_columns = [
        "source_id",
        "ParidFormatted",
        "AssessorLastSaleDate",
        "AssessorLastSaleAmount",
        "YearBuilt",
        "PropertyAddressFull",
        "PropertyAddressCity",
        "PropertyAddressZIP",
        "SitusCounty",
        "SitusStateCode",
        "NeighborhoodCode",
        "PropertyLatitude",
        "PropertyLongitude",
        "CompanyFlag",
        "OwnerTypeDescription1",
        "PartyOwner1NameFull",
        "PropertyUseGroup",
        "BathCount",
        "BathPartialCount",
        "BedroomsCount",
        "AreaBuilding",
        "AreaLotSF",
        "StoriesCount",
        "TaxMarketValueLand",
        "TaxMarketValueImprovements",
        "TaxMarketValueTotal",
    ]

    avm_columns = [
        "source_id",
        "ValuationDate",
        "EstimatedValue",
        "ConfidenceScore",
    ]

    df_assessor_avm = pd.merge(df_assessor[assessor_columns], df_avm[avm_columns], on="source_id", how="left")
    df_assessor_avm["AssessorLastSaleAmount"] = df_assessor_avm.AssessorLastSaleAmount.apply(
        lambda x: x if x > 0 else np.nan
    )
    df_assessor_avm["AssessorLastSaleDate"] = pd.to_datetime(df_assessor_avm.AssessorLastSaleDate, errors="coerce")
    df_assessor_avm["YearBuilt"] = pd.to_numeric(df_assessor_avm.YearBuilt, errors="coerce")
    df_assessor_avm = df_assessor_avm.loc[
        (df_assessor_avm.AssessorLastSaleDate.notna())
        & (df_assessor_avm.YearBuilt.notna())
        & (df_assessor_avm.AssessorLastSaleDate.dt.year >= df_assessor_avm.YearBuilt)
    ]
    # df_assessor_avm["last_sale_date"] = df_assessor_avm.AssessorLastSaleDate
    # df_assessor_avm["last_sale_price"] = df_assessor_avm.AssessorLastSaleAmount

    # df = df_assessor_avm.copy()
    # df["last_sale_date"] = df.AssessorLastSaleDate
    # df["last_sale_price"] = df.AssessorLastSaleAmount
    # df["Parid"] = df["ParidFormatted"]

    df["ParidFormatted"] = format_parid(df, "Parid")
    df = pd.merge(df, df_assessor_avm, on="ParidFormatted", how="left")
    df["Parid"] = df.Parid.fillna(df.ParidFormatted)
    df = df.drop(columns=["source_id", "ParidFormatted"])

    del df_assessor
    del df_avm
    del df_assessor_avm
    gc.collect()

    task.logger.info("Formatting initial raw data")

    for column in [
        "PropertyAddressFull",
        "PropertyAddressCity",
        "CompanyFlag",
        # "Owner Name 1",
        "OwnerTypeDescription1",
        "PartyOwner1NameFull",
        "PropertyUseGroup",
    ]:
        df[column] = df[column].str.title()

    def get_value_rounded(df, base, columns):
        for column in columns:
            df[column] = to_numeric(df[column])
            df[column + "Rounded"] = df[column].apply(lambda x: round_base(x, base))
        return df

    df = get_value_rounded(
        df,
        10,
        ["YearBuilt"],
    )
    df = get_value_rounded(
        df,
        500,
        ["AreaBuilding"],
    )
    df = get_value_rounded(
        df,
        5000,
        ["AreaLotSF"],
    )
    df = get_value_rounded(
        df,
        25000,
        [
            "last_sale_price",
            "TaxMarketValueLand",
            "TaxMarketValueImprovements",
            "TaxMarketValueTotal",
            "EstimatedValue",
        ],
    )

    df.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    def drop_duplicate_sale_years(df_temp):
        return df_temp.drop_duplicates(subset=["Parid", "last_sale_year"], keep="last", ignore_index=True)

    df["last_sale_date"] = pd.to_datetime(df.last_sale_date)
    df["last_sale_year"] = df.last_sale_date.dt.year
    df = drop_duplicate_sale_years(df)
    df["YearBuilt"] = to_numeric(df.YearBuilt)

    task.logger.info("Adding additional historical sales data")
    df_copy = df.copy()
    df_all = df.copy()

    # for column in ["YearBuilt"]:
    # for column in ["AssessorLastSaleDate"]:
    for column in ["AssessorLastSaleDate", "YearBuilt"]:
        df_additional = pd.DataFrame(
            Pool(task.cpu_count).map(
                partial(
                    get_df_additional_data,
                    all_columns=df.columns,
                    ignore_columns=[
                        "AssessorLastSaleDate",
                        "AssessorLastSaleAmount",
                        "CompanyFlag",
                        "Owner Name 1",
                        "OwnerTypeDescription1",
                        "PartyOwner1NameFull",
                        "ValuationDate",
                        "ValidSale",
                        "DeedType",
                        "last_sale_price",
                        "last_sale_priceRounded",
                        "last_sale_date",
                        "last_sale_year",
                    ],
                    previous_column=column,
                    min_sale_price=0,
                ),
                df_copy.loc[df_copy[column].notna()].to_dict("records"),
            )
        )

        df_all = pd.concat([df_additional, df_all], ignore_index=True)

    del df_copy
    gc.collect()

    df_all["last_sale_date"] = pd.to_datetime(df_all.last_sale_date)
    df_all["last_sale_price"] = to_numeric(df_all.last_sale_price)
    df_all["last_sale_year"] = df_all.last_sale_date.dt.year
    df_all = drop_duplicate_sale_years(df_all)
    df_all.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    # task.logger.info("Imputing numeric data")

    # for column in [
    #     "YearBuilt",
    #     "PropertyAddressFull",
    #     "PropertyAddressZIP",
    #     "SitusCounty",
    #     "SitusStateCode",
    #     "PropertyLatitude",
    #     "PropertyLongitude",
    #     "PropertyUseGroup",
    #     "BathCount",
    #     "BathPartialCount",
    #     "BedroomsCount",
    #     "AreaBuilding",
    #     "AreaLotSF",
    #     "StoriesCount",
    #     "TaxMarketValueLand",
    #     "TaxMarketValueImprovements",
    #     "TaxMarketValueTotal",
    #     "EstimatedValue",
    #     "ConfidenceScore",
    # ]:
    #     df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.bfill())
    #     df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.ffill())

    def is_business_owner(owner):
        owner_lower = "" if pd.isna(owner) else str(owner).lower()
        business_identifiers = [" inc", " llc", " ltd"]

        for x in business_identifiers:
            if x in owner_lower:
                return 1

        return 0

    def is_same_owner(row):
        owner = row["PartyOwner1NameFull"]
        last_owner = row["LastPartyOwner1NameFull"]
        last_last_owner = row["LastLastPartyOwner1NameFull"]

        owner_string_lower = str(owner).lower()
        last_owner_string_lower = str(last_owner).lower()
        last_last_owner_string_lower = str(last_last_owner).lower()

        def string_values_to_compare(string_lower):
            return [x for x in string_lower.split() if len(x) >= 4]

        owner_string_array = string_values_to_compare(owner_string_lower)
        last_owner_string_array = string_values_to_compare(last_owner_string_lower)
        last_last_owner_string_array = string_values_to_compare(last_last_owner_string_lower)

        return (
            1
            if (
                pd.notna(owner)
                and pd.notna(last_owner)
                and (
                    owner_string_lower in last_owner_string_lower
                    or last_owner_string_lower in owner_string_lower
                    or last_last_owner_string_lower in owner_string_lower
                    or len(list(set(owner_string_array) & set(last_owner_string_array))) >= 1
                    or len(list(set(owner_string_array) & set(last_last_owner_string_array))) >= 1
                )
            )
            else 0
        )

    def is_good_sale(row):
        return (
            1 if ((row["same_owner"] == 0) and (pd.isna(row["last_sale_price"]) or (row["last_sale_price"] > 0))) else 0
        )

    task.logger.info("Filtering final raw data and adding columns")

    df_all = df_all.loc[
        ((df_all.last_sale_price.isna()) | (df_all.last_sale_price > 0))
        & (df_all.last_sale_date.notna())
        & (df_all.last_sale_date.dt.year >= df_all.YearBuilt)
        & (df_all.PropertyAddressFull.notna())
        & (df_all.TaxMarketValueTotal > 0)
        & (df_all.TaxMarketValueTotalRounded > 0)
        # & (df_all.EstimatedValue > 0)
    ]

    df_all["LastPartyOwner1NameFull"] = df_all.groupby("Parid")["PartyOwner1NameFull"].shift()
    df_all["LastLastPartyOwner1NameFull"] = df_all.groupby("Parid")["LastPartyOwner1NameFull"].shift()
    df_all["business_owner"] = df_all["PartyOwner1NameFull"].apply(is_business_owner)
    df_all["last_business_owner"] = df_all["LastPartyOwner1NameFull"].apply(is_business_owner)
    df_all["same_owner"] = df_all.apply(is_same_owner, axis=1)
    df_all["good_sale"] = df_all.apply(is_good_sale, axis=1)
    df_all["next_business_owner"] = df_all.groupby("Parid").business_owner.shift(-1)
    df_all["next_same_owner"] = df_all.groupby("Parid").same_owner.shift(-1)
    df_all["next_good_sale"] = df_all.groupby("Parid").good_sale.shift(-1)
    df_all["next_sale_date"] = df_all.groupby("Parid").last_sale_date.shift(-1)
    df_all["next_sale_price"] = df_all.groupby("Parid").last_sale_price.shift(-1)

    for column in ["BathCount", "BedroomsCount", "AreaBuilding", "AreaLotSF", "StoriesCount"]:
        df_all[column] = to_numeric(df_all[column])
        df_all = df_all.loc[(df_all[column].notna()) & (df_all[column] > 0)]

    return df_all.astype(str)[
        [
            "Parid",
            # "Owner Name 1",
            # "Last Owner Name 1",
            # "Last Last Owner Name 1",
            "YearBuilt",
            "PropertyAddressFull",
            "PropertyAddressCity",
            "PropertyAddressZIP",
            "SitusCounty",
            "SitusStateCode",
            "NeighborhoodCode",
            "PropertyLatitude",
            "PropertyLongitude",
            "PartyOwner1NameFull",
            "CompanyFlag",
            "OwnerTypeDescription1",
            "PropertyUseGroup",
            "BathCount",
            "BathPartialCount",
            "BedroomsCount",
            "AreaBuilding",
            "AreaLotSF",
            "StoriesCount",
            "TaxMarketValueLand",
            "TaxMarketValueImprovements",
            "TaxMarketValueTotal",
            "ValuationDate",
            "EstimatedValue",
            "ConfidenceScore",
            "YearBuiltRounded",
            "AreaBuildingRounded",
            "AreaLotSFRounded",
            "TaxMarketValueLandRounded",
            "TaxMarketValueImprovementsRounded",
            "TaxMarketValueTotalRounded",
            "EstimatedValueRounded",
            "last_sale_priceRounded",
            # "ValidSale",
            # "DeedType",
            "business_owner",
            "next_business_owner",
            "same_owner",
            "next_same_owner",
            "good_sale",
            "next_good_sale",
            "last_sale_date",
            "last_sale_price",
            "next_sale_date",
            "next_sale_price",
        ]
    ]
