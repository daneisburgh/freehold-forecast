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
        # "ohio-clermont",
        # "ohio-hamilton",
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
        county_landing_directory = os.path.join(landing_directory, county)
        make_directory(county_landing_directory)

        df_county = get_df_county(county, county_landing_directory)
        df_counties.append(df_county)

    df = pd.concat(
        df_counties,
        copy=False,
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)

    def get_df_attom(data_paths, date_column):
        dfs = []

        for data_path in data_paths:
            dfs.append(pd.read_csv(os.path.join(landing_directory, "attom", data_path), sep="\t", low_memory=False))

        return (
            pd.concat(dfs, copy=False, ignore_index=True)
            .sort_values(by=["[ATTOM ID]", date_column], ascending=True, ignore_index=True)
            .drop_duplicates(subset="[ATTOM ID]", keep="last", ignore_index=True)
        )

    def format_parid(df, parid_column):
        return df[parid_column].replace("(-|\.)", "", regex=True)

    assessor_paths = [f"HOMESHAKE_TAXASSESSOR_000{x}/HOMESHAKE_TAXASSESSOR_000{x}.txt" for x in [1, 4, 6]]
    avm_paths = [f"HOMESHAKE_AVM_000{x}/HOMESHAKE_AVM_000{x}.txt" for x in [2, 5, 7]]

    df_assessor = get_df_attom(assessor_paths, "AssrLastUpdated")
    df_assessor["ParidFormatted"] = format_parid(df_assessor, "ParcelNumberRaw")
    df_avm = get_df_attom(avm_paths, "LastUpdateDate")

    assessor_columns = [
        "[ATTOM ID]",
        "ParidFormatted",
        "YearBuilt",
        "PropertyAddressFull",
        "PropertyAddressZIP",
        "SitusCounty",
        "SitusStateCode",
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
        "[ATTOM ID]",
        "ValuationDate",
        "EstimatedValue",
        "ConfidenceScore",
    ]

    df_assessor_avm = pd.merge(df_assessor[assessor_columns], df_avm[avm_columns], on="[ATTOM ID]", how="left")

    for column in [
        "PropertyAddressFull",
        "CompanyFlag",
        "OwnerTypeDescription1",
        "PartyOwner1NameFull",
        "PropertyUseGroup",
    ]:
        df_assessor_avm[column] = df_assessor_avm[column].str.title()

    df["ParidFormatted"] = format_parid(df, "Parid")
    df = pd.merge(df, df_assessor_avm, on="ParidFormatted", how="inner").drop(columns=["[ATTOM ID]", "ParidFormatted"])

    def get_value_rounded(df, base, columns):
        for column in columns:
            df[column] = to_numeric(df[column])
            df[column + " Rounded"] = df[column].apply(lambda x: round_base(x, base))
        return df

    df = get_value_rounded(
        df,
        500,
        ["AreaBuilding"],
    )
    df = get_value_rounded(
        df,
        1000,
        ["AreaLotSF"],
    )
    df = get_value_rounded(
        df,
        25000,
        ["TaxMarketValueLand", "TaxMarketValueImprovements", "TaxMarketValueTotal"],
    )

    df.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    # df["House #"] = to_numeric(df["House #"])
    # df["Year Built"] = to_numeric(df["Year Built"])
    df["last_sale_price"] = to_numeric(df["last_sale_price"])
    # df["Stories"] = df.Stories.apply(lambda x: np.nan if pd.isna(x) else int(np.ceil(x)))

    min_sale_year = datetime.now() - relativedelta(years=+100)
    task.logger.info(f"Min sale year: {min_sale_year}")

    df = df.loc[
        (df["last_sale_price"].notna() & (df["last_sale_price"] > 1000))
        # (df["House #"].notna() & (df["House #"] > 0) & (df["House #"] < 1000000))
    ]

    df["last_sale_date"] = pd.to_datetime(df.last_sale_date)
    df["last_sale_year"] = df.last_sale_date.dt.year

    def drop_duplicate_sale_years(df_temp):
        return df_temp.drop_duplicates(subset=["Parid", "last_sale_year"], keep="last", ignore_index=True)

    df = drop_duplicate_sale_years(df)
    df["YearBuilt"] = to_numeric(df.YearBuilt)

    df_additional = pd.DataFrame(
        Pool(task.cpu_count).map(
            partial(
                get_df_additional_data,
                all_columns=df.columns,
                ignore_columns=[
                    "CompanyFlag",
                    "OwnerTypeDescription1",
                    "PartyOwner1NameFull",
                    "ValuationDate",
                    "EstimatedValue",
                    "ConfidenceScore",
                    "last_sale_price",
                    "last_sale_date",
                    "last_sale_year",
                ],
            ),
            df.loc[df.YearBuilt.notna() & (df.YearBuilt > min_sale_year)].to_dict("records"),
        )
    )

    df_all = pd.concat([df_additional, df], copy=False, ignore_index=True)
    df_all["last_sale_date"] = pd.to_datetime(df_all.last_sale_date)
    df_all["last_sale_year"] = df_all.last_sale_date.dt.year
    df_all = drop_duplicate_sale_years(df_all)
    df_all.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    for column in [
        "YearBuilt",
        "PropertyAddressFull",
        "PropertyAddressZIP",
        "SitusCounty",
        "SitusStateCode",
        "PropertyLatitude",
        "PropertyLongitude",
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
    ]:
        df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.bfill())
        df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.ffill())

    def is_business_owner(owner):
        owner_lower = "" if pd.isna(owner) else str(owner).lower()
        business_identifiers = [" inc", " llc", " ltd"]

        for x in business_identifiers:
            if x in owner_lower:
                return 1

        return 0

    def is_same_owner(row):
        owner = row["Owner Name 1"]
        last_owner = row["Last Owner Name 1"]

        owner_string_lower = str(owner).lower()
        last_owner_string_lower = str(last_owner).lower()

        def string_values_to_compare(string_lower):
            return [x for x in string_lower.split() if len(x) >= 4]

        owner_string_array = string_values_to_compare(owner_string_lower)
        last_owner_string_array = string_values_to_compare(last_owner_string_lower)

        return (
            1
            if (
                pd.notna(owner)
                and pd.notna(last_owner)
                and (
                    owner_string_lower in last_owner_string_lower
                    or last_owner_string_lower in owner_string_lower
                    or len(list(set(owner_string_array) & set(last_owner_string_array))) >= 1
                )
            )
            else 0
        )

    def is_good_sale(row):
        return 1 if row["business_owner"] == 0 and row["same_owner"] == 0 else 0

    df_all["Last Owner Name 1"] = df_all.groupby("Parid")["Owner Name 1"].shift()
    df_all["business_owner"] = df_all["Owner Name 1"].apply(is_business_owner)
    df_all["same_owner"] = df_all.apply(is_same_owner, axis=1)
    df_all["good_sale"] = df_all.apply(is_good_sale, axis=1)
    df_all["next_business_owner"] = df_all.groupby("Parid").business_owner.shift(-1)
    df_all["next_same_owner"] = df_all.groupby("Parid").same_owner.shift(-1)
    df_all["next_good_sale"] = df_all.groupby("Parid").good_sale.shift(-1)
    df_all["next_sale_date"] = df_all.groupby("Parid").last_sale_date.shift(-1)
    df_all["next_sale_price"] = df_all.groupby("Parid").last_sale_price.shift(-1)
    df_all = df_all.loc[df_all.last_sale_date.dt.year > min_sale_year]

    return df_all.astype(str)[
        [
            "Parid",
            "YearBuilt",
            "PropertyAddressFull",
            "PropertyAddressZIP",
            "SitusCounty",
            "SitusStateCode",
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
            "AreaBuilding Rounded",
            "AreaLotSF Rounded",
            "TaxMarketValueLand Rounded",
            "TaxMarketValueImprovements Rounded",
            "TaxMarketValueTotal Rounded",
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
