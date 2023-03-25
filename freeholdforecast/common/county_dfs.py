import numpy as np
import os
import pandas as pd

from functools import partial
from multiprocessing import Pool

from freeholdforecast.common.dat_dfs import get_df_dat
from freeholdforecast.common.hamilton_dfs import get_df_hamilton
from freeholdforecast.common.static import get_df_additional_data, is_valid_sale
from freeholdforecast.common.utils import make_directory, to_numeric


state_counties = {
    "ohio": [
        "ohio-butler",
        "ohio-clermont",
        "ohio-hamilton",
    ],
}


def get_df_state(self, landing_directory):
    df_counties = []

    for county in state_counties[self.state]:
        county_landing_directory = os.path.join(landing_directory, county)
        make_directory(county_landing_directory)

        df_county = get_df_county(self, county, county_landing_directory)
        df_counties.append(df_county)

    return pd.concat(
        df_counties,
        copy=False,
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)


def get_df_county(task, county, landing_directory):
    if county in ["ohio-butler", "ohio-clermont"]:
        df = get_df_dat(county, landing_directory)
    elif county == "ohio-hamilton":
        df = get_df_hamilton(landing_directory)

    df["State"] = county.split("-")[0].title()
    df["County"] = county.split("-")[1].title()
    # df["Parid"] = df[["County", "Parid"]].agg("-".join, axis=1)

    df.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    df["House #"] = to_numeric(df["House #"])
    df["Year Built"] = to_numeric(df["Year Built"])
    df["last_sale_price"] = to_numeric(df["last_sale_price"])
    df["Stories"] = df.Stories.apply(lambda x: np.nan if pd.isna(x) else int(np.ceil(x)))

    min_sale_year = 1900

    df = df.loc[
        (df["House #"].notna() & (df["House #"] > 0) & (df["House #"] < 1000000))
        & (df["last_sale_price"].notna() & (df["last_sale_price"] > 1000))
    ]

    df["last_sale_date"] = pd.to_datetime(df.last_sale_date)
    df["last_sale_year"] = df.last_sale_date.dt.year

    def drop_duplicate_sale_years(df_temp):
        return df_temp.drop_duplicates(subset=["Parid", "last_sale_year"], keep="last", ignore_index=True)

    df = drop_duplicate_sale_years(df)

    def get_single_sale_parcels(df_temp):
        df_size = df_temp.groupby("Parid", as_index=False).size()
        return df_size.loc[df_size["size"] == 1].Parid.unique()

    # df_single_sales = df.loc[df.Parid.isin(get_single_sale_parcels(df))]

    # df.drop(columns=["Last Owner Name 1", "last_sale_year", "different_owner"], inplace=True)

    # df_additional = pd.DataFrame(
    #     Pool(task.cpu_count).map(
    #         partial(get_df_additional_data, columns=df_single_sales.columns),
    #         df_single_sales.to_dict("records"),
    #     )
    # )

    ignore_columns = [
        "Owner Name 1",
        # "Deed Type",
        # "Valid Sale",
        "Sale Price",
        # "Building Value",
        # "Land Value",
        "last_sale_price",
        "last_sale_date",
    ]

    df_additional = pd.DataFrame(
        Pool(task.cpu_count).map(
            partial(get_df_additional_data, all_columns=df.columns, ignore_columns=ignore_columns),
            df.loc[df["Year Built"].notna() & (df["Year Built"] > min_sale_year)].to_dict("records"),
        )
    )

    df_all = pd.concat([df_additional, df], copy=False, ignore_index=True)
    df_all["last_sale_date"] = pd.to_datetime(df_all.last_sale_date)
    df_all["last_sale_year"] = df_all.last_sale_date.dt.year
    df_all = drop_duplicate_sale_years(df_all)
    df_all.sort_values(by="last_sale_date", inplace=True, ignore_index=True)

    for column in ["Building Value", "Land Value", "Tax District", "Year Built"]:
        df_all[column] = to_numeric(df_all[column])
        df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.bfill())
        df_all[column] = df_all.groupby("Parid", as_index=False)[column].transform(lambda x: x.ffill())
        # df_all = df_all.loc[df_all[column].notna()]

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

    def is_business_owner(owner):
        owner_lower = "" if pd.isna(owner) else str(owner).lower()
        business_identifiers = [" inc", " llc", " ltd"]

        for x in business_identifiers:
            if x in owner_lower:
                return 1

        return 0

    df_all["Last Owner Name 1"] = df_all.groupby("Parid")["Owner Name 1"].shift()
    df_all["same_owner"] = df_all.apply(is_same_owner, axis=1)
    df_all["business_owner"] = df_all["Owner Name 1"].apply(is_business_owner)
    df_all["valid_sale"] = df_all.apply(is_valid_sale, axis=1)
    df_all["next_sale_date"] = df_all.groupby("Parid").last_sale_date.shift(-1)
    df_all["next_sale_price"] = df_all.groupby("Parid").last_sale_price.shift(-1)
    df_all["next_same_owner"] = df_all.groupby("Parid").same_owner.shift(-1)
    df_all["next_business_owner"] = df_all.groupby("Parid").business_owner.shift(-1)
    df_all["next_valid_sale"] = df_all.groupby("Parid").valid_sale.shift(-1)
    df_all = df_all.loc[df_all.last_sale_date.dt.year > min_sale_year]

    return df_all.astype(str)
