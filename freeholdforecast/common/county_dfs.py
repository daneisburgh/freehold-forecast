import os

from freeholdforecast.common.dat_dfs import get_df_dat
from freeholdforecast.common.hamilton_dfs import get_df_hamilton
from freeholdforecast.common.utils import make_directory


def get_df_county(run_date, county):
    landing_directory = os.path.join("data", "etl", "landing", run_date, county)
    make_directory(landing_directory)

    if county in ["ohio-butler", "ohio-clermont"]:
        df = get_df_dat(county, landing_directory)
    elif county == "ohio-hamilton":
        df = get_df_hamilton(landing_directory)

    df.loc[:, (df != df.iloc[0]).any()]
    df.dropna(thresh=df.shape[0] * 0.75, how="all", axis=1)

    return df
