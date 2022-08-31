import os

from freeholdforecast.common.dat_dfs import get_df_dat
from freeholdforecast.common.hamilton_dfs import get_df_hamilton
from freeholdforecast.common.utils import make_directory


def get_df_county(today_date, county):
    landing_directory = os.path.join("data", "etl", "landing", today_date, county)
    make_directory(landing_directory)

    if county in ["ohio-butler", "ohio-clermont"]:
        return get_df_dat(county, landing_directory)
    elif county == "ohio-hamilton":
        return get_df_hamilton(landing_directory)
