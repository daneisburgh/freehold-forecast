from freeholdforecast.common.attom_dfs import get_df_attom
from freeholdforecast.common.dat_dfs import get_df_dat
from freeholdforecast.common.hamilton_dfs import get_df_hamilton


def get_df_county(county, landing_directory):
    if county in [
        "california-ventura",
        "colorado-douglas",
        "florida-duval",
        "minnesota-ramsey",
        "northcarolina-mecklenburg",
    ]:
        df = get_df_attom(county, landing_directory)
    elif county in ["ohio-butler", "ohio-clermont"]:
        df = get_df_dat(county, landing_directory)
    elif county == "ohio-hamilton":
        df = get_df_hamilton(landing_directory)

    df.loc[:, (df != df.iloc[0]).any()]  # drop constant columns
    df.dropna(thresh=df.shape[0] * 0.75, axis=1)  # drop columns with too many NA values

    return df.astype(str)
