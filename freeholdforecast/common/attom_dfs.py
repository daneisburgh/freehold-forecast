import os
import pandas as pd


def get_df_attom(county, landing_directory):
    df = pd.read_csv(os.path.join(landing_directory, f"attomdata-{county}.csv"), low_memory=False)
    df["Parid"] = df["ATTOM ID"]
    df["last_sale_date"] = pd.to_datetime(df.RecordingDate)
    df["last_sale_amount"] = pd.to_numeric(df.TransferAmount)
    df.sort_values(by="last_sale_date", ascending=True, inplace=True)
    return df
