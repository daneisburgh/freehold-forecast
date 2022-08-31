def update_ready_data(parcel_id, df_raw_copy):
    import pandas as pd

    from datetime import datetime
    from dateutil.rrule import rrule, YEARLY, MONTHLY

    def get_dates(start_date, end_date):
        # handle 02/29 error
        if start_date.month == 2 and start_date.day == 29:
            start_date = start_date.replace(month=3, day=1)

        end_date = (end_date if pd.notnull(end_date) else datetime.now()).replace(month=12, day=31)

        return [
            x
            for x in rrule(
                dtstart=start_date,
                until=end_date,
                freq=YEARLY,
            )
        ]

    ready_data = []
    dates_since_last_sale = 0

    df_parcel_sales = (
        df_raw_copy.loc[df_raw_copy.Parid == parcel_id]
        .sort_values(by="last_sale_date", ascending=True)
        .reset_index(drop=True)
    )

    df_parcel_sales["next_sale_date"] = df_parcel_sales.last_sale_date.shift(-1)

    for row_index, row in df_parcel_sales.iterrows():
        has_next_sale_date = pd.notnull(row.next_sale_date)

        dates = get_dates(row.last_sale_date, row.next_sale_date)
        total_dates = len(dates)

        for date_index, date in enumerate(dates):
            if date_index < total_dates - 1:
                ready_data_object = {
                    "will_sell_next_year": (1 if date_index >= (total_dates - 2) and has_next_sale_date else 0),
                    "will_sell_next_two_years": (1 if date_index >= (total_dates - 3) and has_next_sale_date else 0),
                    "year": date.year,
                    "dates_since_last_sale": dates_since_last_sale,
                }

                for column in list(df_raw_copy.columns):
                    ready_data_object[column] = row[column]

                ready_data.append(ready_data_object)
                dates_since_last_sale += 1

                if date_index == total_dates - 2:
                    dates_since_last_sale = 0

    return pd.DataFrame(ready_data).to_numpy()


def encode_column(column, df_ready_copy, parid_label_encoder):
    import pandas as pd

    from sklearn.preprocessing import LabelEncoder

    try:
        df_ready_copy[column] = df_ready_copy[column].apply(pd.to_numeric)
    except Exception:
        df_ready_copy[column] = df_ready_copy[column].astype(str)
        pass

    if column == "Parid":
        return parid_label_encoder.transform(df_ready_copy[column])
    else:
        return LabelEncoder().fit_transform(df_ready_copy[column])
