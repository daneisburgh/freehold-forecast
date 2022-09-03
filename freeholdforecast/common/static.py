def get_parcel_prepared_data(parcel_id, df_raw_encoded):
    import pandas as pd

    from datetime import datetime
    from dateutil.rrule import rrule, MONTHLY

    def get_dates(start_date, end_date):
        # handle 02/29 error
        if start_date.month == 2 and start_date.day == 29:
            start_date = start_date.replace(month=3, day=1)

        end_date = end_date if pd.notnull(end_date) else datetime.now()

        return [
            x
            for x in rrule(
                dtstart=start_date,
                until=end_date,
                freq=MONTHLY,
            )
        ]

    def is_transfered(date_diff, has_next_sale_date):
        return 1 if date_index >= (total_dates - date_diff - 1) and has_next_sale_date else 0

    prepared_data = []
    dates_since_last_sale = 0

    df_parcel_sales = (
        df_raw_encoded.loc[df_raw_encoded.Parid == parcel_id]
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
                prepared_data_object = {
                    "transfer_in_6_months": is_transfered(6, has_next_sale_date),
                    "transfer_in_12_months": is_transfered(12, has_next_sale_date),
                    "transfer_in_24_months": is_transfered(24, has_next_sale_date),
                    "date": date.replace(day=1),
                    "year": date.year,
                    "month": date.month,
                    "dates_since_last_sale": dates_since_last_sale,
                }

                for column in list(df_raw_encoded.columns):
                    prepared_data_object[column] = row[column]

                prepared_data.append(prepared_data_object)
                dates_since_last_sale += 1

                if date_index == total_dates - 2:
                    dates_since_last_sale = 0

    return pd.DataFrame(prepared_data)