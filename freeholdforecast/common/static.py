def get_df_county_common(county, landing_directory):
    import os

    from freeholdforecast.common.county_dfs import get_df_county
    from freeholdforecast.common.utils import make_directory

    county_landing_directory = os.path.join(landing_directory, county)
    make_directory(county_landing_directory)

    df_county = get_df_county(county, county_landing_directory)[
        [
            "Parid",
            "Year of Sale",
            "Month of Sale",
            "Day of Sale",
            "Owner Name 1",
            "House #",
            "Street Name",
            "Street Suffix",
            "Deed Type",
            "Valid Sale",
            "Sale Price",
            "Building Value",
            "Land Value",
            "Tax District",
            "Property Class",
            "last_sale_amount",
            "last_sale_date",
        ]
    ]
    df_county["county"] = county

    return df_county


def get_parcel_prepared_data(parcel_ids, df_raw_encoded, train_start_date):
    import pandas as pd

    from datetime import datetime
    from dateutil.rrule import rrule, MONTHLY

    def get_months(start_date, end_date):
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

    def has_next_sale(date_diff, has_next_sale_date):
        return 1 if date_index >= (total_months - date_diff - 1) and has_next_sale_date else 0

    prepared_data = []

    for parcel_id in parcel_ids:
        months_since_last_sale = 0

        df_parcel_sales = (
            df_raw_encoded.loc[df_raw_encoded.Parid == parcel_id]
            .sort_values(by="last_sale_date", ascending=True)
            .reset_index(drop=True)
        )

        df_parcel_sales["next_sale_date"] = df_parcel_sales.last_sale_date.shift(-1)
        df_parcel_sales["next_sale_amount"] = df_parcel_sales.last_sale_amount.shift(-1)

        for row_index, row in df_parcel_sales.iterrows():
            has_next_sale_date = pd.notnull(row.next_sale_date)

            months = get_months(row.last_sale_date, row.next_sale_date)
            total_months = len(months)

            for date_index, date in enumerate(months):
                if date_index < total_months - 1:
                    prepared_data_object = {
                        "sale_in_3_months": has_next_sale(3, has_next_sale_date),
                        # "sale_in_6_months": has_next_sale(6, has_next_sale_date),
                        # "sale_in_12_months": has_next_sale(12, has_next_sale_date),
                        "next_sale_amount": pd.to_numeric(row.next_sale_amount, errors="coerce"),
                        # "next_sale_months": (total_months - date_index - 1) if has_next_sale_date else np.nan,
                        "date": date.replace(day=1),
                        "month": date.month,
                        "months_since_last_sale": months_since_last_sale,
                    }

                    for column in list(df_raw_encoded.columns):
                        prepared_data_object[column] = row[column]

                    if date >= train_start_date:
                        prepared_data.append(prepared_data_object)

                    months_since_last_sale += 1

                    if date_index == total_months - 2:
                        months_since_last_sale = 0

    return pd.DataFrame(prepared_data)


def train_model(training_type, task, label_name, n_jobs, model_directory, X_train, y_train, X_test, y_test):
    import mlflow
    import os
    import shutil
    import time

    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.metrics import f1, r2
    from sklearn.metrics import (
        confusion_matrix,
        average_precision_score,
        precision_score,
        roc_auc_score,
        r2_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
    )

    from freeholdforecast.common.utils import copy_directory_to_storage

    task.logger = task._prepare_logger()  # reset logger

    is_classification = training_type == "classification"
    mlflow_run = task.mlflow_client.create_run(task.mlflow_experiment.experiment_id)
    mlflow_run_id = mlflow_run.info.run_id

    with mlflow.start_run(mlflow_run_id, task.mlflow_experiment.experiment_id):
        task.mlflow_client.log_param(mlflow_run_id, "training_type", training_type)
        task.mlflow_client.log_param(mlflow_run_id, "label_name", label_name)
        task.mlflow_client.log_param(mlflow_run_id, "metric", "f1" if is_classification else "r2")
        task.mlflow_client.log_param(mlflow_run_id, "train_years", task.train_years)
        task.mlflow_client.log_param(mlflow_run_id, "fit_jobs", n_jobs)
        task.mlflow_client.log_param(mlflow_run_id, "fit_minutes", task.fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_fit_minutes", task.per_job_fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_memory_limit_gb", task.per_job_fit_memory_limit_gb)

        task.logger.info(f"Fitting model for {label_name}")
        time_left_for_this_task = 60 * task.fit_minutes
        per_run_time_limit = 60 * task.per_job_fit_minutes
        per_job_fit_memory_limit_mb = int(task.per_job_fit_memory_limit_gb * 1024)

        if is_classification:
            model = AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit=per_job_fit_memory_limit_mb,
                n_jobs=n_jobs,
                metric=f1,
                include={"classifier": ["gradient_boosting"]},
                initial_configurations_via_metalearning=0,
            )
        else:
            model = AutoSklearnRegressor(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit=per_job_fit_memory_limit_mb,
                n_jobs=n_jobs,
                metric=r2,
                include={"regressor": ["gradient_boosting"]},
                initial_configurations_via_metalearning=0,
            )

        model.fit(X_train, y_train)

        task.logger = task._prepare_logger()  # reset logger
        task.logger.info(f"Saving model for {label_name}")

        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)

        mlflow.sklearn.log_model(model, model_directory)
        mlflow.sklearn.save_model(model, model_directory)

        time.sleep(30)

        if os.getenv("APP_ENV") != "local":
            copy_directory_to_storage("models", model_directory)

        if len(X_test) > 0:
            y_pred = model.predict(X_test)

            if is_classification:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                average_precision_value = average_precision_score(y_test, y_pred)
                precision_value = precision_score(y_test, y_pred)
                roc_auc_value = roc_auc_score(y_test, y_pred)

                task.mlflow_client.log_metric(mlflow_run_id, "confusion_tp", tp)
                task.mlflow_client.log_metric(mlflow_run_id, "confusion_fp", fp)
                task.mlflow_client.log_metric(mlflow_run_id, "precision", precision_value)
                task.mlflow_client.log_metric(mlflow_run_id, "precision_average", average_precision_value)
                task.mlflow_client.log_metric(mlflow_run_id, "roc_auc", roc_auc_value)
            else:
                r2_value = r2_score(y_test, y_pred)
                mae_value = mean_absolute_error(y_test, y_pred)
                mape_value = mean_absolute_percentage_error(y_test, y_pred)

                task.mlflow_client.log_metric(mlflow_run_id, "r2", r2_value)
                task.mlflow_client.log_metric(mlflow_run_id, "mae", mae_value)
                task.mlflow_client.log_metric(mlflow_run_id, "mape", mape_value)

        mlflow.end_run()


def get_parcel_predict_data(last_sale_date, current_date):
    from dateutil.rrule import rrule, MONTHLY

    return len([x for x in rrule(dtstart=last_sale_date, until=current_date, freq=MONTHLY)])
