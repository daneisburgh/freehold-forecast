def get_df_additional_data(row, all_columns, ignore_columns, previous_column, min_sale_price):
    import numpy as np
    import pandas as pd

    from datetime import datetime, timedelta

    from freeholdforecast.common.utils import date_string, to_numeric

    additional_object = {}

    for column in all_columns:
        additional_object[column] = np.nan if column in ignore_columns else row[column]

    if previous_column == "AssessorLastSaleDate" and pd.notna(row["AssessorLastSaleDate"]):
        additional_object["last_sale_date"] = row["AssessorLastSaleDate"]
        additional_object["last_sale_price"] = (
            row["AssessorLastSaleAmount"] if to_numeric(row["AssessorLastSaleAmount"]) > min_sale_price else np.nan
        )
    elif previous_column == "YearBuilt":
        additional_object["last_sale_date"] = date_string(datetime(int(row["YearBuilt"]), 1, 1))

    return additional_object


def get_parcel_prepared_data(
    parcel_ids,
    df_raw_encoded,
    train_start_date,
    min_months_since_last_sale,
):
    import math
    import pandas as pd

    from datetime import datetime
    from dateutil.rrule import rrule, MONTHLY

    from freeholdforecast.common.utils import to_numeric, round_base

    def get_months(start_date, end_date):
        # handle 02/29 error
        if start_date.month == 2 and start_date.day == 29:
            start_date = start_date.replace(month=3, day=1)

        start_date = start_date.replace(day=1)

        end_date = end_date if pd.notna(end_date) else datetime.now()
        end_date = end_date.replace(day=1)

        return [
            x
            for x in rrule(
                dtstart=start_date,
                until=end_date,
                freq=MONTHLY,
            )
        ]

    prepared_data = []

    for parcel_id in parcel_ids:
        # months_since_year_built = 0
        total_sales = 0

        df_parcel_sales = df_raw_encoded.loc[df_raw_encoded.Parid == parcel_id].sort_values(
            by="last_sale_date", ascending=True, ignore_index=True
        )

        # df_parcel_sales["next_sale_date"] = df_parcel_sales.last_sale_date.shift(-1)
        # df_parcel_sales["next_sale_price"] = df_parcel_sales.last_sale_price.shift(-1)

        for row_index, row in df_parcel_sales.iterrows():
            months_since_last_sale = 0
            total_sales += 1
            has_next_sale_date = pd.notna(row.next_sale_date)

            months = get_months(row.last_sale_date, row.next_sale_date)
            total_months = len(months)

            for date_index, date in enumerate(months):
                if (total_months - 1) > date_index:
                    # if date >= train_start_date and months_since_last_sale >= min_months_since_last_sale:
                    if date >= train_start_date and months_since_last_sale >= (min_months_since_last_sale * 12):

                        def has_next_sale(date_diff, has_next_sale_date):
                            return 1 if date_index >= (total_months - date_diff - 1) and has_next_sale_date else 0
                            # return 1 if date_index == (total_months - date_diff - 1) and has_next_sale_date else 0

                        months_since_last_sale_floor = math.floor(months_since_last_sale / 12)
                        max_months_since_last_sale_floor = 25

                        prepared_data_object = {
                            "sale_in_3_months": has_next_sale(3, has_next_sale_date),
                            "sale_in_6_months": has_next_sale(6, has_next_sale_date),
                            "sale_in_12_months": has_next_sale(12, has_next_sale_date),
                            "next_sale_price": to_numeric(row.next_sale_price),
                            "next_sale_date": row.next_sale_date,
                            "total_sales": total_sales,
                            "date": date.replace(day=1),
                            "month": date.month,
                            "months_since_last_sale": months_since_last_sale,
                            # "months_since_year_built": months_since_year_built,
                            "months_since_last_sale_max": (
                                months_since_last_sale_floor
                                if months_since_last_sale_floor < max_months_since_last_sale_floor
                                else max_months_since_last_sale_floor
                            ),
                            "months_since_last_sale": months_since_last_sale_floor,
                            # "months_since_last_sale": math.floor(months_since_last_sale / 12),
                            # "months_since_year_built": math.floor(months_since_year_built / 12),
                        }

                        for column in list(df_raw_encoded.columns):
                            prepared_data_object[column] = row[column]

                        prepared_data.append(prepared_data_object)

                    months_since_last_sale += 1
                    # months_since_year_built += 1

                    # if date_index == total_months - 2:
                    #     months_since_last_sale = 0

    return pd.DataFrame(prepared_data)


def train_model(training_type, task, label_name, n_jobs, model_directory, X_train, y_train, X_test, y_test):
    import json
    import mlflow
    import os
    import shutil
    import time

    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.metrics import f1, precision, r2
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        precision_score,
        r2_score,
        roc_auc_score,
    )
    from sklearn.svm import OneClassSVM

    from freeholdforecast.common.utils import copy_directory_to_storage

    is_classification = training_type == "classification"
    time_left_for_this_task = 60 * task.fit_minutes
    per_run_time_limit = 60 * task.per_job_fit_minutes
    per_job_fit_memory_limit_mb = int(task.per_job_fit_memory_limit_gb * 1024)
    algorithm = "gradient_boosting"
    resampling_strategy = "cv"
    resampling_strategy_arguments = {
        "train_size": 0.67,
        "shuffle": False,
        "folds": 15,
    }

    mlflow_run = task.mlflow_client.create_run(task.mlflow_experiment.experiment_id)
    mlflow_run_id = mlflow_run.info.run_id

    with mlflow.start_run(mlflow_run_id, task.mlflow_experiment.experiment_id):
        task.mlflow_client.log_param(mlflow_run_id, "training_type", training_type)
        task.mlflow_client.log_param(mlflow_run_id, "label_name", label_name)
        task.mlflow_client.log_param(mlflow_run_id, "algorithm", algorithm)
        task.mlflow_client.log_param(mlflow_run_id, "resampling_strategy", resampling_strategy)
        task.mlflow_client.log_param(
            mlflow_run_id, "resampling_strategy_arguments", json.dumps(resampling_strategy_arguments)
        )
        task.mlflow_client.log_param(mlflow_run_id, "metric", "f1" if is_classification else "r2")
        # task.mlflow_client.log_param(mlflow_run_id, "train_years", task.train_years)
        task.mlflow_client.log_param(mlflow_run_id, "fit_jobs", n_jobs)
        task.mlflow_client.log_param(mlflow_run_id, "fit_minutes", task.fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_fit_minutes", task.per_job_fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_memory_limit_gb", task.per_job_fit_memory_limit_gb)

        if is_classification:
            model = AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit=per_job_fit_memory_limit_mb,
                n_jobs=n_jobs,
                metric=f1,
                initial_configurations_via_metalearning=0,
                include={"classifier": [algorithm]},
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments=resampling_strategy_arguments,
            )
        else:
            model = AutoSklearnRegressor(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit=per_job_fit_memory_limit_mb,
                n_jobs=n_jobs,
                metric=r2,
                initial_configurations_via_metalearning=0,
                include={"regressor": [algorithm]},
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments=resampling_strategy_arguments,
            )

        model.fit(X_train, y_train)

        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)

        time.sleep(10)
        mlflow.sklearn.log_model(model, model_directory)
        time.sleep(10)
        mlflow.sklearn.save_model(model, model_directory)
        copy_directory_to_storage("models", model_directory)

        if not task.is_local and len(X_test) > 0:
            y_pred = model.predict(X_test)

            if is_classification:
                f1_score_value = f1_score(y_test, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                average_precision_value = average_precision_score(y_test, y_pred)
                precision_value = precision_score(y_test, y_pred)
                roc_auc_value = roc_auc_score(y_test, y_pred)

                task.mlflow_client.log_metric(mlflow_run_id, "f1_score_value", f1_score_value)
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


def get_parcel_months_since_last_sale(last_sale_date, current_date):
    from freeholdforecast.common.utils import diff_month

    return diff_month(last_sale_date, current_date)


def get_parcel_months_since_last_sale_max(last_sale_date, current_date):
    months_since_last_sale_max = 25
    months_since_last_sale = get_parcel_months_since_last_sale(last_sale_date, current_date)
    return months_since_last_sale if months_since_last_sale < months_since_last_sale_max else months_since_last_sale_max


def get_parcel_months_since_year_built(year_built, current_date):
    import numpy as np
    import pandas as pd

    from datetime import datetime
    from freeholdforecast.common.utils import diff_month

    return (
        np.nan
        if pd.isna(year_built) or int(year_built) < 1800
        else diff_month(datetime(int(year_built), 1, 1), current_date)
    )
