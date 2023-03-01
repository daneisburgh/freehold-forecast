# def get_df_attom_recorder_additional_data(row, assessor_columns, recorder_columns):
#     import numpy as np
#     import pandas as pd

#     from datetime import datetime, timedelta

#     from freeholdforecast.common.utils import date_string

#     recorder_additional_object = {}

#     ignore_assessor_columns = [
#         "TaxYearAssessed",
#         "TaxAssessedValueTotal",
#         "TaxAssessedValueImprovements",
#         "TaxAssessedValueLand",
#         "PreviousAssessedValue",
#         "TaxMarketValueYear",
#         "TaxMarketValueTotal",
#         "TaxMarketValueImprovements",
#         "TaxMarketValueLand",
#         "TaxExemptionHomeownerFlag",
#     ]

#     for column in assessor_columns:
#         recorder_additional_object[column] = (
#             np.nan if column in recorder_columns or column in ignore_assessor_columns else row[column]
#         )

#     recorder_additional_object["InstrumentDate"] = np.nan
#     recorder_additional_object["TransferAmount"] = np.nan

#     # min_year_built = datetime.now().year - max_year_diff

#     if pd.notna(row["YearBuilt"]):
#         year_built = int(row["YearBuilt"])

#         # if year_built < min_year_built:
#         #     year_built = min_year_built

#         recorder_additional_object["InstrumentDate"] = date_string(datetime(year_built, 1, 1))

#     return recorder_additional_object


def get_df_additional_data(row, all_columns, ignore_columns):
    import numpy as np
    import pandas as pd

    from datetime import datetime, timedelta

    from freeholdforecast.common.utils import date_string

    additional_object = {}

    for column in all_columns:
        additional_object[column] = np.nan if column in ignore_columns else row[column]

    year_built = int(row["Year Built"])

    additional_object["last_sale_date"] = date_string(datetime(year_built, 1, 1))

    return additional_object


def get_parcel_prepared_data(
    parcel_ids,
    df_raw_encoded,
    train_start_date,
    min_months_since_last_sale,
    # max_months_since_last_sale,
):
    import pandas as pd

    from datetime import datetime
    from dateutil.rrule import rrule, MONTHLY

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

    # def has_next_sale(date_diff, has_next_sale_date):
    #     return 1 if date_index >= (total_months - date_diff - 1) and has_next_sale_date else 0

    prepared_data = []

    for parcel_id in parcel_ids:
        months_since_last_sale = 0
        months_since_year_built = 0

        df_parcel_sales = df_raw_encoded.loc[df_raw_encoded.Parid == parcel_id].sort_values(
            by="last_sale_date", ascending=True, ignore_index=True
        )

        # df_parcel_sales["next_sale_date"] = df_parcel_sales.last_sale_date.shift(-1)
        # df_parcel_sales["next_sale_price"] = df_parcel_sales.last_sale_price.shift(-1)

        for row_index, row in df_parcel_sales.iterrows():
            has_next_sale_date = pd.notna(row.next_sale_date)

            months = get_months(row.last_sale_date, row.next_sale_date)
            total_months = len(months)

            for date_index, date in enumerate(months):
                if (total_months - 1) > date_index:
                    if date >= train_start_date and months_since_last_sale >= min_months_since_last_sale:
                        # if date >= train_start_date:
                        prepared_data_object = {
                            # "sale_in_3_months": has_next_sale(3, has_next_sale_date),
                            "sale_in_3_months": 1 if date_index >= (total_months - 4) and has_next_sale_date else 0,
                            "next_sale_price": pd.to_numeric(row.next_sale_price, errors="coerce"),
                            "next_sale_date": row.next_sale_date,
                            "date": date.replace(day=1),
                            # "month": date.month,
                            "months_since_last_sale": months_since_last_sale,
                            "months_since_year_built": months_since_year_built,
                        }

                        for column in list(df_raw_encoded.columns):
                            prepared_data_object[column] = row[column]

                        prepared_data.append(prepared_data_object)

                    months_since_last_sale += 1
                    months_since_year_built += 1

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

    # task.logger = task._prepare_logger()  # reset logger

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

        # task.logger.info(f"Fitting model for {label_name}")
        time_left_for_this_task = 60 * task.fit_minutes
        per_run_time_limit = 60 * task.per_job_fit_minutes
        per_job_fit_memory_limit_mb = int(task.per_job_fit_memory_limit_gb * 1024)
        resampling_strategy = "cv"

        if is_classification:
            model = AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit=per_job_fit_memory_limit_mb,
                n_jobs=n_jobs,
                metric=f1,
                include={"classifier": ["gradient_boosting"]},
                initial_configurations_via_metalearning=0,
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments={
                    "train_size": 0.67,
                    "shuffle": True,
                    "folds": 3,
                },
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
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments={
                    "train_size": 0.67,
                    "shuffle": True,
                    "folds": 5,
                },
            )

        model.fit(X_train, y_train)

        # task.logger = task._prepare_logger()  # reset logger
        # task.logger.info(f"Saving model for {label_name}")

        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)

        time.sleep(10)
        mlflow.sklearn.log_model(model, model_directory)
        time.sleep(10)
        mlflow.sklearn.save_model(model, model_directory)

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


def get_parcel_months_since_last_sale(last_sale_date, current_date):
    from freeholdforecast.common.utils import diff_month

    return diff_month(last_sale_date, current_date)


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
