def get_parcel_prepared_data(parcel_id, task):
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
        task.df_raw_encoded.loc[task.df_raw_encoded.Parid == parcel_id]
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
                    "transfer_in_3_months": is_transfered(3, has_next_sale_date),
                    "transfer_in_6_months": is_transfered(6, has_next_sale_date),
                    "transfer_in_12_months": is_transfered(12, has_next_sale_date),
                    "transfer_in_24_months": is_transfered(24, has_next_sale_date),
                    "date": date.replace(day=1),
                    "year": date.year,
                    "month": date.month,
                    "dates_since_last_sale": dates_since_last_sale,
                }

                for column in list(task.df_raw_encoded.columns):
                    prepared_data_object[column] = row[column]

                if date >= task.train_start_date:
                    prepared_data.append(prepared_data_object)

                dates_since_last_sale += 1

                if date_index == total_dates - 2:
                    dates_since_last_sale = 0

    return pd.DataFrame(prepared_data)


def train_model(task, label_name, model_directory, X_train, y_train, X_test, y_test):
    import mlflow
    import os
    import shutil

    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.metrics import roc_auc
    from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, roc_auc_score

    from freeholdforecast.common.utils import copy_directory_to_storage

    mlflow_run = task.mlflow_client.create_run(task.mlflow_experiment.experiment_id)
    mlflow_run_id = mlflow_run.info.run_id

    with mlflow.start_run(mlflow_run_id, task.mlflow_experiment.experiment_id):
        task.mlflow_client.log_param(mlflow_run_id, "label_name", label_name)
        task.mlflow_client.log_param(mlflow_run_id, "metric", "roc_auc")
        task.mlflow_client.log_param(mlflow_run_id, "train_years", task.train_years)
        task.mlflow_client.log_param(mlflow_run_id, "fit_jobs", task.fit_jobs)
        task.mlflow_client.log_param(mlflow_run_id, "fit_minutes", task.fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_fit_minutes", task.per_job_fit_minutes)
        task.mlflow_client.log_param(mlflow_run_id, "per_job_memory_limit_mb", task.per_job_fit_memory_limit_mb)

        task.logger.info(f"Fitting model for {label_name}")
        automl = AutoSklearnClassifier(
            time_left_for_this_task=(60 * task.fit_minutes),
            per_run_time_limit=(60 * task.per_job_fit_minutes),
            memory_limit=task.per_job_fit_memory_limit_mb,
            n_jobs=task.fit_jobs,
            metric=roc_auc,
            include={"classifier": ["gradient_boosting"]},
            initial_configurations_via_metalearning=0,
        )
        automl.fit(X_train, y_train)

        task.logger = task._prepare_logger()  # reset logger
        task.logger.info(f"Saving model for {label_name}")

        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)

        mlflow.sklearn.log_model(automl, model_directory)
        mlflow.sklearn.save_model(automl, model_directory)

        if os.getenv("APP_ENV") != "local":
            copy_directory_to_storage("models", model_directory)

        y_pred = automl.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        average_precision_value = average_precision_score(y_test, y_pred)
        precision_value = precision_score(y_test, y_pred)
        roc_auc_value = roc_auc_score(y_test, y_pred)

        task.mlflow_client.log_metric(mlflow_run_id, "tp", tp)
        task.mlflow_client.log_metric(mlflow_run_id, "fp", fp)
        task.mlflow_client.log_metric(mlflow_run_id, "average_precision", average_precision_value)
        task.mlflow_client.log_metric(mlflow_run_id, "precision", precision_value)
        task.mlflow_client.log_metric(mlflow_run_id, "roc_auc", roc_auc_value)

        mlflow.end_run()
