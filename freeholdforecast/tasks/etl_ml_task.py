import gc
import mlflow
import os
import pandas as pd
import psutil
import random

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc
from datetime import datetime, timedelta
from imblearn.under_sampling import RandomUnderSampler
from pandarallel import pandarallel
from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from freeholdforecast.common.county_dfs import get_df_county
from freeholdforecast.common.task import Task
from freeholdforecast.common.static import get_parcel_ready_data
from freeholdforecast.common.utils import (
    copy_directory_to_storage,
    copy_file_to_storage,
    date_string,
    file_exists,
    make_directory,
)


cpu_count = psutil.cpu_count(logical=True)
is_local = os.getenv("APP_ENV") == "local"
label_names = ["transfer_in_6_months", "transfer_in_12_months", "transfer_in_24_months"]

pandarallel.initialize(nb_workers=cpu_count, progress_bar=is_local, use_memory_fs=False)


class ETL_ML_Task(Task):
    def __init__(self):
        super().__init__()
        self.county = "ohio-clermont"
        # self.run_date = date_string(datetime.now().replace(day=1))
        self.run_date = date_string((datetime.now() - timedelta(365 * 6)).replace(day=1))
        self.logger.info(f"Initialized task for {self.county} with run date {self.run_date}")

    def launch(self):
        self.logger.info(f"Launching task")
        self._get_df_raw()
        self._get_df_ready()
        self._get_df_encoded()

        for label_name in label_names:
            self._train_model(label_name)

        self.logger.info(f"Finished task")

    def _get_df_raw(self):
        self.logger.info("Retrieving data")

        raw_directory = os.path.join("data", "etl", "raw", self.run_date)
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, self.county + ".gz")

        if file_exists(raw_path):
            self.logger.info("Loading existing data")
            self.df_raw = pd.read_csv(raw_path, low_memory=False)
            self.df_raw.last_sale_date = pd.to_datetime(self.df_raw.last_sale_date)
        else:
            self.logger.info("Saving landing data")
            self.df_raw = get_df_county(self.run_date, self.county)
            self.logger.info("Saving raw data")
            self.df_raw.to_csv(raw_path, index=False)
            copy_file_to_storage("etl", raw_path)

        self.parcel_ids = list(self.df_raw.Parid.unique())
        self.logger.info(f"Total parcels: {len(self.parcel_ids)}")
        self.logger.info(f"Total sales: {len(self.df_raw)}")

        if is_local:
            self.parcel_ids = random.sample(self.parcel_ids, int(len(self.parcel_ids) * 0.1))

        self.logger.info("Successfully retrieved data")

    def _get_df_ready(self):
        self.logger.info("Preparing data")

        ready_directory = os.path.join("data", "etl", "ready", self.run_date)
        make_directory(ready_directory)

        ready_path = os.path.join(ready_directory, self.county + ".gz")

        if file_exists(ready_path):
            self.logger.info("Loading existing data")
            self.df_ready = pd.read_csv(ready_path, low_memory=False)
            self.df_ready.date = pd.to_datetime(self.df_ready.date)
        else:
            self.df_ready = pd.concat(
                pd.DataFrame({"Parid": self.parcel_ids})
                .Parid.parallel_apply(get_parcel_ready_data, args=(self.df_raw.copy(),))
                .tolist(),
                copy=False,
            )

            self.logger.info("Saving prepared data")
            self.df_ready.to_csv(ready_path, index=False)
            copy_file_to_storage("etl", ready_path)

        if hasattr(self, "df_raw"):
            del self.df_raw

        gc.collect()

        self.logger.info("Successfully prepared data")

    def _get_df_encoded(self):
        self.logger.info("Encoding data")

        test_start_date = datetime.strptime(self.run_date, "%Y-%m-%d")
        test_end_date = test_start_date.replace(year=(test_start_date.year + 2))

        test_is_same_year = test_start_date.month > 1
        train_end_date = test_start_date.replace(
            year=(test_start_date.year if test_is_same_year else test_start_date.year - 1),
            month=(test_start_date.month - 1 if test_is_same_year else 12),
        )
        train_start_date = train_end_date.replace(year=(train_end_date.year - 10))

        self.logger.info(f"Train dates: {date_string(train_start_date)} to {date_string(train_end_date)}")
        self.logger.info(f"Test dates: {date_string(test_start_date)} to {date_string(test_end_date)}")

        parcel_ids = self.df_ready.Parid.values
        parcel_ids_encoder = LabelEncoder()
        parcel_ids_encoder.fit(parcel_ids)

        df_ready_encoded = (
            self.df_ready.drop(columns=["Parid"]).astype(str).parallel_apply(LabelEncoder().fit_transform)
        )
        df_ready_encoded["Parid"] = parcel_ids_encoder.transform(parcel_ids)

        date_columns = ["date", "last_sale_date"]

        self.df_train = df_ready_encoded.loc[
            (self.df_ready.date >= train_start_date) & (self.df_ready.date <= train_end_date)
        ].drop(columns=date_columns)
        self.df_test = df_ready_encoded.loc[
            (self.df_ready.date >= test_start_date) & (self.df_ready.date <= test_end_date)
        ].drop(columns=date_columns)

        self.X_train = self.df_train.drop(columns=label_names).to_numpy()
        self.X_test = self.df_test.drop(columns=label_names).to_numpy()

        if hasattr(self, "df_ready"):
            del self.df_ready

        gc.collect()

        self.logger.info("Successfully encoded data")

    def _train_model(self, label_name):
        self.logger.info(f"Training model for {label_name}")

        gc.collect()

        mlflow.set_experiment(f"/Shared/{self.county}")
        mlflow.start_run(run_name=label_name)

        y_train = self.df_train[label_name].values
        y_test = self.df_test[label_name].values

        rus = RandomUnderSampler(sampling_strategy=0.2)
        X_train_res, y_train_res = rus.fit_resample(self.X_train, y_train)

        def log_y_label_stats(label_name, label_values):
            total_labels = len(label_values)
            sum_labels = sum(label_values)
            p_labels = sum_labels / total_labels * 100
            self.logger.info(f"{label_name}: {sum_labels}/{total_labels} ({p_labels:.2f}%)")

        log_y_label_stats("Train labels", y_train)
        log_y_label_stats("Train res labels", y_train_res)
        log_y_label_stats("Test labels", y_test)

        task_minutes = 30
        model_memory_limit_gb = 8

        if not hasattr(self, "models"):
            self.models = {}

        self.models[label_name] = AutoSklearnClassifier(
            time_left_for_this_task=(60 * task_minutes),
            per_run_time_limit=(60 * 30),
            memory_limit=(1024 * model_memory_limit_gb),
            n_jobs=cpu_count,
            metric=roc_auc,
            include={"classifier": ["gradient_boosting"]},
            initial_configurations_via_metalearning=0,
        )

        self.logger.info("Fitting model")

        self.models[label_name].fit(X_train_res, y_train_res, dataset_name=self.county)

        self.logger = self._prepare_logger()  # reset logger
        self.logger.info("Saving model")

        model_directory = os.path.join("data", "mli", "models", self.run_date, self.county, label_name)
        mlflow.sklearn.log_model(self.models[label_name], model_directory)
        mlflow.sklearn.save_model(self.models[label_name], model_directory)
        copy_directory_to_storage("mli", model_directory)

        self.logger.info(self.models[label_name].sprint_statistics())

        y_pred = self.models[label_name].predict(self.X_test, n_jobs=cpu_count)
        y_pred_proba = self.models[label_name].predict_proba(self.X_test, n_jobs=cpu_count)

        log_y_label_stats("Pred labels", y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision_value = precision_score(y_test, y_pred)
        roc_auc_value = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("tp", tp)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("precision", precision_value)
        mlflow.log_metric("roc_auc", roc_auc_value)

        self.logger.info(f"Precision: {precision_value:.2f}")
        self.logger.info(f"ROC AUC: {roc_auc_value:.2f}")
        self.logger.info("Classification report:\n" + classification_report(y_test, y_pred))
        self.logger.info(
            "Confusion matrix:\n"
            + pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True).to_string()
        )

        for pred_proba in [0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred = [1 if y[1] > pred_proba else 0 for y in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            precision_value = precision_score(y_test, y_pred)
            self.logger.info(
                f"Precision with proba {pred_proba}: {precision_value:.2f} (tp={tp}, fp={fp}, tn={tn}, fn={fn})"
            )

        mlflow.end_run()
        self.logger.info(f"Successfully trained model for {label_name}")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL_ML_Task()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
