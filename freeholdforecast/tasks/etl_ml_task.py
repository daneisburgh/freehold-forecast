import gc
import mlflow
import os
import pandas as pd
import pickle
import psutil
import random

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import average_precision
from datetime import datetime, timedelta
from imblearn.under_sampling import RandomUnderSampler
from pandarallel import pandarallel
from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from freeholdforecast.common.county_dfs import get_df_county
from freeholdforecast.common.task import Task
from freeholdforecast.common.static import get_parcel_prepared_data
from freeholdforecast.common.utils import (
    copy_directory_to_storage,
    copy_file_to_storage,
    date_string,
    file_exists,
    make_directory,
)


is_local = os.getenv("APP_ENV") == "local"
cpu_count = psutil.cpu_count(logical=True)
total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
label_names = ["transfer_in_6_months", "transfer_in_12_months", "transfer_in_24_months"]

pandarallel.initialize(nb_workers=cpu_count, progress_bar=is_local, use_memory_fs=False, verbose=1)


class ETL_ML_Task(Task):
    def __init__(self):
        super().__init__()
        self.county = "ohio-hamilton"
        # self.run_date = date_string(datetime.now().replace(day=1))
        self.run_date = date_string((datetime.now() - timedelta(365 * 6)).replace(day=1))
        self.logger.info(f"Initialized task for {self.county} with run date {self.run_date}")

    def launch(self):
        self.logger.info(f"Launching task")

        self._get_df_raw_encoded()
        self._get_df_prepared()

        for label_name in label_names:
            self._train_model(label_name)

        self.logger.info(f"Finished task")

    def _get_df_raw_encoded(self):
        self.logger.info("Retrieving and encoding data")

        raw_directory = os.path.join("data", "etl", self.run_date, self.county, "1-raw")
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, "raw.gz")

        if file_exists(raw_path):
            self.logger.info("Loading existing raw data")
            self.df_raw = pd.read_parquet(
                raw_path,
            )
        else:
            self.logger.info("Saving landing data")
            landing_directory = os.path.join("data", "etl", self.run_date, self.county, "0-landing")
            make_directory(landing_directory)
            self.df_raw = get_df_county(self.county, landing_directory)

            self.logger.info("Saving raw data")
            self.df_raw.to_parquet(raw_path, index=False)
            copy_file_to_storage("etl", raw_path)

        self.df_raw.last_sale_date = pd.to_datetime(self.df_raw.last_sale_date)
        self.logger.info(f"Total parcels: {len(self.df_raw.Parid.unique())}")
        self.logger.info(f"Total sales: {len(self.df_raw)}")

        encoded_path = os.path.join(raw_directory, "raw-encoded.gz")

        if file_exists(encoded_path):
            self.logger.info("Loading existing encoded data")
            self.df_raw_encoded = pd.read_parquet(
                encoded_path,
            )
        else:
            self.logger.info("Encoding raw data")

            df_raw_features = self.df_raw.drop(columns=["last_sale_date"])
            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_encoder.fit(df_raw_features)
            self.df_raw_encoded = pd.DataFrame(
                self.ordinal_encoder.transform(df_raw_features),
                columns=df_raw_features.columns,
                index=df_raw_features.index,
            )
            self.df_raw_encoded["last_sale_date"] = self.df_raw["last_sale_date"]

            self.logger.info("Saving encoded data")
            self.df_raw_encoded.to_parquet(encoded_path, index=False)
            copy_file_to_storage("etl", encoded_path)

            ordinal_encoder_path = os.path.join(raw_directory, "ordinal-encoder.pkl")
            with open(ordinal_encoder_path, "wb") as ordinal_encoder_file:
                pickle.dump(self.ordinal_encoder, ordinal_encoder_file)
            copy_file_to_storage("etl", ordinal_encoder_path)

    def _get_df_prepared(self):
        gc.collect()
        prepared_directory = os.path.join("data", "etl", self.run_date, self.county, "2-prepared")
        make_directory(prepared_directory)

        prepared_path = os.path.join(prepared_directory, self.county + ".gz")

        if file_exists(prepared_path):
            self.logger.info("Loading existing prepared data")
            self.df_prepared = pd.read_parquet(
                prepared_path,
            )
            self.df_prepared.date = pd.to_datetime(self.df_prepared.date)
        else:
            self.logger.info("Preparing data")
            parcel_ids = list(self.df_raw_encoded.Parid.unique())

            if is_local:
                parcel_ids = random.sample(parcel_ids, int(len(parcel_ids) * 0.25))

            self.df_prepared = pd.concat(
                pd.DataFrame({"Parid": parcel_ids})
                .Parid.parallel_apply(get_parcel_prepared_data, args=(self.df_raw_encoded,))
                .tolist(),
                copy=False,
                ignore_index=True,
            )

            self.logger.info("Saving prepared data")
            self.df_prepared.to_parquet(prepared_path, index=False)
            copy_file_to_storage("etl", prepared_path)

        self.logger.info("Splitting data")

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

        date_columns = ["date", "last_sale_date"]

        self.df_train = self.df_prepared.loc[
            (self.df_prepared.date >= train_start_date) & (self.df_prepared.date <= train_end_date)
        ].drop(columns=date_columns)
        self.df_test = self.df_prepared.loc[
            (self.df_prepared.date >= test_start_date) & (self.df_prepared.date <= test_end_date)
        ].drop(columns=date_columns)

        self.X_train = self.df_train.drop(columns=label_names).to_numpy()
        self.X_test = self.df_test.drop(columns=label_names).to_numpy()

    def _train_model(self, label_name):
        gc.collect()
        self.logger.info(f"Training model for {label_name}")

        mlflow.set_experiment(f"/Shared/{self.county}")
        mlflow.start_run(run_name=label_name)

        y_train = self.df_train[label_name].values
        y_test = self.df_test[label_name].values

        rus = RandomUnderSampler(sampling_strategy=0.1)
        X_train_res, y_train_res = rus.fit_resample(self.X_train, y_train)

        def log_y_label_stats(label_name, label_values):
            total_labels = len(label_values)
            sum_labels = sum(label_values)
            p_labels = sum_labels / total_labels * 100
            self.logger.info(f"{label_name}: {sum_labels}/{total_labels} ({p_labels:.2f}%)")

        log_y_label_stats("Train labels", y_train)
        log_y_label_stats("Train res labels", y_train_res)
        log_y_label_stats("Test labels", y_test)

        if not hasattr(self, "models"):
            self.models = {}

        fit_minutes = 120
        per_model_fit_minutes = 60
        per_model_memory_limit_mb = int(total_memory_mb / cpu_count)

        self.logger.info(f"Fit minutes: {fit_minutes}")
        self.logger.info(f"Per model fit minutes: {per_model_fit_minutes}")
        self.logger.info(f"Per model memory limit (MB): {per_model_memory_limit_mb}")

        self.models[label_name] = AutoSklearnClassifier(
            time_left_for_this_task=(60 * fit_minutes),
            per_run_time_limit=(60 * per_model_fit_minutes),
            memory_limit=per_model_memory_limit_mb,
            n_jobs=cpu_count,
            metric=average_precision,
            include={"classifier": ["gradient_boosting"]},
            initial_configurations_via_metalearning=0,
        )

        self.logger.info("Fitting model")

        self.models[label_name].fit(X_train_res, y_train_res)

        self.logger = self._prepare_logger()  # reset logger
        self.logger.info("Saving model")

        model_directory = os.path.join("data", "models", self.run_date, self.county, label_name)
        mlflow.sklearn.log_model(self.models[label_name], model_directory)
        mlflow.sklearn.save_model(self.models[label_name], model_directory)
        copy_directory_to_storage("models", model_directory)

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


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL_ML_Task()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
