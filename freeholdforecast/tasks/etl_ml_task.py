import gc
import mlflow
import numpy as np
import os
import pandas as pd
import pickle
import random

from datetime import datetime, timedelta
from functools import partial
from imblearn.under_sampling import RandomUnderSampler
from mlflow.tracking import MlflowClient
from multiprocessing import Pool, Process
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import OrdinalEncoder

from freeholdforecast.common.county_dfs import get_df_county
from freeholdforecast.common.task import Task
from freeholdforecast.common.static import get_parcel_prepared_data, train_model
from freeholdforecast.common.utils import (
    copy_file_to_storage,
    date_string,
    file_exists,
    make_directory,
)


class ETL_ML_Task(Task):
    def __init__(self):
        super().__init__()
        self.county = "ohio-hamilton"
        # self.run_date = date_string(datetime.now().replace(day=1))
        # self.run_date = date_string((datetime.now() - timedelta(365 * 6)).replace(day=1))
        self.run_date = "2022-07-01"
        self.logger.info(f"Initializing task for {self.county} with run date {self.run_date}")

        self.test_start_date = datetime.strptime(self.run_date, "%Y-%m-%d")
        self.test_end_date = self.test_start_date

        test_is_same_year = self.test_start_date.month > 1
        self.train_end_date = self.test_start_date.replace(
            year=(self.test_start_date.year if test_is_same_year else self.test_start_date.year - 1),
            month=(self.test_start_date.month - 1 if test_is_same_year else 12),
        )

        self.train_years = 2
        self.train_start_date = self.train_end_date.replace(year=(self.train_end_date.year - self.train_years))

        self.logger.info(f"Train years: {self.train_years}")
        self.logger.info(f"Train dates: {date_string(self.train_start_date)} to {date_string(self.train_end_date)}")
        self.logger.info(f"Test dates: {date_string(self.test_start_date)} to {date_string(self.test_end_date)}")

        self.classification_label_names = [
            "sale_in_3_months",
            # "sale_in_6_months",
            # "sale_in_12_months",
        ]

        self.regression_label_names = [
            "next_sale_amount",
            # "next_sale_months",
        ]

        self.label_names = self.classification_label_names + self.regression_label_names
        self.rus = RandomUnderSampler(sampling_strategy=0.1, random_state=1234)

    def launch(self):
        self.logger.info(f"Launching task")

        self._get_df_raw()
        self._get_df_encoded()
        self._get_df_prepared()
        self._train_models()

        self.logger.info(f"Finished task")

    def _get_df_raw(self):
        raw_directory = os.path.join("data", "etl", self.run_date, self.county, "1-raw")
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, "raw.gz")

        if file_exists(raw_path):
            self.logger.info("Loading existing raw data")
            self.df_raw = pd.read_parquet(raw_path)
            self.df_raw["last_sale_amount"] = pd.to_numeric(self.df_raw.last_sale_amount)
        else:
            self.logger.info("Retrieving landing data")
            landing_directory = os.path.join("data", "etl", self.run_date, self.county, "0-landing")
            make_directory(landing_directory)
            self.df_raw = get_df_county(self.county, landing_directory)

            self.logger.info("Saving raw data")
            self.df_raw.to_parquet(raw_path, index=False)
            copy_file_to_storage("etl", raw_path)

        self.df_raw.last_sale_date = pd.to_datetime(self.df_raw.last_sale_date)
        self.logger.info(f"Total parcels: {len(self.df_raw.Parid.unique())}")
        self.logger.info(f"Total sales: {len(self.df_raw)}")

    def _get_df_encoded(self):
        encoded_directory = os.path.join("data", "etl", self.run_date, self.county, "2-encoded")
        make_directory(encoded_directory)

        encoded_path = os.path.join(encoded_directory, "raw-encoded.gz")

        # self.non_encoded_date_columns = ["last_sale_date"]
        # self.non_encoded_numeric_columns = ["last_sale_amount", "Sale Price", "Land Value", "Building Value"]
        # self.non_encoded_columns = self.non_encoded_date_columns + self.non_encoded_numeric_columns
        self.non_encoded_columns = ["last_sale_date", "last_sale_amount"]

        if file_exists(encoded_path):
            self.logger.info("Loading existing encoded data")
            self.df_raw_encoded = pd.read_parquet(
                encoded_path,
            )
        else:
            self.logger.info("Encoding raw data")
            df_raw_features = self.df_raw.drop(columns=self.non_encoded_columns)

            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_encoder.fit(df_raw_features)

            self.df_raw_encoded = pd.DataFrame(
                self.ordinal_encoder.transform(df_raw_features),
                columns=df_raw_features.columns,
                index=df_raw_features.index,
            )

            for column in self.non_encoded_columns:
                self.df_raw_encoded[column] = self.df_raw[column]

            # for column in self.non_encoded_numeric_columns:
            #     self.df_raw_encoded[column] = pd.to_numeric(self.df_raw[column], errors="coerce")

            self.logger.info("Saving encoded data")
            self.df_raw_encoded.to_parquet(encoded_path, index=False)
            copy_file_to_storage("etl", encoded_path)

            ordinal_encoder_path = os.path.join(encoded_directory, "ordinal-encoder.pkl")
            with open(ordinal_encoder_path, "wb") as ordinal_encoder_file:
                pickle.dump(self.ordinal_encoder, ordinal_encoder_file)
            copy_file_to_storage("etl", ordinal_encoder_path)

    def _get_df_prepared(self):
        gc.collect()
        prepared_directory = os.path.join("data", "etl", self.run_date, self.county, "3-prepared")
        make_directory(prepared_directory)

        prepared_path = os.path.join(prepared_directory, self.county + ".gz")

        if file_exists(prepared_path):
            self.logger.info("Loading existing prepared data")
            self.df_prepared = pd.read_parquet(prepared_path)
            self.df_prepared.date = pd.to_datetime(self.df_prepared.date)
        else:
            self.logger.info("Preparing data")

            parcel_ids = list(self.df_raw_encoded.Parid.unique())

            # if self.is_local:
            #     parcel_ids = random.sample(parcel_ids, int(len(parcel_ids) * 0.5))

            parcel_ids_split = np.array_split(parcel_ids, self.cpu_count)

            self.df_prepared = pd.concat(
                Pool(self.cpu_count).map(
                    partial(
                        get_parcel_prepared_data,
                        df_raw_encoded=self.df_raw_encoded,
                        train_start_date=self.train_start_date,
                    ),
                    parcel_ids_split,
                ),
                copy=False,
                ignore_index=True,
            ).drop_duplicates(ignore_index=True)

            for label_name in self.label_names:
                self.df_prepared[label_name] = pd.to_numeric(self.df_prepared[label_name])

            self.logger.info("Saving prepared data")
            self.df_prepared.to_parquet(prepared_path, index=False)
            copy_file_to_storage("etl", prepared_path)

        self.logger.info("Splitting data")
        # prepared_drop_columns = ["date", "last_sale_date", "last_sale_amount"]
        prepared_drop_columns = ["date"] + self.non_encoded_columns

        self.df_train = self.df_prepared.loc[
            (self.df_prepared.date >= self.train_start_date) & (self.df_prepared.date <= self.train_end_date)
        ].drop(columns=prepared_drop_columns)

        self.df_test = self.df_prepared.loc[
            (self.df_prepared.date >= self.test_start_date)
            & (self.df_prepared.date <= self.test_end_date)
            & (self.df_prepared.months_since_last_sale > 1)
        ].drop(columns=prepared_drop_columns)

    def _train_models(self):
        gc.collect()

        self.mlflow_client = MlflowClient()
        self.mlflow_experiment = mlflow.set_experiment(f"/Shared/{self.county}")

        # self.fit_jobs = 5 if self.is_local else int(self.cpu_count / len(self.label_names))
        # self.total_jobs = 10

        self.fit_minutes = 120 if self.is_local else 180
        self.per_job_fit_minutes = 10 if self.is_local else 30
        self.per_job_fit_memory_limit_mb = (3 if self.is_local else 8) * 1024

        # self.logger.info(f"Total labels: {len(self.label_names)}")
        # self.logger.info(f"Jobs per label: {self.fit_jobs}")
        self.logger.info(f"Total fit minutes: {self.fit_minutes}")
        self.logger.info(f"Job fit minutes: {self.per_job_fit_minutes}")
        self.logger.info(f"Job memory limit (MB): {self.per_job_fit_memory_limit_mb}")

        self.model_directories = {}
        procs = []

        def log_y_label_stats(label_name, label_values):
            total_labels = len(label_values)
            sum_labels = sum(label_values)
            p_labels = sum_labels / total_labels * 100
            self.logger.info(f"{label_name}: {sum_labels}/{total_labels} ({p_labels:.2f}%)")

        for label_name in self.label_names:
            is_classification = label_name in self.classification_label_names
            self.fit_jobs = 8 if is_classification else 4

            self.logger.info(f"Initialize AutoML for {label_name} with {self.fit_jobs} jobs")
            self.model_directories[label_name] = os.path.join("data", "models", self.run_date, self.county, label_name)

            df_train = self.df_train.loc[self.df_train[label_name].notna()]
            df_test = self.df_test.loc[self.df_test[label_name].notna()]

            if is_classification:
                X_train = df_train.drop(columns=self.label_names).to_numpy()
                X_test = df_test.drop(columns=self.label_names).to_numpy()

                y_train = df_train[label_name].values
                y_test = df_test[label_name].values

                log_y_label_stats(f"Train labels", y_train)

                X_train_res, y_train_res = self.rus.fit_resample(X_train, y_train)
                log_y_label_stats(f"Train labels resampled", y_train_res)

                if len(y_test) > 0:
                    log_y_label_stats(f"Test labels", y_test)

                proc = Process(
                    target=train_model,
                    args=(
                        "classification",
                        self,
                        label_name,
                        self.model_directories[label_name],
                        X_train_res,
                        y_train_res,
                        X_test,
                        y_test,
                    ),
                )
            else:
                X_train = df_train.drop(columns=self.label_names).to_numpy()
                X_test = df_test.drop(columns=self.label_names).to_numpy()

                y_train = df_train[label_name].values
                y_test = df_test[label_name].values

                proc = Process(
                    target=train_model,
                    args=(
                        "regression",
                        self,
                        label_name,
                        self.model_directories[label_name],
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                    ),
                )

            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        for label_name in self.label_names:
            model = mlflow.sklearn.load_model(self.model_directories[label_name])
            self.logger.info(f"Model {label_name} " + model.sprint_statistics())

            df_train = self.df_train.loc[self.df_train[label_name].notna()]
            df_test = self.df_test.loc[self.df_test[label_name].notna()]

            for index_of_df_temp, df_temp in enumerate([df_train, df_test]):
                if len(df_temp) > 0:
                    type_of_df_temp = "training" if index_of_df_temp == 0 else "testing"
                    self.logger.info(f"Metrics with {type_of_df_temp} data:")
                    is_classification = label_name in self.classification_label_names

                    if is_classification:
                        X_test = df_temp.drop(columns=self.label_names).to_numpy()
                        y_test = df_temp[label_name].values

                        if type_of_df_temp == "training":
                            X_test, y_test = self.rus.fit_resample(X_test, y_test)

                        y_pred = model.predict(X_test)
                        log_y_label_stats(f"Pred labels", y_pred)

                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        average_precision_value = average_precision_score(y_test, y_pred)
                        precision_value = precision_score(y_test, y_pred)
                        roc_auc_value = roc_auc_score(y_test, y_pred)

                        self.logger.info(f"Precision: {precision_value:.2f}")
                        self.logger.info(f"Average precision: {average_precision_value:.2f}")
                        self.logger.info(f"ROC AUC: {roc_auc_value:.2f}")
                        self.logger.info(f"Classification report:\n" + classification_report(y_test, y_pred))
                        self.logger.info(
                            f"Confusion matrix:\n"
                            + pd.crosstab(
                                y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True
                            ).to_string()
                        )

                        y_pred_proba = model.predict_proba(X_test)

                        for pred_proba in [0.5, 0.6, 0.7, 0.8, 0.9]:
                            y_pred = [1 if y[1] > pred_proba else 0 for y in y_pred_proba]
                            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                            average_precision_value = average_precision_score(y_test, y_pred)
                            precision_value = precision_score(y_test, y_pred)
                            self.logger.info(
                                f"AP/P proba {pred_proba}: {average_precision_value:.2f}/{precision_value:.2f} (tp={tp}, fp={fp}, tn={tn}, fn={fn})"
                            )
                    else:
                        X_test = df_temp.drop(columns=self.label_names).to_numpy()
                        y_test = df_temp[label_name].values
                        y_pred = model.predict(X_test)

                        r2_value = r2_score(y_test, y_pred)
                        mae_value = mean_absolute_error(y_test, y_pred)
                        mape_value = mean_absolute_percentage_error(y_test, y_pred)

                        self.logger.info(f"R2: {r2_value:.2f}")
                        self.logger.info(f"MAE: {mae_value:.2f}")
                        self.logger.info(f"MAPE: {mape_value:.2f}")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL_ML_Task()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
