import gc
import mlflow
import numpy as np
import os
import pandas as pd
import pickle

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from pandarallel import pandarallel
from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from freeholdforecast.common.county_dfs import get_df_county
from freeholdforecast.common.static import encode_column, update_ready_data
from freeholdforecast.common.task import Task
from freeholdforecast.common.utils import copy_file_to_storage, file_exists, make_directory

pandarallel.initialize(progress_bar=(os.getenv("APP_ENV") == "local"), use_memory_fs=False)


class ETL_ML_Task(Task):
    def __init__(self):
        super().__init__()
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.county = "ohio-clermont"

    def launch(self):
        self.logger.info(f"Launching ETL ML task for {self.county}")
        self.get_df_raw()
        self.get_df_ready()
        self.get_df_encoded()
        self.train_model("will_sell_next_year")
        self.train_model("will_sell_next_two_years")
        self.logger.info(f"Finished ETL ML task for  {self.county}")

    def get_df_raw(self):
        self.logger.info("Retrieving data")

        raw_directory = os.path.join("data", "etl", "raw", self.today_date)
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, self.county + ".csv")

        if file_exists(raw_path):
            self.logger.info("Loading existing data")
            self.df_raw = pd.read_csv(raw_path, low_memory=False)
            self.df_raw.last_sale_date = pd.to_datetime(self.df_raw.last_sale_date)
        else:
            self.logger.info("Saving landing data")
            self.df_raw = get_df_county(self.today_date, self.county)
            self.df_raw.to_csv(raw_path, index=False)
            self.logger.info("Saving raw data")
            copy_file_to_storage("etl", raw_path)

        self.logger.info("Successfully retrieved data")

    def get_df_ready(self):
        self.logger.info("Preparing data")

        ready_directory = os.path.join("data", "etl", "ready", self.today_date)
        make_directory(ready_directory)

        ready_path = os.path.join(ready_directory, self.county + ".csv")

        if file_exists(ready_path):
            self.logger.info("Loading existing data")
            self.df_ready = pd.read_csv(ready_path, low_memory=False)
            self.df_ready.year = pd.to_datetime(self.df_ready.year)
        else:
            parcel_ids = list(self.df_raw.Parid.unique())
            self.logger.info(f"Total parcels: {len(parcel_ids)}")
            self.logger.info(f"Total sales: {len(self.df_raw)}")

            results = pd.DataFrame({"Parid": parcel_ids}).Parid.parallel_apply(
                update_ready_data, args=(self.df_raw.copy(),)
            )

            self.df_ready = pd.DataFrame(
                np.concatenate([x for x in results if len(x) > 0]),
                columns=[
                    "will_sell_next_year",
                    "will_sell_next_two_years",
                    "year",
                    "dates_since_last_sale",
                ]
                + list(self.df_raw.columns),
            )
            self.df_ready.year = pd.to_datetime(self.df_ready.year, format="%Y")
            self.df_ready.drop(columns=["last_sale_date"], inplace=True)
            self.df_ready = self.df_ready.loc[
                :, (self.df_ready != self.df_ready.iloc[0]).any()
            ]  # remove constant columns
            self.df_ready.to_csv(ready_path, index=False)

            self.logger.info("Saving prepared data")
            copy_file_to_storage("etl", ready_path)

        self.logger.info("Successfully prepared data")

    def get_df_encoded(self):

        self.logger.info("Encoding data")

        encoded_directory = os.path.join("data", "etl", "encoded", self.today_date, self.county)
        make_directory(encoded_directory)

        train_path = os.path.join(encoded_directory, "train.csv")
        test_path = os.path.join(encoded_directory, "test.csv")

        if file_exists(train_path) and file_exists(test_path):
            self.logger.info("Loading existing data")
            self.df_encoded_train = pd.read_csv(train_path, low_memory=False)
            self.df_encoded_test = pd.read_csv(test_path, low_memory=False)
        else:
            self.parid_label_encoder = LabelEncoder()
            self.parid_label_encoder.fit(self.df_ready["Parid"])
            results = pd.DataFrame({"column": self.df_ready.columns}).column.parallel_apply(
                encode_column,
                args=(
                    self.df_ready.copy(),
                    self.parid_label_encoder,
                ),
            )

            df_encoded = pd.DataFrame([x for x in results.to_numpy() if len(x) > 0]).transpose()
            df_encoded.columns = self.df_ready.columns

            test_split_year = datetime.now().year - 4
            min_train_year = test_split_year - 10
            max_train_year = test_split_year - 1
            min_test_year = test_split_year
            max_test_year = test_split_year + 1

            self.logger.info(f"Train years: {min_train_year}-{max_train_year}")
            self.logger.info(f"Test years: {min_test_year}-{max_test_year}")

            self.df_encoded_train = df_encoded.loc[
                (self.df_ready.year.dt.year >= min_train_year) & (self.df_ready.year.dt.year <= max_train_year)
            ]
            self.df_encoded_test = df_encoded.loc[
                (self.df_ready.year.dt.year >= min_test_year) & (self.df_ready.year.dt.year <= max_test_year)
            ]

            self.df_encoded_train.to_csv(train_path, index=False)
            self.df_encoded_test.to_csv(test_path, index=False)

        self.logger.info("Successfully encoded data")

    def train_model(self, label_name):
        self.logger.info(f"Training model for label {label_name}")

        experiment_name = f"{self.county} {label_name}"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        X_train = self.df_encoded_train.drop(columns=["will_sell_next_year", "will_sell_next_two_years"]).to_numpy()
        y_train = self.df_encoded_train[label_name].values

        X_test = self.df_encoded_test.drop(columns=["will_sell_next_year", "will_sell_next_two_years"]).to_numpy()
        y_test = self.df_encoded_test[label_name].values

        if hasattr(self, "df_raw"):
            del self.df_raw
        if hasattr(self, "df_ready"):
            del self.df_ready

        gc.collect()

        rus = RandomUnderSampler(sampling_strategy=0.2)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

        def log_y_label_stats(label_name, label_values):
            total_labels = len(label_values)
            sum_labels = sum(label_values)
            p_labels = sum_labels / total_labels * 100
            self.logger.info(f"{label_name}: {sum_labels}/{total_labels} ({p_labels:.2f}%)")

        log_y_label_stats("Train labels", y_train)
        log_y_label_stats("Train res labels", y_train_res)
        log_y_label_stats("Test labels", y_test)

        task_minutes = 60
        cpu_count = -1  # -1 means using all processors
        model_memory_limit_gb = 6

        if not hasattr(self, "models"):
            self.models = {"will_sell_next_year": None, "will_sell_next_two_years": None}

        self.models[label_name] = AutoSklearnClassifier(
            time_left_for_this_task=(60 * task_minutes),
            per_run_time_limit=(60 * 30),
            memory_limit=(1024 * model_memory_limit_gb),
            n_jobs=cpu_count,
            metric=roc_auc,
            include={"classifier": ["gradient_boosting"]},
            initial_configurations_via_metalearning=0,
        )

        self.models[label_name].fit(X_train_res, y_train_res, dataset_name=self.county)

        self.logger.info("Saving model")

        model_directory = os.path.join("data", "mli", "models", self.today_date, self.county)
        make_directory(model_directory)

        mlflow.sklearn.log_model(self.models[label_name], experiment_name)

        model_path = os.path.join(model_directory, f"model_{label_name}.pkl")

        with open(model_path, "wb") as model_file:
            pickle.dump(self.models[label_name], model_file)

        copy_file_to_storage("mli", model_path)

        self.logger = self._prepare_logger()
        self.logger.info(self.models[label_name].sprint_statistics())

        y_pred = self.models[label_name].predict(X_test, n_jobs=cpu_count)
        y_pred_proba = self.models[label_name].predict_proba(X_test, n_jobs=cpu_count)

        log_y_label_stats("Pred labels", y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision_value = precision_score(y_test, y_pred)
        roc_auc_value = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("tp", tp)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("precision", precision_value)
        mlflow.log_metric("roc_auc", roc_auc_value)

        self.logger.info(f"Precision: {precision_value:.2f}")
        self.logger.info(f"ROC AUC: {roc_auc_value}")
        self.logger.info("Classification report:\n" + classification_report(y_test, y_pred))
        self.logger.info(
            "Confusion matrix:\n"
            + pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True).to_string()
        )

        for pred_proba in [0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred = [1 if y[1] > pred_proba else 0 for y in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            precision_value = precision_score(y_test, y_pred)
            self.logger.info(f"Precision with proba {pred_proba}: {precision_value:.2f} (tp={tp}, fp={fp})")

        mlflow.end_run()
        self.logger.info(f"Successfully trained model for label {label_name}")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL_ML_Task()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
