import gc
import googlemaps
import json
import math
import mlflow
import numpy as np
import os
import pandas as pd
import pickle
import pytz
import random
import sqlalchemy

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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

# from freeholdforecast.common.attom_dfs import get_df_attom

from freeholdforecast.common.county_dfs import get_df_county, get_df_state
from freeholdforecast.common.task import Task
from freeholdforecast.common.static import (
    get_parcel_months_since_last_sale,
    get_parcel_months_since_year_built,
    get_parcel_prepared_data,
    train_model,
)
from freeholdforecast.common.utils import (
    copy_directory_to_storage,
    copy_file_to_storage,
    date_string,
    file_exists,
    make_directory,
)


class ETL_ML_Task(Task):
    def __init__(self, state: str, run_date=None):
        super().__init__()
        self.state = state
        self.run_date = run_date if run_date is not None else date_string(datetime.now())
        self.logger.info(f"Initializing task for {self.state} with run date {self.run_date}")

        self.test_date = datetime.strptime(self.run_date, "%Y-%m-%d")
        self.prediction_period_start_date = self.test_date + relativedelta(months=+1)
        self.prediction_period_end_date = self.test_date + relativedelta(months=+3)

        test_is_same_year_as_train = self.test_date.month > 2
        self.train_end_date = self.test_date.replace(
            year=(self.test_date.year if test_is_same_year_as_train else self.test_date.year - 1),
            month=(self.test_date.month - 2 if test_is_same_year_as_train else 12),
        )

        self.train_years = 1
        self.train_start_date = self.train_end_date.replace(year=(self.train_end_date.year - self.train_years))

        self.logger.info(f"Train years: {self.train_years}")
        self.logger.info(f"Train dates: {date_string(self.train_start_date)} to {date_string(self.train_end_date)}")
        self.logger.info(f"Test date: {date_string(self.test_date)}")
        self.logger.info(
            f"Prediction period: {date_string(self.prediction_period_start_date)} to {date_string(self.prediction_period_end_date)}"
        )

        self.min_months_since_last_sale = 12 * 20
        # self.max_months_since_last_sale = 12 * 100
        # self.min_sale_price_quantile = 0.10
        # self.max_sale_price_quantile = 0.99

        self.classification_label_names = ["sale_in_3_months"]
        self.regression_label_names = ["next_sale_price"]
        self.label_names = self.classification_label_names + self.regression_label_names

        self.classification_proba_threshold = 0.5
        self.logger.info(f"Classification probability threshold: {self.classification_proba_threshold}")

        # self.non_encoded_columns = ["last_sale_date", "last_sale_price", "next_sale_date", "next_sale_price"]
        # self.non_encoded_columns = ["last_sale_date", "last_sale_price", "Year Built"]
        # self.non_encoded_columns = ["last_sale_date", "last_sale_price"]

        self.numeric_columns = [
            "House #",
            "Livable Sqft",
            "Stories",
            "Year Built",
            "Sale Price",
            "Building Value",
            "Land Value",
            "same_owner",
            "business_owner",
            "next_same_owner",
            "next_business_owner",
            "last_sale_price",
            "next_sale_price",
        ]

        self.datetime_columns = ["last_sale_date", "next_sale_date"]

        self.non_encoded_columns = self.numeric_columns + self.datetime_columns

        self.model_directories = {}

        for label_name in self.label_names:
            self.model_directories[label_name] = os.path.join("data", "models", self.run_date, self.state, label_name)

        self.rus_strategy = 0.05  # BEST RUN 0.05
        self.rus = RandomUnderSampler(sampling_strategy=self.rus_strategy, random_state=1234)

    def launch(self):
        self.logger.info(f"Launching task")

        self._get_df_raw()
        self._get_df_encoded()
        self._get_df_prepared()
        self._train_models()
        self._get_df_predictions()

        self.logger.info(f"Finished task")

    def _get_df_raw(self):
        raw_directory = os.path.join("data", "etl", self.run_date, self.state, "1-raw")
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, "raw.gz")

        if file_exists(raw_path):
            self.logger.info("Loading existing raw data")
            self.df_raw = pd.read_parquet(raw_path)
        else:
            self.logger.info("Retrieving landing data")
            landing_directory = os.path.join("data", "etl", self.run_date, self.state, "0-landing")
            make_directory(landing_directory)

            # self.df_raw = get_df_county(self, self.county, landing_directory)
            self.df_raw = get_df_state(self, landing_directory)

            self.logger.info("Saving raw data")
            self.df_raw.to_parquet(raw_path, index=False)
            copy_file_to_storage("etl", raw_path)

        for column in self.numeric_columns:
            self.df_raw[column] = pd.to_numeric(self.df_raw[column], errors="coerce")

        for column in self.datetime_columns:
            self.df_raw[column] = pd.to_datetime(self.df_raw[column])

        self.logger.info(f"Total parcels: {len(self.df_raw.Parid.unique())}")
        self.logger.info(f"Total sales: {len(self.df_raw)}")

        engine = sqlalchemy.create_engine(os.getenv("DB_STRING_LOCAL"))
        engine.connect()

        meta = sqlalchemy.MetaData(bind=engine)
        meta.reflect(bind=engine)

        prediction_table = meta.tables["prediction"]

        predictions_to_update = engine.execute(
            prediction_table.select().where(
                (prediction_table.c.actualSalePrice.is_(None)) & (prediction_table.c.actualSaleDate.is_(None))
            )
        ).all()

        total_predictions_to_update = len(predictions_to_update)
        total_predictions_updated = 0

        if total_predictions_to_update == 0:
            self.logger.info(f"No predictions to update")
        else:
            self.logger.info(f"Attempting to update {total_predictions_to_update} predictions")

            df_raw_update = self.df_raw.loc[
                self.df_raw.last_sale_date.notna()
                & self.df_raw.last_sale_price.notna()
                & self.df_raw.Parid.isin([x.parcel for x in predictions_to_update])
                & self.df_raw.County.isin(list(set([x.county for x in predictions_to_update])))
                & self.df_raw.State.isin(list(set([x.state for x in predictions_to_update])))
            ]

            for p in predictions_to_update:
                df_existing_prediction = (
                    df_raw_update.loc[
                        (df_raw_update.Parid == p.parcel)
                        & (df_raw_update.County == p.county)
                        & (df_raw_update.State == p.state)
                        & (df_raw_update.last_sale_date.dt.date >= p.predictionPeriodStart)
                    ]
                    .sort_values("last_sale_date")
                    .drop_duplicates(subset="Parid", keep="first", ignore_index=True)
                )

                if len(df_existing_prediction) > 0:
                    total_predictions_updated += 1
                    new_values = df_existing_prediction.to_dict(orient="records")[0]
                    engine.execute(
                        prediction_table.update()
                        .where(prediction_table.c.id == p.id)
                        .values(
                            actualSaleDate=new_values["last_sale_date"], actualSalePrice=new_values["last_sale_price"]
                        )
                    )

            self.logger.info(f"Updated {total_predictions_updated} predictions")

    def _get_df_encoded(self):
        encoded_directory = os.path.join("data", "etl", self.run_date, self.state, "2-encoded")
        make_directory(encoded_directory)

        encoded_path = os.path.join(encoded_directory, "raw-encoded.gz")

        if file_exists(encoded_path):
            self.logger.info("Loading existing encoded data")
            self.df_raw_encoded = pd.read_parquet(encoded_path)
        else:
            self.logger.info("Encoding raw data")
            df_raw_features = self.df_raw.drop(columns=self.non_encoded_columns).astype(str)

            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_encoder.fit(df_raw_features)

            self.df_raw_encoded = pd.DataFrame(
                self.ordinal_encoder.transform(df_raw_features),
                columns=df_raw_features.columns,
                index=df_raw_features.index,
            )

            for column in self.non_encoded_columns:
                self.df_raw_encoded[column] = self.df_raw[column]

            self.logger.info("Saving encoded data")
            self.df_raw_encoded.to_parquet(encoded_path, index=False)
            copy_file_to_storage("etl", encoded_path)

            ordinal_encoder_path = os.path.join(encoded_directory, "ordinal-encoder.pkl")
            with open(ordinal_encoder_path, "wb") as ordinal_encoder_file:
                pickle.dump(self.ordinal_encoder, ordinal_encoder_file)
                copy_file_to_storage("etl", ordinal_encoder_path)

    def _get_df_prepared(self):
        gc.collect()
        prepared_directory = os.path.join("data", "etl", self.run_date, self.state, "3-prepared")
        make_directory(prepared_directory)

        prepared_path = os.path.join(prepared_directory, "prepared.gz")

        if file_exists(prepared_path):
            self.logger.info("Loading existing prepared data")
            self.df_prepared = pd.read_parquet(prepared_path)
            self.df_prepared.date = pd.to_datetime(self.df_prepared.date)
        else:
            self.logger.info("Preparing data")

            parcel_ids = list(self.df_raw_encoded.Parid.unique())

            # if self.is_local:
            #     parcel_ids = random.sample(parcel_ids, int(len(parcel_ids) * 0.25))
            #     self.df_raw_encoded = self.df_raw_encoded.loc[self.df_raw_encoded.Parid.isin(parcel_ids)]

            self.df_prepared = pd.concat(
                Pool(self.cpu_count).map(
                    partial(
                        get_parcel_prepared_data,
                        df_raw_encoded=self.df_raw_encoded,
                        train_start_date=self.train_start_date.replace(
                            year=self.train_start_date.year - self.train_years
                        ),
                        min_months_since_last_sale=self.min_months_since_last_sale,
                        # max_months_since_last_sale=self.max_months_since_last_sale,
                    ),
                    np.array_split(parcel_ids, self.cpu_count),
                ),
                copy=False,
                ignore_index=True,
            ).drop_duplicates(ignore_index=True)

            for label_name in self.label_names:
                self.df_prepared[label_name] = pd.to_numeric(self.df_prepared[label_name])

            self.logger.info("Saving prepared data")
            self.df_prepared.to_parquet(prepared_path, index=False)
            copy_file_to_storage("etl", prepared_path)

        self.parcel_ids = list(self.df_prepared.Parid.unique())

        self.logger.info("Splitting data")
        identifier_columns = [
            "Parid",
            # "Book",
            # "Plat",
            # "Parcel",
            # "ParcelID",
            # "PropertyNumber",
            # "PriorOwner",
            "Owner Name 1",
            "Last Owner Name 1",
            # "Owner Name 2",
            "House #",
            "Street Name",
            "Street Suffix",
            # "Month of Sale",
            # "Day of Sale",
            # "Year of Sale",
        ]

        non_train_columns = [
            "date",
            "next_sale_date",
            "last_sale_date",
            "last_sale_price",
            "next_same_owner",
            "next_business_owner",
        ]

        prepared_drop_columns = non_train_columns + identifier_columns

        # df_prepared_filtered = self.df_prepared.copy()
        df_prepared_filtered = self.df_prepared.loc[
            (self.df_prepared.months_since_last_sale >= self.min_months_since_last_sale)
            & (self.df_prepared["Building Value"].notna())
            & (self.df_prepared["Land Value"].notna())
            # & (self.df_prepared.months_since_last_sale <= self.max_months_since_last_sale)
            # (
            #     (self.df_prepared.next_sale_price.isna())
            #     | (
            #         self.df_prepared.next_sale_price.between(
            #             self.df_prepared.next_sale_price.quantile(self.min_sale_price_quantile),
            #             self.df_prepared.next_sale_price.quantile(self.max_sale_price_quantile),
            #         )
            #     )
            # )
        ]

        self.df_test = df_prepared_filtered.loc[
            (df_prepared_filtered.date >= self.test_date) & (df_prepared_filtered.date <= self.test_date)
        ]

        # df_test_pos_labels = self.df_test.loc[self.df_test.sale_in_3_months == 1]
        # self.df_test = self.df_test if len(df_test_pos_labels) > 0 else df_test_pos_labels

        self.df_train = df_prepared_filtered.loc[
            (df_prepared_filtered.date >= self.train_start_date)
            & (df_prepared_filtered.date <= self.train_end_date)
            & (~df_prepared_filtered.Parid.isin(self.df_test.loc[self.df_test.sale_in_3_months == 1].Parid))
            & (
                (df_prepared_filtered.next_sale_price.isna())
                | (
                    (
                        self.df_prepared.next_sale_price
                        > (self.df_prepared["Building Value"] + self.df_prepared["Land Value"])
                    )
                )
                # | (
                #     df_prepared_filtered.next_sale_price.between(
                #         df_prepared_filtered.next_sale_price.quantile(self.min_sale_price_quantile),
                #         df_prepared_filtered.next_sale_price.quantile(self.max_sale_price_quantile),
                #     )
                # )
            )
            & ((df_prepared_filtered.next_same_owner.isna()) | (df_prepared_filtered.next_same_owner == 0))
            & ((df_prepared_filtered.next_business_owner.isna()) | (df_prepared_filtered.next_business_owner == 0))
        ]

        self.df_test.drop(columns=prepared_drop_columns, inplace=True)
        self.df_train.drop(columns=prepared_drop_columns, inplace=True)

        del df_prepared_filtered

    def _train_models(self):
        gc.collect()

        self.mlflow_client = MlflowClient()
        self.mlflow_experiment = mlflow.set_experiment(f"/Shared/{self.state}")

        max_jobs = self.cpu_count
        self.fit_minutes = 120
        self.per_job_fit_minutes = 20
        self.per_job_fit_memory_limit_gb = 8

        if self.is_local:
            # max_jobs = self.cpu_count - 3
            self.fit_minutes = 60
            self.per_job_fit_minutes = 15  # BEST RUN 15
            self.per_job_fit_memory_limit_gb = 3.5

        self.logger.info(f"Total fit minutes: {self.fit_minutes}")
        self.logger.info(f"Job fit minutes: {self.per_job_fit_minutes}")
        self.logger.info(f"Job memory limit (GB): {self.per_job_fit_memory_limit_gb}")

        procs = []

        def log_y_label_stats(label_name, label_values):
            total_labels = len(label_values)
            sum_labels = sum(label_values)
            p_labels = sum_labels / total_labels * 100
            self.logger.info(f"{label_name}: {sum_labels}/{total_labels} ({p_labels:.2f}%)")

        for label_name in self.label_names:
            is_classification = label_name in self.classification_label_names
            n_jobs = math.floor(max_jobs * 0.9) if is_classification else math.ceil(max_jobs * 0.1)
            # n_jobs = 7 if is_classification else 2

            self.logger.info(f"Running AutoML for {label_name} with {n_jobs} jobs")

            df_train = self.df_train.loc[self.df_train[label_name].notna()].drop_duplicates()
            df_test = self.df_test.loc[self.df_test[label_name].notna()].drop_duplicates()

            if is_classification:
                X_train = df_train.drop(columns=self.label_names).to_numpy()
                X_test = df_test.drop(columns=self.label_names).to_numpy()

                y_train = df_train[label_name].values
                y_test = df_test[label_name].values

                log_y_label_stats(f"{label_name} train labels", y_train)

                # X_train_res = X_train
                # y_train_res = y_train
                X_train_res, y_train_res = self.rus.fit_resample(X_train, y_train)  # BEST RUN enable
                log_y_label_stats(f"{label_name} train labels resampled", y_train_res)

                if len(y_test) > 0:
                    log_y_label_stats(f"{label_name} test labels", y_test)

                proc = Process(
                    target=train_model,
                    args=(
                        "classification",
                        self,
                        label_name,
                        n_jobs,
                        self.model_directories[label_name],
                        X_train_res,
                        y_train_res,
                        X_test,
                        y_test,
                    ),
                )
            else:
                df_train = df_train.loc[(df_train[label_name].notna()) & (df_train[label_name] > 0)]
                df_test = df_test.loc[(df_test[label_name].notna()) & (df_test[label_name] > 0)]

                self.logger.info(f"{label_name} train total: {len(df_train)}")
                self.logger.info(f"{label_name} test total: {len(df_test)}")

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
                        n_jobs,
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

            df_train = self.df_train.loc[self.df_train[label_name].notna()].copy().drop_duplicates()
            df_test = self.df_test.loc[self.df_test[label_name].notna()].copy().drop_duplicates()

            for index_of_df_temp, df_temp in enumerate([df_train, df_test]):
                if len(df_temp) > 0:
                    is_classification = label_name in self.classification_label_names
                    type_of_df_temp = "training" if index_of_df_temp == 0 else "testing"
                    self.logger.info(f"Metrics with {type_of_df_temp} data:")

                    if is_classification:
                        X_test = df_temp.drop(columns=self.label_names).to_numpy()
                        y_test = df_temp[label_name].values

                        y_pred_proba = [y[1] for y in model.predict_proba(X_test)]
                        y_pred = [1 if y > self.classification_proba_threshold else 0 for y in y_pred_proba]

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

                        for pred_proba in np.arange(0.1, 1.0, 0.1):
                            pred_proba = round(pred_proba, 1)
                            y_pred = [1 if y > pred_proba else 0 for y in y_pred_proba]
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

    def _get_df_predictions(self):
        self.logger.info("Creating current data frame")

        self.df_current = self.df_prepared.loc[
            (self.df_prepared.date >= self.test_date) & (self.df_prepared.date <= self.test_date)
        ].drop(columns=["date"])

        if "sale_in_3_months" in list(self.df_current.columns):
            df_current_pos_labels = self.df_current.loc[self.df_current.sale_in_3_months == 1]
            self.df_current = self.df_current if len(df_current_pos_labels) > 0 else df_current_pos_labels

        if len(self.df_current) == 0:
            self.df_current = (
                self.df_raw_encoded.loc[
                    (self.df_raw_encoded.last_sale_date < self.test_date.replace(month=self.test_date.month + 1))
                ]
                .sort_values(by=["Parid", "last_sale_date"], ascending=True)
                .drop_duplicates(subset="Parid", keep="last", ignore_index=True)
            )

            self.df_current["months_since_last_sale"] = Pool(self.cpu_count).map(
                partial(get_parcel_months_since_last_sale, current_date=self.test_date),
                self.df_current.last_sale_date,
            )
            self.df_current["Year Built"].fillna(0, inplace=True)
            self.df_current["months_since_year_built"] = Pool(self.cpu_count).map(
                partial(get_parcel_months_since_year_built, current_date=self.test_date),
                self.df_current["Year Built"],
            )

        self.df_current = self.df_current.loc[
            (self.df_current.months_since_last_sale >= self.min_months_since_last_sale)
            # & (self.df_current.months_since_last_sale <= self.max_months_since_last_sale)
            & (self.df_current["Building Value"].notna())
            & (self.df_current["Land Value"].notna())
        ]

        self.logger.info(f"Total properties in current data frame: {len(self.df_current)}")
        train_columns = list(self.df_train.drop(columns=self.label_names).columns)
        X_current = self.df_current[train_columns].to_numpy()

        for label_name in self.classification_label_names:
            self.logger.info(f"Loading predictions for {label_name}")
            model = mlflow.sklearn.load_model(self.model_directories[label_name])

            pred_label_name = f"pred_{label_name}"
            self.df_current[pred_label_name] = [y[1] for y in model.predict_proba(X_current)]

            for pred_proba in np.arange(0.1, 1.0, 0.1):
                pred_proba = round(pred_proba, 1)
                predicted_parcels = len(self.df_current.loc[self.df_current[pred_label_name] > pred_proba])
                self.logger.info(f"Predicted with proba {pred_proba}: {predicted_parcels}")

            self.df_current = self.df_current.loc[
                self.df_current[pred_label_name] > self.classification_proba_threshold
            ]

        X_current = self.df_current[train_columns].to_numpy()

        for label_name in self.regression_label_names:
            self.logger.info(f"Loading predictions for {label_name}")
            model = mlflow.sklearn.load_model(self.model_directories[label_name])

            pred_label_name = f"pred_{label_name}"
            self.df_current[pred_label_name] = model.predict(X_current).round(decimals=-4)

            # df_train = self.df_train.loc[self.df_train[label_name].notna()]
            # X_train = df_train.drop(columns=self.label_names).to_numpy()
            # y_train = df_train[label_name].values

            # y_pred_train = model.predict(X_train)
            # df_train["y_pred_train"] = y_pred_train
            # df_train = df_train.loc[
            #     df_train.y_pred_train.between(
            #         df_train.y_pred_train.quantile(self.min_sale_price_quantile),
            #         df_train.y_pred_train.quantile(self.max_sale_price_quantile),
            #     )
            # ]

            # y_train = df_train[label_name].values
            # y_pred_train = df_train.y_pred_train

            # mape_value = mean_absolute_percentage_error(y_train, y_pred_train)

            mape_value = 0.125
            self.logger.info(f"MAPE for {label_name}: {mape_value:.2f}")

            y_pred_current_interval = self.df_current[pred_label_name] * mape_value

            self.df_current[f"{pred_label_name}_lower"] = (
                self.df_current[pred_label_name] - y_pred_current_interval
            ).round(decimals=-4)
            self.df_current[f"{pred_label_name}_upper"] = (
                self.df_current[pred_label_name] + y_pred_current_interval
            ).round(decimals=-4)

        self.df_current["last_sale_price"] = pd.to_numeric(self.df_current.last_sale_price, errors="coerce")

        ordinal_encoder_path = os.path.join(
            "data", "etl", self.run_date, self.state, "2-encoded", "ordinal-encoder.pkl"
        )

        with open(ordinal_encoder_path, "rb") as ordinal_encoder_file:
            ordinal_encoder = pickle.load(ordinal_encoder_file)
            encoded_columns = list(self.df_raw_encoded.drop(columns=self.non_encoded_columns).columns)

        self.df_predictions = pd.DataFrame(
            ordinal_encoder.inverse_transform(self.df_current[encoded_columns]),
            columns=encoded_columns,
            index=self.df_current.index,
        ).join(self.df_current[list(set(list(self.df_current.columns)) - set(encoded_columns))])

        self.df_predictions = self.df_predictions.loc[
            self.df_predictions.pred_next_sale_price
            < (2 * (self.df_predictions["Building Value"] + self.df_predictions["Land Value"]))
        ]

        def get_latest_address(parcel_id):
            raw_row = (
                self.df_raw.loc[self.df_raw.Parid == parcel_id]
                .sort_values(by="last_sale_date", ascending=False)
                .iloc[0]
            )
            house_number = (
                ""
                if np.isnan(pd.to_numeric(raw_row["House #"], errors="coerce"))
                else str(int(pd.to_numeric(raw_row["House #"])))
            )
            street_name = raw_row["Street Name"].title().replace("Nan", "")
            street_suffix = raw_row["Street Suffix"].title().replace("Nan", "")
            return f"{house_number} {street_name} {street_suffix}".strip()

        self.df_predictions["Address"] = self.df_predictions.Parid.apply(get_latest_address)
        self.df_predictions["Latitude"] = None
        self.df_predictions["Longitude"] = None

        self.logger.info("Retrieving address coordinates")
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

        def get_address_coordinates(row):
            address = row["Address"]
            county = row["County"] + " County"
            geocode_results = gmaps.geocode(f"{address}, {county}, OH")

            for result in geocode_results:
                good_street_number = False
                good_county = False

                for component in result["address_components"]:
                    if component["types"][0] == "street_number" and component["long_name"] in address:
                        good_street_number = True
                    elif component["types"][0] == "administrative_area_level_2" and component["long_name"] == county:
                        good_county = True

                if good_street_number and good_county:
                    return result["geometry"]["location"]

            return None

        for index, row in self.df_predictions.iterrows():
            coordinates = get_address_coordinates(row)

            if coordinates:
                self.df_predictions.at[index, "Latitude"] = coordinates["lat"]
                self.df_predictions.at[index, "Longitude"] = coordinates["lng"]

        self.df_predictions = self.df_predictions.loc[
            self.df_predictions.Latitude.notna() & self.df_predictions.Longitude.notna()
        ]

        self.logger.info(f"Total predictions: {len(self.df_predictions)}")
        self.df_predictions["goodSale"] = None

        if "sale_in_3_months" in list(self.df_predictions.columns):
            acc_pred_sale_in_3_months = self.df_predictions.sale_in_3_months.sum() / len(self.df_predictions)
            self.df_predictions["pred_next_sale_price_error"] = np.abs(
                (self.df_predictions.next_sale_price - self.df_predictions.pred_next_sale_price)
                / self.df_predictions.next_sale_price
            )
            self.df_predictions["pred_next_sale_price_in_bound"] = (
                self.df_predictions.next_sale_price >= self.df_predictions.pred_next_sale_price_lower
            ) & (self.df_predictions.next_sale_price <= self.df_predictions.pred_next_sale_price_upper)

            acc_pred_next_sale_price = 1 - np.mean(self.df_predictions.pred_next_sale_price_error)
            acc_pred_next_sale_price_bound = len(
                self.df_predictions[self.df_predictions.pred_next_sale_price_in_bound == True]
            ) / len(self.df_predictions)

            self.logger.info(f"Final accuracy for sale_in_3_months: {acc_pred_sale_in_3_months:.3f}")
            self.logger.info(f"Final accuracy for next_sale_price: {acc_pred_next_sale_price:.3f}")
            self.logger.info(f"Final accuracy for next_sale_price_lower/upper: {acc_pred_next_sale_price_bound:.3f}")

            def get_good_sale(row):
                return (
                    row["next_sale_price"] > (row["Building Value"] + row["Land Value"])
                    and (pd.isna(row["next_same_owner"]) or row["next_same_owner"] == 0)
                    and (pd.isna(row["next_business_owner"]) or row["next_business_owner"] == 0)
                )

            self.df_predictions["goodSale"] = self.df_predictions.apply(get_good_sale, axis=1)
            df_predictions_ignore_sales = self.df_predictions.loc[self.df_predictions.goodSale == True]

            df_predictions_ignore_sales = self.df_predictions.loc[
                (
                    self.df_predictions.next_sale_price
                    > (self.df_predictions["Building Value"] + self.df_predictions["Land Value"])
                )
                & ((self.df_predictions.next_same_owner.isna()) | (self.df_predictions.next_same_owner == 0))
                & ((self.df_predictions.next_business_owner.isna()) | (self.df_predictions.next_business_owner == 0))
            ]

            self.logger.info(f"Total predictions with ignored sales: {len(df_predictions_ignore_sales)}")

            acc_pred_next_sale_price = 1 - np.mean(df_predictions_ignore_sales.pred_next_sale_price_error)
            acc_pred_next_sale_price_bound = len(
                df_predictions_ignore_sales[df_predictions_ignore_sales.pred_next_sale_price_in_bound == True]
            ) / len(df_predictions_ignore_sales)

            self.logger.info(f"Final accuracy for sale_in_3_months with ignored sales: {acc_pred_sale_in_3_months:.3f}")
            self.logger.info(f"Final accuracy for next_sale_price with ignored sales: {acc_pred_next_sale_price:.3f}")
            self.logger.info(
                f"Final accuracy for next_sale_price_lower/upper with ignored sales: {acc_pred_next_sale_price_bound:.3f}"
            )

        self.df_predictions = self.df_predictions.astype(str)

        for column in [
            "Year Built",
            "Stories",
            "Livable Sqft",
            "Building Value",
            "Land Value",
            "next_sale_price",
            "pred_next_sale_price",
        ]:
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) or pd.isnull(value) else pd.to_numeric(value, errors="coerce")
            )

        for column in list(self.df_predictions.columns):
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) or str(value).lower() in ["nan", "nat", "none"] else value
            )

        self.df_predictions["Prediction Period Start"] = self.prediction_period_start_date
        self.df_predictions["Prediction Period End"] = self.prediction_period_end_date

        for column in ["Prediction Period Start", "Prediction Period End", "next_sale_date"]:
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) else pd.to_datetime(value).date()
            )

        self.df_predictions["goodSale"] = self.df_predictions["goodSale"].map(
            {"None": None, "True": True, "False": False}
        )

        self.df_predictions["updatedAt"] = datetime.utcnow()
        self.df_predictions = self.df_predictions[
            [
                "Prediction Period Start",
                "Prediction Period End",
                "Parid",
                "Address",
                "County",
                "State",
                "Latitude",
                "Longitude",
                "Year Built",
                "Stories",
                "Livable Sqft",
                "Building Value",
                "Land Value",
                "last_sale_date",
                "pred_next_sale_price",
                "next_sale_price",
                "next_sale_date",
                "goodSale",
                "updatedAt",
            ]
        ].rename(
            columns={
                "Prediction Period Start": "predictionPeriodStart",
                "Prediction Period End": "predictionPeriodEnd",
                "Parid": "parcel",
                "Address": "address",
                "County": "county",
                "State": "state",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "Year Built": "yearBuilt",
                "Stories": "stories",
                "Livable Sqft": "livableSqft",
                "Building Value": "buildingValue",
                "Land Value": "landValue",
                "last_sale_date": "lastSaleDate",
                "pred_next_sale_price": "predictedSalePrice",
                "next_sale_price": "actualSalePrice",
                "next_sale_date": "actualSaleDate",
            },
        )

        self.df_predictions = self.df_predictions.replace({np.nan: None})

        predictions_directory = os.path.join("data", "predictions", self.run_date, self.state)
        make_directory(predictions_directory)
        self.df_predictions.to_parquet(os.path.join(predictions_directory, "predictions.gz"), index=False)
        self.df_predictions.to_json(
            os.path.join(predictions_directory, "predictions.json"), index=False, orient="split"
        )
        copy_directory_to_storage("predictions", predictions_directory)

        self.logger.info(f"Writing to {os.getenv('APP_ENV')} database")

        engine = sqlalchemy.create_engine(os.getenv("DB_STRING_LOCAL"))
        engine.connect()

        meta = sqlalchemy.MetaData(bind=engine)
        meta.reflect(bind=engine)

        prediction_table = meta.tables["prediction"]

        index_columns = ["predictionPeriodStart", "predictionPeriodEnd", "parcel"]
        update_columns = [x for x in list(self.df_predictions.columns) if x not in index_columns]

        existing_predictions = engine.execute(
            prediction_table.select().where(
                (prediction_table.c.predictionPeriodStart == self.prediction_period_start_date)
                & (prediction_table.c.predictionPeriodEnd == self.prediction_period_end_date)
                & (prediction_table.c.parcel.in_(self.df_predictions.parcel.unique()))
                & (prediction_table.c.county.in_(self.df_predictions.county.unique()))
                & (prediction_table.c.state.in_(self.df_predictions.state.unique()))
            )
        ).all()

        total_existing_predictions = len(existing_predictions)
        existing_parcels = []

        if total_existing_predictions == 0:
            self.logger.info(f"No predictions to update")
        else:
            self.logger.info(f"Updating {total_existing_predictions} predictions")

            for p in existing_predictions:
                existing_parcels.append(p.parcel)
                df_existing_prediction = self.df_predictions.loc[
                    (self.df_predictions.parcel == p.parcel)
                    & (self.df_predictions.county == p.county)
                    & (self.df_predictions.state == p.state)
                ]
                new_values = df_existing_prediction[update_columns].to_dict(orient="records")[0]
                engine.execute(prediction_table.update().where(prediction_table.c.id == p.id).values(new_values))

        df_new_predictions = self.df_predictions.loc[~self.df_predictions.parcel.isin(existing_parcels)]
        total_new_predictions = len(df_new_predictions)

        if total_new_predictions == 0:
            self.logger.info(f"No predictions to insert")
        else:
            self.logger.info(f"Inserting {total_new_predictions} predictions")
            engine.execute(prediction_table.insert().values(df_new_predictions.to_dict(orient="records")))

        engine.dispose()


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ETL_ML_Task()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
