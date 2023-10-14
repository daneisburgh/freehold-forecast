import gc
import math
import mlflow
import numpy as np
import os
import pandas as pd
import pickle
import sqlalchemy

from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import partial
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

from freeholdforecast.common.task import Task
from freeholdforecast.common.static import (
    get_parcel_months_since_last_sale,
    get_parcel_months_since_last_sale_max,
    get_parcel_months_since_year_built,
    get_parcel_prepared_data,
    train_model,
)
from freeholdforecast.common.utils import (
    copy_directory_to_storage,
    copy_file_to_storage,
    file_exists,
    make_directory,
    remove_directory,
    to_numeric,
)


class ML_Task(Task):
    def __init__(self, state: str, run_date=None):
        super().__init__()

    def launch(self):
        self.logger.info(f"Launching task")
        self._get_df_prepared()
        self._train_models()
        self._get_df_predictions()
        self.logger.info(f"Finished task")

    def _get_df_prepared(self):
        gc.collect()
        prepared_directory = os.path.join("data", "etl", self.run_date, self.state, "3-prepared")
        make_directory(prepared_directory)

        prepared_path = os.path.join(prepared_directory, "prepared.gz")
        parcel_ids = list(self.df_raw_encoded.Parid.unique())

        if file_exists(prepared_path):
            self.logger.info("Loading existing prepared data")
            self.df_prepared = pd.read_parquet(prepared_path)
            self.df_prepared.date = pd.to_datetime(self.df_prepared.date)
        else:
            self.logger.info("Preparing data")

            self.df_prepared = pd.concat(
                Pool(self.cpu_count).map(
                    partial(
                        get_parcel_prepared_data,
                        df_raw_encoded=self.df_raw_encoded.loc[self.df_raw_encoded.Parid.isin(parcel_ids)].copy(),
                        train_start_date=self.train_start_date,
                        min_months_since_last_sale=self.min_months_since_last_sale,
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

        self.df_prepared["months_since_year_built"] = Pool(self.cpu_count).map(
            partial(get_parcel_months_since_year_built, current_date=self.test_date),
            self.df_prepared.YearBuilt,
        )

        self.logger.info("Splitting data")

        self.classification_train_columns = [
            "total_sales",
            "CompanyFlag",
            "OwnerTypeDescription1",
            "PropertyUseGroup",
            "BathPartialCount",
            "StoriesCount",
            "ConfidenceScore",
            "TaxMarketValueTotalRounded",
            "business_owner",
        ]

        self.regression_train_columns = [
            "PropertyUseGroup",
            "YearBuiltRounded",
            "AreaBuildingRounded",
            "TaxMarketValueLandRounded",
            "TaxMarketValueImprovementsRounded",
            "TaxMarketValueTotalRounded",
            "EstimatedValueRounded",
            "last_sale_priceRounded",
            "ConfidenceScore",
        ]

        self.df_test = self.df_prepared.loc[
            (self.df_prepared.date >= self.test_date)
            & (self.df_prepared.date <= self.test_date)
            & (self.df_prepared.months_since_last_sale >= self.min_months_since_last_sale)
            & (self.df_prepared.months_since_last_sale <= self.max_months_since_last_sale)
            & ((self.df_prepared.next_good_sale.isna()) | (self.df_prepared.next_good_sale == 1))
        ]

        df_train = self.df_prepared.loc[
            (self.df_prepared.date >= self.train_start_date)
            & (self.df_prepared.date <= self.train_end_date)
            & (self.df_prepared.months_since_last_sale >= self.min_months_since_last_sale)
            & (self.df_prepared.months_since_last_sale <= self.max_months_since_last_sale)
            & ((self.df_prepared.next_good_sale.isna()) | (self.df_prepared.next_good_sale == 1))
        ].sort_values(by="date", ignore_index=True)

        df_test_pos_parcels = self.df_test.loc[self.df_test[self.classification_label_names[0]] == 1].Parid.unique()

        df_train_in_test_pos = df_train.loc[df_train.Parid.isin(df_test_pos_parcels)]
        df_train_not_in_test_pos = df_train.loc[~df_train.Parid.isin(df_test_pos_parcels)]
        df_train_in_test_pos[self.classification_label_names[0]] = 0
        df_train_in_test_pos["next_sale_price"] = np.nan

        self.df_train_classification = pd.concat(
            [df_train_in_test_pos, df_train_not_in_test_pos], ignore_index=True
        ).sort_values(by="date", ignore_index=True)

        self.df_train_regression = self.df_train_classification.loc[
            (self.df_train_classification.next_sale_price.notna())
            & (
                (self.df_train_classification.next_good_sale.isna())
                | (self.df_train_classification.next_good_sale == 1)
            )
        ]

        self.df_test = self.df_test[
            list(set(self.drop_label_names + self.classification_train_columns + self.regression_train_columns))
        ].drop_duplicates(ignore_index=True)
        self.df_train_classification = self.df_train_classification[
            self.drop_label_names + self.classification_train_columns
        ].drop_duplicates(ignore_index=True)
        self.df_train_regression = self.df_train_regression[
            self.drop_label_names + self.regression_train_columns
        ].drop_duplicates(ignore_index=True)

    def _train_models(self):
        gc.collect()
        remove_directory("mlruns")

        self.mlflow_client = MlflowClient()
        self.mlflow_experiment = mlflow.set_experiment(f"/Shared/{self.state}")

        max_jobs = self.cpu_count
        self.fit_minutes = 120
        self.per_job_fit_minutes = 20
        self.per_job_fit_memory_limit_gb = 8

        if self.is_local:
            self.fit_minutes = 30
            self.per_job_fit_minutes = 5
            self.per_job_fit_memory_limit_gb = 3

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
            n_jobs = math.floor(max_jobs * 0.7) if is_classification else math.ceil(max_jobs * 0.3)

            self.logger.info(f"Running AutoML for {label_name} with {n_jobs} jobs")
            df_test = self.df_test.loc[self.df_test[label_name].notna()].copy().drop_duplicates()

            if is_classification:
                df_train = self.df_train_classification.loc[
                    self.df_train_classification[label_name].notna()
                ].drop_duplicates()
                X_train = df_train.drop(columns=self.drop_label_names).to_numpy()
                X_test = df_test.drop(columns=self.drop_label_names)[self.classification_train_columns].to_numpy()

                y_train = df_train[label_name].values
                y_test = df_test[label_name].values

                log_y_label_stats(f"{label_name} train labels", y_train)

                X_train_res = X_train
                y_train_res = y_train

                if sum(y_test) > 0:
                    log_y_label_stats(f"{label_name} test labels", y_test)

                remove_directory(self.model_directories[label_name])

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
                df_train = self.df_train_regression.loc[self.df_train_regression[label_name].notna()].drop_duplicates()
                df_test = df_test.loc[df_test[self.classification_label_names[0]] == 1].drop_duplicates()

                self.logger.info(f"{label_name} train total: {len(df_train)}")

                if len(df_test) > 0:
                    self.logger.info(f"{label_name} test total: {len(df_test)}")

                X_train = df_train.drop(columns=self.drop_label_names).to_numpy()
                X_test = df_test.drop(columns=self.drop_label_names)[self.regression_train_columns].to_numpy()

                y_train = df_train[label_name].values
                y_test = df_test[label_name].values

                remove_directory(self.model_directories[label_name])

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
            is_classification = label_name in self.classification_label_names

            df_train_classification = (
                self.df_train_classification.loc[self.df_train_classification[label_name].notna()]
                .copy()
                .drop_duplicates()
            )
            df_train_regression = (
                self.df_train_regression.loc[self.df_train_classification[label_name].notna()].copy().drop_duplicates()
            )

            df_train = df_train_classification if is_classification else df_train_regression
            df_test = self.df_test.loc[self.df_test[label_name].notna()].copy().drop_duplicates()

            model = mlflow.sklearn.load_model(self.model_directories[label_name])
            self.logger.info(f"Model {label_name} " + model.sprint_statistics())

            for index_of_df_temp, df_temp in enumerate([df_train, df_test]):
                if len(df_temp) > 0 and df_temp[self.classification_label_names[0]].sum() > 0:
                    type_of_df_temp = "training" if index_of_df_temp == 0 else "testing"
                    self.logger.info(f"Metrics with {type_of_df_temp} data:")

                    if is_classification:
                        X_test = df_temp.drop(columns=self.drop_label_names)[
                            self.classification_train_columns
                        ].to_numpy()
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
                        df_temp = df_temp.loc[df_temp[self.classification_label_names[0]] == 1].drop_duplicates()
                        X_test = df_temp.drop(columns=self.drop_label_names)[self.regression_train_columns].to_numpy()
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
        date_filter = (datetime.strptime(self.run_date, "%Y-%m-%d") + relativedelta(months=+1)).replace(day=1)
        df_raw_encoded_filtered = self.df_raw_encoded.loc[(self.df_raw_encoded.last_sale_date < date_filter)]

        self.df_current = df_raw_encoded_filtered.sort_values(
            by=["Parid", "last_sale_date"], ascending=True
        ).drop_duplicates(subset="Parid", keep="last", ignore_index=True)
        self.df_current["month"] = self.test_date.month
        self.df_current["months_since_last_sale"] = Pool(self.cpu_count).map(
            partial(get_parcel_months_since_last_sale, current_date=self.test_date),
            self.df_current.last_sale_date,
        )
        self.df_current["months_since_last_sale_max"] = Pool(self.cpu_count).map(
            partial(get_parcel_months_since_last_sale_max, current_date=self.test_date),
            self.df_current.last_sale_date,
        )
        self.df_current["months_since_year_built"] = Pool(self.cpu_count).map(
            partial(get_parcel_months_since_year_built, current_date=self.test_date),
            self.df_current.YearBuilt,
        )

        df_raw_encoded_total_sales = (
            df_raw_encoded_filtered.groupby("Parid", as_index=False).size().rename(columns={"size": "total_sales"})
        )
        self.df_current = pd.merge(self.df_current, df_raw_encoded_total_sales, on="Parid")

        self.df_current = self.df_current.loc[
            (self.df_current.months_since_last_sale >= self.min_months_since_last_sale)
        ]

        self.logger.info(f"Total properties in current data frame: {len(self.df_current)}")

        for label_name in self.classification_label_names:
            self.logger.info(f"Loading predictions for {label_name}")
            model = mlflow.sklearn.load_model(self.model_directories[label_name])

            pred_label_name = f"pred_{label_name}"
            self.df_current[pred_label_name] = [
                y[1] for y in model.predict_proba(self.df_current[self.classification_train_columns].to_numpy())
            ]

            for pred_proba in np.arange(0.1, 1.0, 0.1):
                pred_proba = round(pred_proba, 1)
                predicted_parcels = len(self.df_current.loc[self.df_current[pred_label_name] > pred_proba])
                self.logger.info(f"Predicted with proba {pred_proba}: {predicted_parcels}")

            self.df_current = self.df_current.loc[
                self.df_current[pred_label_name] > self.classification_proba_threshold
            ]

        for label_name in self.regression_label_names:
            self.logger.info(f"Loading predictions for {label_name}")
            model = mlflow.sklearn.load_model(self.model_directories[label_name])

            pred_label_name = f"pred_{label_name}"
            self.df_current[pred_label_name] = model.predict(
                self.df_current[self.regression_train_columns].to_numpy()
            ).round(decimals=-4)

            mape_value = 0.125
            self.logger.info(f"MAPE for {label_name}: {mape_value:.3f}")

            y_pred_current_interval = self.df_current[pred_label_name] * mape_value

            self.df_current[f"{pred_label_name}_lower"] = (
                self.df_current[pred_label_name] - y_pred_current_interval
            ).round(decimals=-4)
            self.df_current[f"{pred_label_name}_upper"] = (
                self.df_current[pred_label_name] + y_pred_current_interval
            ).round(decimals=-4)

        self.df_current["last_sale_price"] = to_numeric(self.df_current.last_sale_price)

        ordinal_encoder_path = os.path.join(
            "data", "etl", self.run_date, self.state, "2-encoded", "ordinal-encoder.pkl"
        )

        with open(ordinal_encoder_path, "rb") as ordinal_encoder_file:
            ordinal_encoder = pickle.load(ordinal_encoder_file)
            encoded_columns = list(self.df_raw_encoded.drop(columns=self.non_encoded_columns).columns)

        self.df_predictions = (
            pd.DataFrame(
                ordinal_encoder.inverse_transform(self.df_current[encoded_columns]),
                columns=encoded_columns,
                index=self.df_current.index,
            )
            .join(self.df_current[list(set(list(self.df_current.columns)) - set(encoded_columns))])
            .astype(str)
        )

        for column in self.numeric_columns + ["pred_next_sale_price"]:
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) or pd.isnull(value) else to_numeric(value)
            )

        for column in list(self.df_predictions.columns):
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) or str(value).lower() in ["nan", "nat", "none"] else value
            )

        self.df_predictions["prediction_period_start"] = self.prediction_period_start_date
        self.df_predictions["prediction_period_end"] = self.prediction_period_end_date
        self.df_predictions["updated_at"] = datetime.utcnow()

        for column in ["prediction_period_start", "prediction_period_end"] + self.datetime_columns:
            self.df_predictions[column] = self.df_predictions[column].apply(
                lambda value: None if pd.isna(value) else pd.to_datetime(value).date()
            )

        self.df_predictions["months_since_last_sale"] = pd.to_numeric(self.df_predictions.months_since_last_sale)
        self.df_predictions["pred_sale_in_3_months"] = pd.to_numeric(self.df_predictions.pred_sale_in_3_months)
        self.df_predictions["total_sales"] = pd.to_numeric(self.df_predictions.total_sales)
        self.logger.info(f"Total predictions: {len(self.df_predictions)}")

        self.df_predictions = self.df_predictions[
            [
                "Parid",
                "YearBuilt",
                "PropertyAddressFull",
                "PropertyAddressCity",
                "PropertyAddressZIP",
                "SitusCounty",
                "SitusStateCode",
                "PropertyLatitude",
                "PropertyLongitude",
                "PropertyUseGroup",
                "business_owner",
                "AreaBuilding",
                "AreaLotSF",
                "StoriesCount",
                "TaxMarketValueLand",
                "TaxMarketValueImprovements",
                "TaxMarketValueTotal",
                "EstimatedValue",
                "ConfidenceScore",
                "last_sale_date",
                "last_sale_price",
                "next_sale_date",
                "next_sale_price",
                "next_good_sale",
                "pred_next_sale_price",
                "prediction_period_start",
                "prediction_period_end",
                "updated_at",
            ]
        ].rename(
            columns={
                "Parid": "parcel",
                "YearBuilt": "yearBuilt",
                "PropertyAddressFull": "address",
                "PropertyAddressCity": "city",
                "PropertyAddressZIP": "zipCode",
                "SitusCounty": "county",
                "SitusStateCode": "state",
                "PropertyLatitude": "latitude",
                "PropertyLongitude": "longitude",
                "PropertyUseGroup": "propertyUse",
                "business_owner": "businessOwner",
                "AreaBuilding": "areaBuildingSqft",
                "AreaLotSF": "areaLotSqft",
                "StoriesCount": "stories",
                "TaxMarketValueLand": "taxMarketValueLand",
                "TaxMarketValueImprovements": "taxMarketValueImprovements",
                "TaxMarketValueTotal": "taxMarketValueTotal",
                "EstimatedValue": "estimatedValue",
                "ConfidenceScore": "estimatedValueConfidenceScore",
                "last_sale_date": "lastSaleDate",
                "last_sale_price": "lastSalePrice",
                "next_sale_date": "actualSaleDate",
                "next_sale_price": "actualSalePrice",
                "next_good_sale": "goodSale",
                "pred_next_sale_price": "predictedSalePrice",
                "prediction_period_start": "predictionPeriodStart",
                "prediction_period_end": "predictionPeriodEnd",
                "updated_at": "updatedAt",
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
