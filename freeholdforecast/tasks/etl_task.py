import gc
import numpy as np
import os
import pandas as pd
import pickle
import sqlalchemy

from calendar import monthrange
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import OrdinalEncoder

from freeholdforecast.common.county_dfs import get_df_state
from freeholdforecast.common.task import Task
from freeholdforecast.common.utils import (
    copy_file_to_storage,
    date_string,
    file_exists,
    make_directory,
    to_numeric,
)


class ETL_Task(Task):
    def __init__(self, state: str, run_date=None):
        super().__init__()
        self.state = state
        self.run_date = run_date if run_date is not None else date_string(datetime.now())
        self.logger.info(f"Initializing task for {self.state} with run date {self.run_date}")

        self.test_date = (datetime.strptime(self.run_date, "%Y-%m-%d")).replace(day=1)
        self.prediction_period_start_date = (self.test_date + relativedelta(months=+1)).replace(day=1)

        self.prediction_period_end_date = self.prediction_period_start_date + relativedelta(months=+2)
        first_weekday, month_days = monthrange(
            self.prediction_period_end_date.year, self.prediction_period_end_date.month
        )
        self.prediction_period_end_date = self.prediction_period_end_date.replace(day=month_days)

        self.train_end_date = self.test_date - relativedelta(months=+6)
        first_weekday, month_days = monthrange(self.train_end_date.year, self.train_end_date.month)
        self.train_end_date = self.train_end_date.replace(day=month_days)

        self.train_start_date = self.train_end_date - relativedelta(months=+12)
        self.train_start_date = self.train_start_date.replace(day=1)

        self.logger.info(f"Train dates: {date_string(self.train_start_date)} to {date_string(self.train_end_date)}")
        self.logger.info(f"Test date: {date_string(self.test_date)}")
        self.logger.info(
            f"Prediction period: {date_string(self.prediction_period_start_date)} to {date_string(self.prediction_period_end_date)}"
        )

        self.min_months_since_last_sale = 10
        self.max_months_since_last_sale = 20
        self.logger.info(f"Min months since last sale: {self.min_months_since_last_sale}")
        self.logger.info(f"Max months since last sale: {self.max_months_since_last_sale}")

        self.classification_label_names = ["sale_in_3_months"]
        self.regression_label_names = ["next_sale_price"]
        self.label_names = self.classification_label_names + self.regression_label_names
        self.drop_label_names = self.label_names + ["sale_in_6_months", "sale_in_12_months"]

        self.classification_proba_threshold = 0.7
        self.logger.info(f"Classification probability threshold: {self.classification_proba_threshold}")

        self.numeric_columns = [
            "YearBuilt",
            "PropertyAddressZIP",
            "PropertyLatitude",
            "PropertyLongitude",
            "BathCount",
            "BathPartialCount",
            "BedroomsCount",
            "AreaBuilding",
            "AreaLotSF",
            "StoriesCount",
            "TaxMarketValueLand",
            "TaxMarketValueImprovements",
            "TaxMarketValueTotal",
            "EstimatedValue",
            "ConfidenceScore",
            "YearBuiltRounded",
            "AreaBuildingRounded",
            "AreaLotSFRounded",
            "TaxMarketValueLandRounded",
            "TaxMarketValueImprovementsRounded",
            "TaxMarketValueTotalRounded",
            "EstimatedValueRounded",
            "last_sale_priceRounded",
            "business_owner",
            "next_business_owner",
            "same_owner",
            "next_same_owner",
            "good_sale",
            "next_good_sale",
            "last_sale_price",
            "next_sale_price",
        ]

        self.datetime_columns = [
            "ValuationDate",
            "last_sale_date",
            "next_sale_date",
        ]

        self.non_encoded_columns = self.numeric_columns + self.datetime_columns
        self.model_directories = {}

        for label_name in self.label_names:
            self.model_directories[label_name] = os.path.join("data", "models", self.run_date, self.state, label_name)

    def launch(self):
        self.logger.info(f"Launching task")
        self._get_df_raw()
        self._get_df_encoded()
        self.logger.info(f"Finished task")

    def _get_df_raw(self):
        raw_directory = os.path.join("data", "etl", self.run_date, self.state, "1-raw")
        landing_directory = os.path.join("data", "etl", self.run_date, self.state, "0-landing")
        make_directory(raw_directory)

        raw_path = os.path.join(raw_directory, "raw.gz")

        if file_exists(raw_path):
            self.logger.info("Loading existing raw data")
            self.df_raw = pd.read_parquet(raw_path)
        else:
            self.logger.info("Retrieving landing data")
            make_directory(landing_directory)

            self.df_raw = get_df_state(self, landing_directory)

            self.logger.info("Saving raw data")
            self.df_raw.to_parquet(raw_path, index=False)
            copy_file_to_storage("etl", raw_path)

        for column in self.numeric_columns:
            self.df_raw[column] = to_numeric(self.df_raw[column])

        for column in self.datetime_columns:
            self.df_raw[column] = pd.to_datetime(self.df_raw[column])

        self.logger.info(f"Total raw parcels: {self.df_raw.Parid.nunique()}")
        self.logger.info(f"Total raw sales: {len(self.df_raw)}")

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
                & self.df_raw.SitusCounty.isin(list(set([x.county for x in predictions_to_update])))
                & self.df_raw.SitusStateCode.isin(list(set([x.state for x in predictions_to_update])))
            ]

            for p in predictions_to_update:
                df_existing_prediction = (
                    df_raw_update.loc[
                        (df_raw_update.Parid == p.parcel)
                        & (df_raw_update.SitusCounty == p.county)
                        & (df_raw_update.SitusStateCode == p.state)
                        & (df_raw_update.last_sale_date.dt.date >= p.predictionPeriodStart)
                    ]
                    .sort_values("last_sale_date", ignore_index=True)
                    .drop_duplicates(subset="Parid", keep="first", ignore_index=True)
                )

                if len(df_existing_prediction) > 0:
                    total_predictions_updated += 1
                    new_values = df_existing_prediction.to_dict(orient="records")[0]
                    engine.execute(
                        prediction_table.update()
                        .where(prediction_table.c.id == p.id)
                        .values(
                            updatedAt=datetime.utcnow(),
                            actualSaleDate=new_values["last_sale_date"],
                            actualSalePrice=new_values["last_sale_price"],
                            goodSale=(None if pd.isna(new_values["good_sale"]) else new_values["good_sale"]),
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

        self.logger.info(f"Total encoded parcels: {self.df_raw_encoded.Parid.nunique()}")
        self.logger.info(f"Total encoded sales: {len(self.df_raw_encoded)}")
