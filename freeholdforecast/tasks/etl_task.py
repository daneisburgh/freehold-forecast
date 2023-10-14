import os
import pandas as pd
import pickle
import sqlalchemy

from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder

from freeholdforecast.common.county_dfs import get_df_state
from freeholdforecast.common.task import Task
from freeholdforecast.common.utils import copy_file_to_storage, date_string, file_exists, make_directory


class ETL_Task(Task):
    """
    Task class to execute ETL processes for loading and preparing data.
    """

    def __init__(self, state="ohio", run_date=None):
        """Initialize ETL task class and variables

        Args:
            state (str): State to load data
            run_date (str, optional): Run date required for historical performance testing
        """

        super().__init__()
        self.state = state
        self.run_date = run_date if run_date is not None else date_string(datetime.now())
        self.logger.info(f"Initializing task for {self.state} with run date {self.run_date}")

    def launch(self):
        """Execute task processes"""

        self.logger.info(f"Launching task")
        self._get_df_raw()
        self._get_df_encoded()
        self.logger.info(f"Finished task")

    def _get_df_raw(self):
        """Load and save raw data"""

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
        """Encode raw data for training"""

        encoded_directory = os.path.join("data", "etl", self.run_date, self.state, "2-encoded")
        make_directory(encoded_directory)

        encoded_path = os.path.join(encoded_directory, "raw-encoded.gz")

        if file_exists(encoded_path):
            self.logger.info("Loading existing encoded data")
            self.df_raw_encoded = pd.read_parquet(encoded_path)
        else:
            self.logger.info("Encoding raw data")
            df_raw_features = self.df_raw.astype(str)

            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_encoder.fit(df_raw_features)

            self.df_raw_encoded = pd.DataFrame(
                self.ordinal_encoder.transform(df_raw_features),
                columns=df_raw_features.columns,
                index=df_raw_features.index,
            )

            self.logger.info("Saving encoded data")
            self.df_raw_encoded.to_parquet(encoded_path, index=False)
            copy_file_to_storage("etl", encoded_path)

            ordinal_encoder_path = os.path.join(encoded_directory, "ordinal-encoder.pkl")
            with open(ordinal_encoder_path, "wb") as ordinal_encoder_file:
                pickle.dump(self.ordinal_encoder, ordinal_encoder_file)
                copy_file_to_storage("etl", ordinal_encoder_path)

        self.logger.info(f"Total encoded parcels: {self.df_raw_encoded.Parid.nunique()}")
        self.logger.info(f"Total encoded sales: {len(self.df_raw_encoded)}")


if __name__ == "__main__":
    task = ETL_Task()
    task.launch()
