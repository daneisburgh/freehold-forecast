import gc

import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool

from freeholdforecast.common.static import get_df_attom_recorder_additional_data


def get_df_attom(task, landing_directory):
    good_columns = [
        "[ATTOM ID]",
        "SitusCounty",
        "PropertyJurisdictionName",
        "SitusStateCountyFIPS",
        "MinorCivilDivisionName",
        "MinorCivilDivisionCode",
        "NeighborhoodCode",
        "CensusTract",
        "CensusBlockGroup",
        "CensusBlock",
        "APNFormatted",
        "ParcelNumberRaw",
        "ParcelNumberYearAdded",
        "PropertyAddressFull",
        "PropertyAddressHouseNumber",
        "PropertyAddressStreetName",
        "PropertyAddressStreetSuffix",
        "PropertyAddressCity",
        "PropertyAddressZIP",
        "PropertyAddressZIP4",
        "PropertyAddressCRRT",
        "PropertyLatitude",
        "PropertyLongitude",
        "LegalTractNumber",
        "PartyOwner1NameFull",
        "PartyOwner1NameLast",
        "StatusOwnerOccupiedFlag",
        "OwnerTypeDescription1",
        "ContactOwnerMailingCounty",
        "ContactOwnerMailingFIPS",
        "ContactOwnerMailAddressFull",
        "ContactOwnerMailAddressHouseNumber",
        "ContactOwnerMailAddressStreetName",
        "ContactOwnerMailAddressStreetSuffix",
        "ContactOwnerMailAddressCity",
        "ContactOwnerMailAddressState",
        "ContactOwnerMailAddressZIP",
        "ContactOwnerMailAddressZIP4",
        "ContactOwnerMailAddressCRRT",
        "DeedOwner1NameFull",
        "DeedOwner1NameLast",
        "TaxYearAssessed",
        "TaxAssessedValueTotal",
        "TaxAssessedValueImprovements",
        "TaxAssessedValueLand",
        "PreviousAssessedValue",
        "TaxMarketValueYear",
        "TaxMarketValueTotal",
        "TaxMarketValueImprovements",
        "TaxMarketValueLand",
        "TaxExemptionHomeownerFlag",
        "TaxFiscalYear",
        "TaxRateArea",
        "YearBuilt",
        "PropertyUseMuni",
        "PropertyUseGroup",
        "PropertyUseStandardized",
        "AreaBuilding",
        "AreaLotSF",
        "ParkingGarage",
        "StoriesCount",
        "Exterior1Code",
        "ViewDescription",
        "InstrumentDate",
        "RecordingDate",
        "Mortgage1RecordingDate",
        "Mortgage2RecordingDate",
        "TransferAmount",
        "Mortgage1Amount",
        "Mortgage2Amount",
        "TransferInfoPurchaseTypeCode",
        "TransferInfoDistressCircumstanceCode",
        "ForeclosureAuctionSale",
    ]

    primary_column = "ParcelNumberRaw"

    assessor_file = landing_directory + "/DANEISBURGH_TAXASSESSOR_0001.txt"
    recorder_file = landing_directory + "/DANEISBURGH_RECORDER_0001.txt"

    df_assessor = pd.read_csv(assessor_file, sep="\t", low_memory=False)
    df_recorder = pd.read_csv(recorder_file, sep="\t", low_memory=False)

    df_assessor = df_assessor.loc[
        (df_assessor[primary_column].notna())
        & (df_assessor.PropertyAddressFull.notna())
        & (df_assessor.PropertyLatitude.notna())
        & (df_assessor.PropertyLongitude.notna())
        & (df_assessor.GeoQuality.isin([0, 1]))
        & (df_assessor.TaxAssessedValueImprovements > 0)
        & (df_assessor.OwnerTypeDescription1.str.lower() == "individual")
    ]

    # df_recorder = df_recorder.loc[df_recorder.TransferInfoPurchaseTypeCode == 40]

    df_recorder[primary_column] = df_recorder["APNFormatted"]
    df_recorder.InstrumentDate.fillna(df_recorder.RecordingDate, inplace=True)
    df_recorder["InstrumentDate"] = pd.to_datetime(df_recorder.InstrumentDate)
    df_recorder = df_recorder.loc[
        (df_recorder.InstrumentDate.notna()) & (df_recorder.TransferAmount.notna()) & (df_recorder.TransferAmount > 0)
    ]
    df_recorder["InstrumentDateYear"] = df_recorder.InstrumentDate.dt.year
    df_recorder.drop_duplicates(
        subset=[primary_column, "InstrumentDateYear"], keep="last", ignore_index=True, inplace=True
    )
    df_recorder.drop(columns=["InstrumentDateYear"], inplace=True)

    assessor_columns = [x for x in df_assessor.columns if x in good_columns]
    recorder_columns = [x for x in df_recorder.columns if x in good_columns]
    df_assessor = df_assessor.drop_duplicates(ignore_index=True)[assessor_columns]
    df_recorder = df_recorder.drop_duplicates(ignore_index=True)[recorder_columns]

    gc.collect()

    df_recorder_unique_columns = df_recorder.columns.difference(df_assessor.columns).tolist()

    df_recorder_size = df_recorder.groupby(primary_column, as_index=False).size()
    df_assessor_not_in_recorder = df_assessor.loc[
        (df_assessor.YearBuilt.notna())
        & (
            (~df_assessor[primary_column].isin(df_recorder[primary_column].unique()))
            | (df_assessor[primary_column].isin(df_recorder_size.loc[df_recorder_size["size"] == 1][primary_column]))
        )
    ]

    df_recorder_additional = pd.DataFrame(
        Pool(task.cpu_count).map(
            partial(
                get_df_attom_recorder_additional_data,
                assessor_columns=df_assessor_not_in_recorder.columns,
                recorder_columns=df_recorder_unique_columns,
            ),
            df_assessor_not_in_recorder.to_dict("records"),
        )
    )

    df_recorder_all = pd.concat([df_recorder, df_recorder_additional], copy=False, ignore_index=True)

    df_recorder_columns = [primary_column] + df_recorder_unique_columns
    df = pd.merge(df_assessor, df_recorder_all[df_recorder_columns], on=primary_column)
    # df = pd.merge(df_assessor, df_recorder[df_recorder_columns], on=primary_column)
    df["InstrumentDate"] = pd.to_datetime(df.InstrumentDate)
    df["TransferAmount"] = pd.to_numeric(df.TransferAmount)
    df = df.loc[
        (df.InstrumentDate.notna())
        & (df.TransferAmount.notna())
        # & (df.InstrumentDate.dt.year > 1960)
        # & (df.TransferAmount != 0)
    ].drop_duplicates(subset=[primary_column, "InstrumentDate"], ignore_index=True)

    del df_assessor
    del df_recorder
    del df_recorder_additional
    del df_assessor_not_in_recorder
    del df_recorder_all
    gc.collect()

    df["Parid"] = df["ParcelNumberRaw"]
    df["last_sale_date"] = pd.to_datetime(df.InstrumentDate)
    df["last_sale_price"] = pd.to_numeric(df.TransferAmount)
    df.sort_values(by="last_sale_date", ascending=True, inplace=True)

    return df
