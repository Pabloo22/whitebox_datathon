from whitebox_datathon.preprocessing import (
    filter_column,
    clip_column,
    drop_clip_column,
    drop_columns,
    one_hot_encode,
    one_hot_encode_top_n,
    frequency_encode,
)
from whitebox_datathon.preprocessing2 import add_antiguedad

from whitebox_datathon.paths import DATA_RAW
from whitebox_datathon.mappings import FUEL_MAPPING, SHIFT_MAPPING, YEAR_MAPPING
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def load_data():
    """
    Load data from the raw directory.

    Returns
    -------
    pd.DataFrame
        Raw data.
    """
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")

    return train, test


def preprocess_data(data):
    df = one_hot_encode(df, "website")

    # Column 'make': Lowercase, remove spaces, and remove accents
    df["make"] = (
        df["make"]
        .str.lower()
        .str.replace(" ", "")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    # Column 'model': No specific preprocessing required
    # Column 'version': No specific preprocessing required

    # Column 'fuel': Mapping fuel types
    df = filter_column(df, "fuel", FUEL_MAPPING)

    # Column 'year': Mapping years
    df = filter_column(df, "year", YEAR_MAPPING)

    # Columns 'kms' and 'power': Apply power scaler
    scaler = PowerTransformer()
    df["kms"] = scaler.fit_transform(df["kms"].values.reshape(-1, 1))

    scaler = PowerTransformer()
    df["power"] = scaler.fit_transform(df["power"].values.reshape(-1, 1))

    # Column 'doors': Mapping and setting rare values to zero
    rare_doors = df["doors"].value_counts()[df["doors"].value_counts() < 10].index
    df["doors"] = df["doors"].apply(lambda x: 0 if x in rare_doors else x)

    # Column 'shift': Binary mapping for shift types
    df = filter_column(df, "shift", SHIFT_MAPPING)
    print(df["shift"])
    df["shift"] = df["shift"].apply(
        lambda x: 1 if str(x).lower() == "automático" else 0
    )

    # Column 'color': No specific preprocessing required

    # Column 'photos': Clip at 100
    df = clip_column(df, "photos", lower=0, upper=100)

    # Column 'price': Drop rows with prices greater than 500k or less than 200
    df = drop_clip_column(df, "price", lower=200, upper=500000)

    # Column 'location': Frequency encoding
    df = frequency_encode(df, "location")

    # Column 'publish_date': Extract year and compute antiguedad
    df = add_antiguedad(df, "publish_date", "year")

    # Column 'dealer_id': Frequency encoding
    df = frequency_encode(df, "dealer_id")

    df = drop_columns(df, ["make", "model", "version"])

    return df
