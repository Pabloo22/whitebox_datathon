import pandas as pd


def add_antiguedad(df):
    """
    Adds a new column 'published_year' extracted from 'publish_date' and
    calculates 'antiguedad' as the difference between 'published_year' and 'year'.

    Args:
        df (pd.DataFrame): Input data frame with 'publish_date' and 'year' columns.

    Returns:
        pd.DataFrame: Data frame with new 'published_year' and 'antiguedad' columns.
    """
    # Extract the year from 'publish_date' and add as a new column 'published_year'
    df["published_year"] = pd.to_datetime(df["publish_date"]).dt.year

    # Calculate 'antiguedad' as the difference between 'published_year' and 'year'
    df["antiguedad"] = df["published_year"] - df["year"]
