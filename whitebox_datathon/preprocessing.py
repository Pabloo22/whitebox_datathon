import pandas as pd


def filter_column(data, column, map_dict, na_values=None):
    """
    Filter data by a dictionary of filters.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter.
    map_dict : dict
        Dictionary of maps to apply to the data. Keys are column names and values are lists of values to keep.
    na_values : str, optional
        Values to consider as missing values, by default None.

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    data[column] = data[column].map(map_dict)
    if na_values:
        data[column] = data[column].fillna(na_values)

    return data


def clip_column(data, column, lower, upper):
    """
    Clip data by a dictionary of filters.

    Parameters
    ----------
    data : pd.DataFrame
        Data to clip.
    lower : int
        Lower bound to clip the data.
    upper : int
        Upper bound to clip the data.

    Returns
    -------
    pd.DataFrame
        Clipped data.
    """
    data[column] = data[column].clip(lower=lower, upper=upper)

    return data


def drop_clip_column(data, column, lower, upper):
    """
    Drop rows outside of a range.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter.
    column : str
        Column to filter.
    lower : int
        Lower bound to clip the data.
    upper : int
        Upper bound to clip the data.

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    data = data[(data[column] >= lower) & (data[column] <= upper)]

    return data


def drop_columns(data, columns):
    """
    Drop columns from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter.
    columns : list
        Columns to drop.

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    data = data.drop(columns=columns)

    return data


def one_hot_encode(data, column):
    """
    One hot encode a column.

    Parameters
    ----------
    data : pd.DataFrame
        Data to encode.
    column : str
        Column to encode.

    Returns
    -------
    pd.DataFrame
        Encoded data.
    """
    data = pd.get_dummies(data, columns=[column])

    return data


def frequency_encode(data, column):
    """
    Frequency encode a column.

    Parameters
    ----------
    data : pd.DataFrame
        Data to encode.
    column : str
        Column to encode.

    Returns
    -------
    pd.DataFrame
        Encoded data.
    """
    freq = data[column].value_counts(normalize=True)
    data[column] = data[column].map(freq)

    return data


def add_antiguedad(data):
    """
    Get the antiquity of a column.

    Parameters
    ----------
    data : pd.DataFrame
        Data to encode.
    column : str
        Column to encode.

    Returns
    -------
    pd.DataFrame
        Encoded data.
    """
    # Get the publsh year from publish_date
    publish_year = data["publish_date"].dt.year

    data["antiguedad"] = publish_year - data["year"]

    return data
