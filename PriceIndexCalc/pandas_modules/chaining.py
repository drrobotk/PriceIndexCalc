"""Module to perform annual chain linking with Pandas.

This module provides the functions:

* :func:`chain_linking_pandas_join`
* :func:`chain_linking_pandas_matrix`

Chain linking is the process of joining together two indices that
overlap in one period by rescaling one of them to make its value equal
to that of the other in the same period thus combining them into a
single time series. This is achieved by multiplying the index value by
the linking factor.
"""
from typing import List
import pandas as pd
import numpy as np

__author__ = ['Dr. Usman Kayani']

def chain_linking_pandas_join(
    dfs: List[pd.DataFrame],
    date_col: str = 'period',
    link_month: int = 1,
) -> pd.DataFrame:
    """
    Chain linking method using pandas shift and join in pandas.

    This method uses a cumulative product of unchained January indices
    to determine the linking factors and a shift function to calculate
    the chained indices.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the timeseries and unchained index values.
    date_col: str, default 'period'
        User-defined name for the date column.
    link_month: int, default 1
        Linking month for chaining.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and chained index values.
    """
    df, df_link_month = dfs
    # Determine chained indices from unchained multiplied by a link factor.
    df = (
        df
        .assign(year=df[date_col].dt.year)
        .merge(df_link_month, on='year')
    )
    df['link_factor'] = df.link_factor.shift(1, fill_value=1)
    df['index_value'] = df.index_value * df.link_factor

    return df.drop(columns=['link_factor', 'year'])


def chain_linking_pandas_matrix(
    dfs: List[pd.DataFrame],
    date_col: str = 'period',
    link_month: int = 1,
) -> pd.DataFrame:
    """
    Chain linking method using matrix operations with pandas and numpy.

    This method uses a cumulative product of unchained January indices
    to determine the linking factors, then converting these to an array
    with repeated values to calculate the chained indices via pointwise
    multiplication (Hadamard product).

    Parameters
    ----------
    df: Pandas DataFrame
        Dataframe containing the timeseries and unchained index values.
    date_col: str, default 'period'
        User-defined name for the date column.
    link_month: int, default 1
        Linking month for chaining.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and chained index values.
    """
    df, df_link_month = dfs
    # Convert link factors to an array for pointwise multiplication.
    if len(df) % 12 == 1:
        link_factor = df_link_month.link_factor.to_numpy()[:-1]
    else:
        link_factor = df_link_month.link_factor.to_numpy()

    # Repeat link factors for 12 months (Feb(year)-Jan(year+1)) and
    # insert 1 for the first.
    link_factor = np.insert(np.repeat(link_factor, 12), 0, 1)

    # Determine chained indices from unchained multiplied by a link factor.
    return df.assign(index_value=df.index_value * link_factor)

def linking_method(
    df: pd.DataFrame,
    date_col: str = 'period',
    link_month: int = 1,
    method: str = 'pandas_join',
) -> pd.DataFrame:
    """
    Chain linking method using pandas shift and join in pandas.

    This method uses a cumulative product of unchained January indices
    to determine the linking factors and a shift function to calculate
    the chained indices.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the timeseries and unchained index values.
    date_col: str, default 'period'
        User-defined name for the date column.
    link_month: int, default 1
        Linking month for chaining.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and chained index values.
    """
    # Conver date column to datetime for computation.
    df[date_col] = pd.to_datetime(df[date_col])

    # Determine linking factors from cumulative product of unchained January
    # indices.
    df_link_month = df.loc[df[date_col].dt.month == link_month]
    df_link_month = (
        df_link_month
        .assign(link_factor=(df_link_month.index_value / 100).cumprod())
        .drop(columns=[date_col, 'index_value'])
    )
    # Determine the linking method.
    if method == 'pandas_join':
        return chain_linking_pandas_join(
            dfs=[df, df_link_month],
            date_col=date_col,
            link_month=link_month,
        )
    elif method == 'pandas_matrix':
        return chain_linking_pandas_matrix(
            dfs=[df, df_link_month],
            date_col=date_col,
            link_month=link_month,
        )
    else:
        raise ValueError(
            'Invalid linking method. '
            'Valid methods are "pandas_join" and "pandas_matrix".'
        )