"""Module to perform annual chain linking with PySpark.

This module provides the functions:

* :func:`chain_linking_window`
* :func:`chain_linking_join`

Chain linking is the process of joining together two indices that
overlap in one period by rescaling one of them to make its value equal
to that of the other in the same period thus combining them into a
single time series. This is achieved by multiplying the index value by
the linking factor.
"""
from pyspark.sql import (
    DataFrame as SparkDF,
    functions as F,
    Window,
)

from .helpers import _cumprod_over_period

__author__ = ['Dr. Usman Kayani']

def chain_linking_window(
    df: SparkDF,
    date_col: str = 'period',
) -> SparkDF:
    """Chain linking method using only window functions.

    This method uses a cumulative product of unchained January indices
    to determine the linking factors, from a window partitioned by year,
    and a lag function over a period window to calculate the chained
    indices using pointwise multiplication of columns (Hadarmard
    product).

    Parameters
    ----------
    df: SparkDF
        Dataframe containing the timeseries and unchained index values.
    date_col: str, default 'period'
        User-defined name for the date column.

    Returns
    -------
    SparkDF
        Dataframe containing the timeseries and chained index values.
    """
    # Determine linking factors from cumulative product of unchained
    # January indices, which we determine by a window function
    # partitioned by year and by selecting the first value.
    window_year = Window.partitionBy(F.year(date_col)).orderBy(date_col)
    link_index = F.first('index_value').over(window_year)
    linking_factor = _cumprod_over_period(link_index/100)

    # Lag linking factors by one month to adjust for the correct
    # periods.
    lagged_linking_factor = (
        F.lag(linking_factor, 1, 1).over(Window.orderBy(date_col))
    )

    # Determine chained indices from unchained indices multiplied by a
    # link factor.
    chained_indices = F.col('index_value') * lagged_linking_factor

    return df.withColumn('index_value', chained_indices)


def chain_linking_join(
    df: SparkDF,
    date_col: str = 'period',
    link_month: int = 1,
) -> SparkDF:
    """
    Chain linking method using lag over window and join in PySpark.

    This method uses a cumulative product of unchained January indices
    to determine the linking factors and a lag function over a period
    window to calculate the chained indices using pointwise
    multiplication of columns (Hadarmard product).

    Parameters
    ----------
    df: SparkDF
        Dataframe containing the timeseries and unchained index values.
    date_col: str, default 'period'
        User-defined name for the date column.
    link_month: int, default 1
        Linking month for chaining.

    Returns
    -------
    SparkDF
        Dataframe containing the timeseries and chained index values.
    """
    # Determine linking factors from cumulative product of unchained January
    # indices.
    df_link_month = (
            df
            .filter(F.month(date_col) == link_month)
            .withColumn('year', F.year(date_col))
            .withColumn(
                'link_factor',
                _cumprod_over_period(F.col('index_value') / 100, date_col)
            )
            .drop(date_col, 'index_value')
    )

    # Determine chained indices from unchained multiplied by a link factor.
    return (
        df
        .withColumn('year', F.year(date_col))
        .join(df_link_month, on='year')
        .withColumn(
            'link_factor',
            F.lag('link_factor', 1, 1).over(Window.orderBy(date_col))
        )
        .withColumn(
            'index_value',
            F.col('index_value') * F.col('link_factor')
        )
        .drop('link_factor', 'year')
    )


