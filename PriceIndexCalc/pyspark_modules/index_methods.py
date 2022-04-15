"""
Multilateral methods using native PySpark functionality. 

Provides the following function:

* :func:`multilateral_methods_pyspark`
"""
from typing import Sequence, Optional

import pandas as pd
from pyspark.sql import (
    DataFrame as SparkDF,
)

from .helpers import _weights_calc_pyspark
from .multilateral import geary_khamis_pyspark, time_dummy_pyspark

__author__ = ['Dr. Usman Kayani']

def multilateral_methods_pyspark(
    df: SparkDF,
    method: str,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
    product_id_col: str = 'id',
    characteristics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Calculate multilateral index numbers in PySpark.

    Currently supported: Time Product Dummy (tpd) and Time Hedonic Dummy (tdh).

    Parameters
    ----------
    df : Spark DataFrame
        Contains price and quantity columns, a time series column, and a product
        ID column as a minimum. A characteristics column should also be present
        for hedonic methods.
    price_col : str, defaults to 'price'
        User-defined name for the price column.
    quantity_col : str, defaults to 'quantity'
        User-defined name for the quantity column.
    date_col : str, defaults to 'date'
        User-defined name for the date column.
    product_id_col: str, defaults to 'id'
        User-defined name for the product id column.
    characteristics: list of str, defaults to None
        The names of the characteristics columns.
    method: str
        Options: {tpd', 'tdh', 'gk'}
        The multilateral method to apply.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe of the index values.
    """
    method = method.lower()

    if method not in {'tpd', 'tdh', 'gk'}:
        raise ValueError(
            "Invalid method or not implemented yet."
        )
    
    # Get timeseries for output index.;;
    time_series = [i.month for i in df.select(date_col).distinct().collect()]
    
    # Calculate weights for each item in each period.
    df = df.withColumn(
        'weights',
        _weights_calc_pyspark(price_col, quantity_col, date_col)
    )

    if method == 'gk':
        index_vals = geary_khamis_pyspark(df, price_col, date_col, product_id_col)
    if method == 'tpd':
        index_vals = time_dummy_pyspark(
            df,
            len(time_series),
            price_col,
            date_col,
            product_id_col,
        )
    elif method == 'tdh':
        if not characteristics:
            raise ValueError(
                "Characteristics required for TDH."
            )
        else: 
            index_vals = time_dummy_pyspark(
                df,
                len(time_series),
                price_col,
                date_col,
                product_id_col,
                characteristics,
            )

    return (
        pd.DataFrame(
            index_vals, 
            index=time_series
        )
        .rename({0: 'index_value'}, axis=1)
    )

