"""
Multilateral methods using native PySpark functionality. 

Provides the following function:

* :func:`multilateral_methods_pyspark`
"""
from typing import Sequence, List, Optional

import pandas as pd
import numpy as np
from pyspark.sql import (
    DataFrame as SparkDF,
    functions as F,
    Column as SparkCol,
)
from pyspark.sql.window import Window

from wls import wls

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

    Currently supported: Time Product Dummy (TPD) and Time Hedonic Dummy (TDH).

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
        Options: {TPD', 'TDH'}
        The multilateral method to apply.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe of the index values.
    """
    if method not in {'TPD', 'TDH'}:
        raise ValueError(
            "Invalid method or not implemented yet."
        )
    
    # Get timeseries for output index.
    time_series = [i.month for i in df.select(date_col).distinct().collect()]
    
    # Calculate weights for each item in each period.
    df = df.withColumn(
        'weights',
        _weights_calc(price_col, quantity_col, date_col)
    )
      
    if method == 'TPD':
        index_vals =_get_time_dummy_index(
            df,
            len(time_series),
            price_col,
            date_col,
            product_id_col,
        )
    elif method == 'TDH':
        if not characteristics:
            raise ValueError(
                "Characteristics required for TDH."
            )
        else: 
            index_vals = _get_time_dummy_index(
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

def _get_time_dummy_index(
    df: SparkDF,
    Number_of_periods: int,
    price_col: str = 'price',
    date_col: str = 'month',
    product_id_col: str = 'id',
    characteristics: Optional[Sequence[str]] = None,
) -> List:
    """Obtain the time dummy indices for a given dataframe in PySpark.

    Calculates the time dummy indices using a formula with weighted least
    squares regression.  When passed with characteristics, this function returns
    the Time Dummy Hedonic indices. When passed without it returns the Time
    Product Dummy indices.
    """  
    # Calculate logarithm of the prices for each item.
    df = df.withColumn('log_price', F.log(price_col))

    non_time_vars = characteristics if characteristics else [product_id_col]

    # WLS regression with labels, features & weights -> fit model.
    model = (
        wls(
            df,
            dependent_var='log_price',
            independent_vars=[date_col, *non_time_vars],
            engine='pyspark'
        )
    )

    # Extracting time dummy coefficients.
    time_dummy_coeff = model.coefficients[:Number_of_periods-1][::-1]
    
    # Get indices from the time dummy coefficients & set first = 1.
    return [1, *np.exp(time_dummy_coeff)]


def _weights_calc(
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
)-> SparkCol:
    """Calculate weights from expenditure shares in PySpark."""
    window = Window.partitionBy(date_col)
    expenditure = F.col(price_col)*F.col(quantity_col)
    
    return expenditure / F.sum(expenditure).over(window)
  
