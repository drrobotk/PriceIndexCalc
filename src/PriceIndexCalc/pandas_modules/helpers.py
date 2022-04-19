from typing import List, Tuple

import pandas as pd
import numpy as np

__author__ = ['Dr. Usman Kayani']

def diag(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert df to diagonal matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert to diagonal matrix.

    Returns
    ------- 
    pd.DataFrame
        Diagonal matrix.
    """
    return pd.DataFrame(np.diag(df), index=df.index)

def _weights_calc(
    df: pd.DataFrame, 
    price_col: str='price',
    quantity_col: str='quantity',
    date_col: str='month',
    product_id_col: str='id',
) -> List:
    """
    Calculate weights from expenditure shares.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to calculate weights from.
    price_col : str, optional
        Column name of price. The default is 'price'.
    quantity_col : str, optional
        Column name of quantity. The default is 'quantity'.
    date_col : str, optional
        Column name of date. The default is 'month'.
    product_id_col : str, optional
        Column name of product id. The default is 'id'.

    Returns
    ------- 
    List
        List of weights.
    """
    # Pivot dataframe for weights calculation.
    df_pivot = df.pivot(index=product_id_col, columns=date_col).fillna(0)

    # Set up expenditure and weights df from expenditure shares.
    expenditure = df_pivot[price_col] * df_pivot[quantity_col]
    weights = expenditure.divide(expenditure.sum())
    
    # Melt df for outpout.
    weights = pd.melt(
        weights.reset_index(),
        id_vars=[product_id_col],
        value_vars=df[date_col].unique(),
        var_name=date_col,
        value_name='weights',
    )

    # Merge back with input df.
    return pd.merge(
            right=weights, 
            left=df, 
            on=[date_col, product_id_col]
    )

def _vars_split(df) -> Tuple[List[str]]:
    """
    Split vars into categorical and numerical.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to split.

    Returns
    -------
    Tuple[List[str]]
        Tuple of categorical and numerical vars.
    """
    numerical_vars = (
        df
        .select_dtypes(include='number')
        .columns
        .tolist()
    )
    categorical_vars = list(set(df.columns) - set(numerical_vars))   
    return categorical_vars, numerical_vars
