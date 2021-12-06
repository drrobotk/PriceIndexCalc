from typing import List

import pandas as pd
import numpy as np

__author__ = ['Dr. Usman Kayani']

def diag(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Convert df to diagonal matrix."""
    return pd.DataFrame(np.diag(df), index=df.index)

def _weights_calc(
    df: pd.DataFrame, 
    price_col: str='price',
    quantity_col: str='quantity',
    date_col: str='month',
    product_id_col: str='id',
) -> List:
    """Calculate weights from expenditure shares."""
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

def _vars_split(df):
    """Split vars into categorical and numerical."""
    numerical_vars = (
        df
        .select_dtypes(include='number')
        .columns
        .tolist()
    )
    categorical_vars = list(set(df.columns) - set(numerical_vars))   
    return categorical_vars, numerical_vars
