from typing import List, Tuple

import pandas as pd
import numpy as np
import os
from math import comb

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


def optimal_max_workers(number_of_periods: int, 
                        cpu_factor: float = 1.0, 
                        io_factor: float = 1.0) -> int:
    """
    Calculate the optimal number of max_workers for ThreadPoolExecutor.

    Parameters:
    - number_of_periods: int
        The number of unique time periods for which you are calculating the bilateral index.
    - cpu_factor: float, optional (default=1.0)
        A multiplicative factor to adjust for CPU-bound tasks. Values greater than 1 will 
        increase the number of workers, while values less than 1 will decrease it.
    - io_factor: float, optional (default=1.0)
        A multiplicative factor to adjust for IO-bound tasks. Values greater than 1 will 
        increase the number of workers, while values less than 1 will decrease it.

    Returns:
    - int: The optimal number of max_workers
    """
    
    # Get the number of CPU cores available on the machine
    num_cpu_cores = os.cpu_count()
    
    # Calculate the number of unique combinations
    num_combinations = comb(number_of_periods, 2)
    
    # Initial guess based on CPU cores
    optimal_workers = int(num_cpu_cores * cpu_factor)
    
    # Adjust for IO-bound tasks
    optimal_workers = int(optimal_workers * io_factor)
    
    # Cap it to the number of unique combinations if it exceeds
    optimal_workers = min(optimal_workers, num_combinations)
    
    # Ensure it's at least 1
    optimal_workers = max(1, optimal_workers)
    
    return optimal_workers
