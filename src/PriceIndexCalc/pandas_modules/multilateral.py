"""
Provides the following multilateral methods:

* :func:`time_dummy`
* :func:`geary_khamis`
* :func:`geks`
paired with
    * :func:`carli`
    * :func:`jevons`
    * :func:`dutot`
    * :func:`laspeyres`
    * :func:`paasche`
    * :func:`geom_laspeyres`
    * :func:`geom_paasche`
    * :func:`drobish`
    * :func:`marshall_edgeworth`
    * :func:`palgrave`
    * :func:`fisher`
    * :func:`tornqvist`
    * :func:`walsh`
    * :func:`sato_vartia`
    * :func:`geary_khamis_b`
    * :func:`tpd`
    * :func:`rothwell`

The TDH/TPD methods are model-based multilateral index number methods
which have been proposed to incorporate scanner data. They are part of
many multilateral methods motivated by an attempt to minimize the risk
of chain drift, particularly within a window, while maximizing the
number of matches in the data.

TDH index is used when information on item characteristics are
available, and the TPD index when this information is lacking. The
TDH produces an explicit hedonic price index, while the TPD produces
an implicit hedonic price index, which are both estimated on the
pooled data of one or more periods via an application of expenditure
shares weighted least squares regression.
"""

from typing import List, Sequence, Dict, Tuple
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean

from .helpers import diag, _weights_calc
from .weighted_least_squares import wls
from .bilateral import *

__author__ = ['Dr. Usman Kayani']

def compute_bilateral(
        df_by_period: Dict,
        periods: Sequence,
        i: int,
        j: int,
        bilateral_method: str,
        price_col: str = 'price',
        quantity_col : str = 'quantity',
        product_id_col: str = 'id',
) -> Tuple:
    """
    Compute bilateral indices for a given pair of periods.

    Parameters
    ----------
    df_by_period : Dict
        Dictionary of dataframes by period.
    periods : Sequence
        Sequence of periods.
    i : int
        Index of first period.
    j : int
        Index of second period.
    bilateral_method : str
        Name of the bilateral method.
    price_col : str, optional
        Name of the column containing the price information.
    quantity_col : str, optional
        Name of the column containing the quantity information.
    product_id_col : str, optional
        Name of the column containing the product id information.
    """
    df_base = df_by_period[periods[i]]
    df_curr = df_by_period[periods[j]]

    common_products = (
        set(df_base[product_id_col])
        .intersection(set(df_curr[product_id_col]))
    )

    df_base = df_base[df_base[product_id_col].isin(common_products)]
    df_curr = df_curr[df_curr[product_id_col].isin(common_products)]

    if bilateral_method == 'tpd':
        df_matched = pd.concat([df_base, df_curr]).drop_duplicates()
        df_matched = _weights_calc(df_matched)
        return i, j, time_dummy(df_matched)[-1]
    else:
        bilateral_func = globals()[bilateral_method]
        p_base = df_base[price_col].values
        p_curr = df_curr[price_col].values
        data = (p_base, p_curr)

        if bilateral_method in {
            'laspeyres', 
            'paasche', 
            'geom_laspeyres', 
            'geom_paasche', 
            'drobish', 
            'marshall_edgeworth', 
            'tornqvist', 
            'fisher', 
            'walsh', 
            'sato_vartia', 
            'geary_khamis_b', 
            'rothwell', 
            'lowe'
        }:
            q_base = df_base[quantity_col].values
            data += (q_base,)

        if bilateral_method in {
            'paasche', 
            'geom_paasche', 
            'drobish', 
            'marshall_edgeworth', 
            'tornqvist', 
            'fisher', 
            'walsh', 
            'sato_vartia', 
            'geary_khamis_b'
        }:
            q_curr = df_curr[quantity_col].values
            data += (q_curr,)

        return i, j, bilateral_func(*data)

def geks(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str='month',
    product_id_col: str='id',
    bilateral_method: str = 'tornqvist',
) -> List:
    """
    Obtain the GEKS indices paired with a bilateral method for a given dataframe.

    Calculate the index values using a for loop to determine the matrix of
    bilaterals, where we exploit the symmetry condition a_{i j} = 1/a_{j i} and
    a_{i i} = 1 to save computation time, followed by a geometric mean.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    price_col : str, optional
        Name of the column containing the price information.
    quantity_col : str, optional
        Name of the column containing the quantity information.
    date_col : str, optional
        Name of the column containing the date information.
    product_id_col : str, optional
        Name of the column containing the product id information.
    bilateral_method : str, optional

    Returns
    -------
    List
        List of the GEKS indices.
    """
    # Get unique periods and length of time series.
    # Pre-filter data for each unique period
    periods = df[date_col].unique()
    no_of_periods = len(periods)
    df_by_period = {
        period: df[df[date_col] == period] 
        for period in periods
    }

    pindices = np.zeros((no_of_periods, no_of_periods))

    with ThreadPoolExecutor(max_workers=no_of_periods) as executor:
        futures = [
            executor.submit(
                compute_bilateral, 
                df_by_period, 
                periods, 
                i, 
                j, 
                bilateral_method, 
                price_col, 
                quantity_col, 
                product_id_col
            )
            for i, j in combinations(range(no_of_periods), 2)
        ]

    for future in as_completed(futures):
        i, j, result = future.result()
        pindices[i, j] = result

    pindices_sym = np.copy(pindices.T)
    mask = pindices_sym != 0
    pindices_sym[mask] = 1/pindices_sym[mask]
    pindices += pindices_sym + np.identity(no_of_periods)

    pgeo = gmean(pindices)

    return pd.Series(
        pgeo/pgeo[0],
        index=periods
    )


def time_dummy(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
    product_id_col: str = 'id',
    characteristics: List = None,
    engine: str = 'numpy'
) -> List:
    """Obtain the time dummy indices for a given dataframe.

    Calculates the time dummy indices using a formula with weighted least
    squares regression.  When passed with characteristics, this function returns
    the Time Dummy Hedonic indices. When passed without it returns the Time
    Product Dummy indices.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    price_col : str, optional
        Name of the column containing the price information.
    quantity_col : str, optional
        Name of the column containing the quantity information.
    date_col : str, optional
        Name of the column containing the date information.
    product_id_col : str, optional
        Name of the column containing the product id information.
    characteristics: List, optional
        List of the characteristics to use for the calculation.
    engine : str, optional
        Name of the engine to use for the calculation.

    Returns
    -------
    List
        List of the time dummy indices.
    """
    # Set the dtype for ID columns, in case it is numerical.
    df[product_id_col] = df[product_id_col].astype(str)

    # Calculate logarithm of the prices for each item for dependent variable.
    df['log_price'] = np.log(df[price_col])

    # Get time series for output index.
    time_series = df[date_col].unique()

    # Get terms for wls regression where characteristics are used if available.
    non_time_vars = [product_id_col]

    model_params = wls(
        df,
        dependent_var='log_price',
        independent_vars=[date_col, *non_time_vars],
        engine=engine
    )

    # Get indices from the time dummy coefficients & set first = 1.
    is_time_dummy = model_params.index.str.contains(date_col)

    return pd.Series(
        [1, *np.exp(model_params.loc[is_time_dummy])],
        index=time_series,
    )


def geary_khamis(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
    product_id_col: str = 'id',
    method_type: str = 'matrix',
) -> List:
    r"""Obtain the Geary-Khamis indices for a given dataframe.

    Calculates the Geary-Khamis indices using matrix operations.
    
    Parameters
    ----------
    price_col : str, defaults to 'price'
        User-defined price column name.
    quantity_col : str, defaults to 'quantity'
        User-defined quantity column name.
    product_id_col : str, defaults to 'product_id'
        The column name containing product ID values or product names.
    method_type: str, defaults to 'matrix'
        Options: {'matrix', 'iterative'}

        The method type to use for the GK computation.
    Returns
    -------
    List
        The sorted list of indices for each group.

    Notes
    -----
    For Geary-Khamis with the matrix method, we can determine the
    quality adjustment factors by solving the system of equations:

    .. math::

            \vec{b}=\left[I_{N}-C+R\right]^{-1} \vec{c}

    where :math:`\vec{c} = [1,0,\ldots, 0]^T` is an :math:`N \times 1`
    vector and :math:`R` is an :math:`N \times N` matrix given by,

    .. math::

            R=\left[\begin{array}{cccc}
                1 & 1 & \ldots & 1 \\
                0 & \ldots & \ldots & 0 \\
                \vdots & & & \vdots \\
                0 & \ldots & \ldots & 0
            \end{array}\right]

    and :math:`C` is the :math:`N \times N` matrix defined by,

    .. math::

            C=\hat{q}^{-1} \sum_{t=1}^{T} s^{t} q^{t \mathbf{T}}

    where :math:`\hat{q}^{-1}` is the inverse of an :math:`N \times N`
    diagonal matrix :math:`\hat{q}`, where the diagonal elements are the
    total quantities purchased for each good over all time periods,
    :math:`s^{t}` is a vector of the expenditure shares for time period
    :math:`t`, and :math:`q^{t \mathbf{T}}` is the transpose of the
    vector of quantities purchased in time period :math:`t`.

    Once the :math:`\vec{b}` vector has been calculated, the price
    levels can be computed from the equation:

    .. math::

            P_{t} =\frac{p^{t} \cdot q^{t}}{ \vec{b} \cdot q^{t}}

    The price index values can be determined by normalizing the price
    levels by the first period as,

    .. math::

            I_{t} = \frac{P_{t}}{P_{0}}

    References
    ----------
    Diewart, W. E, and Kevin, F. (2017). Substitution Bias in
    Multilateral Methods for CPI Construction Using Scanner Data.
    Discussion Paper 1702. Department of Economics, University of
    British Columbia.
    """
    if method_type not in ('matrix', 'iterative'):
        raise ValueError('The method type must be `matrix` or `iterative`')

    # We need to deal with missing values and reshape the df for the
    # required vectors and matrices.
    df = df.pivot(index=date_col, columns=product_id_col)

    # We need to deal with missing values and reshape the df for the
    # required vectors and matrices.
    df = _matrix_method_reshape(df)

    # Get number of unique products for the size of the vectors and
    # matrices.
    N = len(df.index.unique(level=product_id_col))

    # Matrices for the prices, quantities and weights.
    prices = df.loc[price_col]
    quantities = df.loc[quantity_col]
    weights = df.loc['weights']

    # Use iterative method directly if specified.
    if method_type == 'iterative':
        return _geary_khamis_iterative(prices, quantities)

    # Inverse of diagonal matrix with total quantities for each good over all
    # time periods as diagonal elements and matrix product of weights and
    # transpose of quantities to produce a square matrix, both required for C
    # matrix.
    q_matrix_inverse = np.diag(1/quantities.T.sum())
    prod_weights_qt_matrix = weights @ quantities.T

    # Product of above matrices to give the C square matrix, with the fixed
    # identity and R matrix, and c vector all required determine the quality
    # adjustment factors b.
    C_matrix = q_matrix_inverse @ prod_weights_qt_matrix
    R_matrix = np.zeros(shape=(N, N))
    R_matrix[:1] = 1
    
    # Define combo matrix used for isolating the singular matrices and
    # calculating the index values from the combo matrix `I_n - C + R`.
    combo_matrix = (np.identity(N) - C_matrix + R_matrix).fillna(0)
    
    if abs(np.linalg.det(combo_matrix)) <= 1e-7:
        # Fallback to iterative method for singular matrices.
        return _geary_khamis_iterative(prices, quantities)
    else:
        # Primary matrix method for non-singular matrices.
        return _geary_khamis_matrix(prices, quantities, combo_matrix)


def _matrix_method_reshape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape df for matrix method and deal with missing values.
    We first drop columns which contain all missing values, transpose
    the dataframe and then fill the remaining missing values with zero,
    to deal with missing items in some periods.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to reshape.

    Returns
    -------
    pd.DataFrame
        The reshaped dataframe.
    """
    return df.dropna(how='all', axis=1).T.fillna(0)


def _geary_khamis_iterative(
    prices: pd.DataFrame,
    quantities: pd.DataFrame,
    no_of_iterations: int = 100,
    precision: float = 1e-8,
) -> pd.Series:
    """
    Geary-Khamis iterative method.
    
    Parameters
    ----------
    prices : pd.DataFrame
        The price dataframe.
    quantities : pd.DataFrame
        The quantity dataframe.
    no_of_iterations : int, defaults to 100
        The number of iterations to perform.
    precision : float, defaults to 1e-8
        The precision to use for the iterative method.

    Returns
    -------
    pd.Series
        The price index values.
    """
    # Initialise index vals as 1's to find the solution with iteration.
    price_levels = pd.Series(1.0, index=prices.columns)
    quantity_share = quantities.T / quantities.sum(axis=1)

    # Iterate until we reach the set level of precision, or after a set
    # number of iterations if they do not converge.
    for _ in range(no_of_iterations):
        # Obtain matrices for iterative calculation.
        deflated_prices = prices / price_levels
        factors = diag(deflated_prices @ quantity_share)

        # Calculate new price levels from previous value.
        new_price_levels = (
            diag(prices.T @ quantities)
            .div(quantities.T @ factors)
            .squeeze()
        )

        pl_abs_diff = abs(price_levels - new_price_levels)
        
        if (pl_abs_diff <= precision).all():
            # Break loop when we reach given precision for final price levels.
            break
        else:
            # Otherwise set price level for next iteration.
            price_levels = new_price_levels

    # Normalize by first period for final output.
    return price_levels / price_levels.iloc[0]


def _geary_khamis_matrix(
    prices: pd.DataFrame,
    quantities: pd.DataFrame,
    combo_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Geary-Khamis matrix method.

    Parameters
    ----------
    prices : pd.DataFrame
        The price dataframe.
    quantities : pd.DataFrame
        The quantity dataframe.
    combo_matrix : pd.DataFrame
        The combo matrix.

    Returns
    -------
    pd.Series
        The price index values.
    """
    # Calculation of the vector b (factors) required to produce the
    # price levels. Corresponds to `b = [I_n - C + R]^-1 [1,0,..,0]^T`.
    # We use the Moore-Penrose inverse for the matrix inverse.
    factors = np.linalg.pinv(combo_matrix) @ np.eye(len(prices.index), 1)

    # Determine price levels to compute the final index values.
    price_levels = diag(prices.T @ quantities).div(quantities.T @ factors)

    # Normalize price levels to first period for final index values.
    index_vals = price_levels / price_levels.iloc[0]

    # Output as Pandas series for dynamic window.
    return index_vals.iloc[:, 0]
