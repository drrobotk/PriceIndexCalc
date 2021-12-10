"""
Provides the following functions:

* :func:`bilateral_methods`
* :func:`multilateral_methods`

With the following bilateral price index methods:

* Carli, Jevons, Dutot, Laspeyres, Paasche, geom_Laspeyres, geom_Paasche,
  Drobish, Marshall Edgeworth, Palgrave, Fisher, Tornqvist, Walsh, Sato Vartia,
  Geary Khamis, Rothwell.

and the following multilateral price index methods:

* The GEKS method paired with a bilateral method, the Geary-Khamis method (GK)
  and Time Dummy methods (TPD, TDH).
"""
from typing import Sequence, Optional

import pandas as pd
import seaborn as sns

from .helpers import _weights_calc
from .bilateral import *
from .multilateral import *
from .extension_methods import *

__author__ = ['Dr. Usman Kayani']

def bilateral_methods(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str='month',
    product_id_col: str='id',
    groups: Optional[Sequence[str]] = None,
    method: str = 'tornqvist',
    base_month: int = 1,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Calculate all the bilateral indices.

    Parameters
    ----------
    df: pandas DataFrame
        Contains price and quantity columns, a time series column, and a product
        ID column as a minimum. A characteristics column should also be present
        for hedonic methods.
    price_col: str, defaults to 'price'
        User-defined name for the price column.
    quantity_col: str, defaults to 'quantity'
        User-defined name for the quantity column.
    date_col: str, defaults to 'month'
        User-defined name for the date column.
    product_id_col: str, defaults to 'id'
        User-defined name for the product ID column.
    groups: list of str, defaults to None
        The names of the groups columns.
    method: str, defaults to 'tornqvist'
        Options: {'carli', 'jevons', 'dutot', 'laspeyres', 'paasche',
        'geom_laspeyres', 'geom_paasche', 'drobish', 'marshall_edgeworth',
        'palgrave', 'fisher', 'tornqvist', 'walsh', 'sato_vartia',
        'geary_khamis_b', 'tpd', 'rothwell'}

        The bilateral method to use.
    base_month: int, defaults to 1
        Integer specifying the base month.
    plot: bool, defaults to False
        Boolean parameter on whether to plot the resulting timeseries for price
        indices.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and index values.
    """
    method = method.lower()

    valid_bilateral_methods = {
        'carli', 'jevons', 'dutot', 'laspeyres',
        'paasche', 'geom_laspeyres', 'geom_paasche', 'drobish',
        'marshall_edgeworth', 'palgrave', 'fisher', 'tornqvist',
        'walsh', 'sato_vartia', 'geary_khamis_b', 'tpd', 'rothwell'
    }

    if method not in valid_bilateral_methods:
        raise ValueError("Invalid option, please select a valid bilateral method.")

    args = (price_col, quantity_col, date_col, product_id_col)

    if groups:
        return (
            df
            .groupby(groups)
            .apply(
                    lambda df_group: bilateral_methods(
                        df_group,
                        *args,
                        method=method,
                        plot=plot,
                    )
            )
        )

    periods = df[date_col].unique()
    no_of_periods = len(periods)

    index_vals = np.zeros(no_of_periods)

    if method != 'tpd':
        # Obtain bilateral function for bilateral method.
        func = globals()[method]

    for i in range(no_of_periods):
        df_base = df.loc[df[date_col] == periods[base_month-1]]
        df_curr = df.loc[df[date_col] == periods[i]]

        # Make sure the sample is matched for given periods.
        df_base = df_base[df_base[product_id_col].isin(df_curr[product_id_col])]
        df_curr = df_curr[df_curr[product_id_col].isin(df_base[product_id_col])]

        if method == 'tpd':
            # Use multilateral TPD method with two periods.
            df_matched = (
                pd.concat([df_base, df_curr])
                .drop_duplicates()
            )
            # Recalculate weights for matched df.
            df_matched = _weights_calc(df_matched)
            # Append values to upper triangular of matrix.
            index_vals[i] = time_dummy(df_matched)[-1]
        else:
            # Find price and quantity vectors of base period and current period.
            p_base = df_base[price_col].to_numpy()
            p_curr = df_curr[price_col].to_numpy()
            data = (p_base, p_curr)

            # Get quantities for bilateral methods that use this information.
            if method in {
                'laspeyres', 'drobish', 'marshall_edgeworth',
                'geom_laspeyres', 'tornqvist', 'fisher',
                'walsh', 'sato_vartia', 'geary_khamis_b', 
                'rothwell'
            }:
                q_base = df_base[quantity_col].to_numpy()
                data += (q_base, )
            if method in {
                'paasche', 'drobish','palgrave',
                'marshall_edgeworth', 'geom_paasche', 'tornqvist',
                'fisher', 'walsh', 'sato_vartia',
                'geary_khamis_b'
            }:
                q_curr = df_curr[quantity_col].to_numpy()
                data += (q_curr, )

            # Determine the bilaterals for each base and current period and
            # append to upper tringular of matrix.
            index_vals[i] = func(*data)

    output_df = (
        pd.DataFrame(
            index_vals,
            index=periods
        )
        .rename({0: 'index_value'}, axis=1)
    )
    if plot:
        sns.set(rc={'figure.figsize':(11, 4)})
        (output_df * 100).plot(linewidth=2)
    return output_df

def multilateral_methods(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str='month',
    product_id_col: str='id',
    extension_method: str = 'pure',
    window: int = None,
    groups: Optional[Sequence[str]] = None,
    method: str = 'all',
    bilateral_method: str = 'tornqvist',
    td_engine: str = 'numpy',
    plot: bool = False,
) -> pd.DataFrame:
    """
    Calculate all the multilateral indices.

    Currently supported: GEKS (geks), Geary-Khamis (gk), Time Product Dummy
    (tpd) and Time Hedonic Dummy (tdh).

    Parameters
    ----------
    df: pandas DataFrame
        Contains price and quantity columns, a time series column, and a product
        ID column as a minimum. A characteristics column should also be present
        for hedonic methods.
    price_col: str, defaults to 'price'
        User-defined name for the price column.
    quantity_col: str, defaults to 'quantity'
        User-defined name for the quantity column.
    date_col: str, defaults to 'month'
        User-defined name for the date column.
    product_id_col: str, defaults to 'id'
        User-defined name for the product ID column.
    characteristics: list of str, defaults to None
        The names of the characteristics columns.
    groups: list of str, defaults to None
        The names of the groups columns.
    method: str, defaults to 'all'
        Options: {'all', 'geks', gk', 'tpd', 'tdh'}

        The multilateral method to apply. The 'all' option uses the
        GEKS paired with a bilateral, GK and TPD index.
    bilateral_method: str, defaults to 'tornqvist'
        Options: {'carli', 'jevons', 'dutot', 'laspeyres', 'paasche',
        'geom_laspeyres', 'geom_paasche', 'drobish', 'marshall_edgeworth',
        'palgrave', 'fisher', 'tornqvist', 'walsh', 'sato_vartia',
        'geary_khamis_b', 'rothwell'}

        The bilateral method to pair with `method='geks'`.
    td_engine: str, defaults to 'numpy'
        Options: {'numpy', 'statsmodels', 'sklearn', 'pyspark'}

        Engine to use for wls computation with `method='tpd'`.
    plot: bool, defaults to False
        Boolean parameter on whether to plot the resulting timeseries for price
        indices.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and index values.
    """
    # Obtain unique time periods present in the data.
    periods = df[date_col].unique()

    if extension_method == 'pure':
        window = len(periods)
    else:
        if not window:
            window = 13

    method, bilateral_method = method.lower(), bilateral_method.lower()

    valid_methods =  {'all', 'geks', 'gk', 'tpd', 'tdh'}
    valid_bilateral_methods = {
        'carli', 'jevons', 'dutot', 'laspeyres',
        'paasche', 'geom_laspeyres', 'geom_paasche', 'drobish',
        'marshall_edgeworth', 'palgrave', 'fisher', 'tornqvist',
        'walsh', 'sato_vartia', 'geary_khamis_b', 'tpd', 'rothwell'
    }

    if method not in valid_methods:
        raise ValueError("Invalid option, please select a valid method.")

    if method in {'all', 'geks'} and bilateral_method not in valid_bilateral_methods:
        raise ValueError("Invalid option, please select a valid bilateral method for GEKS.")

    args = (price_col, quantity_col, date_col, product_id_col)

    if groups:
        return (
            df
            .groupby(groups)
            .apply(
                    lambda df_group: multilateral_methods(
                        df_group,
                        *args,
                        extension_method=extension_method,
                        window=window,
                        groups=None,
                        method=method,
                        bilateral_method=bilateral_method,
                        td_engine=td_engine,
                        plot=plot
                    )
            )
        )
    # Calculate weights for each item in each period.
    df = _weights_calc(df, *args)

    func_dict = {'geks': geks, 'tpd': time_dummy, 'gk': geary_khamis}
    func = func_dict.get(method)

    rolling_revisions = rolling_window(df, method, *args, window, bilateral_method)

    if extension_method == 'pure':
        if method == 'geks':
            args += (bilateral_method, )
        df_pivoted = df.set_index([date_col, product_id_col]).unstack(product_id_col)
        return (
            func(df_pivoted, *args)
            .rename_axis(date_col)
            .to_frame()
            .rename({0: 'index_value'}, axis=1)
        )
    else:
        splice_method_dict = {'movement': movement_splice, 'wisp': wisp}
        splice_method = splice_method_dict.get(extension_method)
        index_vals = splice_method(rolling_revisions, window-1)
    
    initial_window = rolling_revisions.iloc[:window, window-1]
    index_vals =  pd.concat([initial_window, index_vals])

    if plot:
        sns.set(rc={'figure.figsize':(11, 4)})
        (index_vals * 100).plot(linewidth=2)

    return index_vals.rename({0: 'index_value'}, axis=1)


def rolling_window(
    df: pd.DataFrame,
    method,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'date',
    product_id_col: str = 'id',
    window: int = 13,
    bilateral_method: str = 'tornqvist',
) -> pd.DataFrame:
    """Calculate the time dummy indices over a dynamic window.

    This function will calculate the TPD indices for each window
    (rolling or expanding). 

    Parameters
    ----------
    window_type: str, default 'rolling'
        The type of the dynamic window, whether to use a rolling or
        expanding window.

    Returns
    -------
    pandas DataFrame
        A square matrix dataframe containing time dummy index values
        for each (rolling or expanding) window.

    """
    args = (price_col, quantity_col, date_col, product_id_col)
    if method == 'geks':
        args += (bilateral_method, )

    func_dict = {'geks': geks, 'tpd': time_dummy, 'gk': geary_khamis}
    func = func_dict.get(method)

    # Gets a single time series across the index axis. This is necessary
    # to apply the window functions over each period.
    pivoted = df.set_index([date_col, product_id_col]).unstack(product_id_col)

    windows = pivoted.rolling(window)

    output_df = pd.DataFrame()
    for window_df in windows:
        # Get latest period for name for this window's output index.
        latest_period = window_df.index[-1]

        index_vals = (
            func(window_df, *args)
            .rename(latest_period)
        )

        output_df = pd.concat([output_df, index_vals], axis=1)

    return output_df.rename_axis(date_col)

