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
from typing import Sequence, Optional, Union

import pandas as pd
import seaborn as sns

from .helpers import _weights_calc
from .bilateral import *
from .multilateral import *

__author__ = ['Dr. Usman Kayani']

def bilateral_methods(
    df: pd.DataFrame,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str='month',
    product_id_col: str='id',
    groups: Optional[Sequence[str]] = None,
    method: str = 'tornqvist',
    base_month: Union[int, str] = 1,
    reference_month: Union[int, str] = 1,
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
        'palgrave', 'fisher', 'tornqvist', 'walsh', 'sato_vartia', 'lowe',
        'geary_khamis_b', 'tpd', 'rothwell'}

        The bilateral method to use.
    base_month: int or str, defaults to 1
        Integer or string specifying the base month. An integer specifies the
        position while a string species the month in the format 'YYYY-MM'.
    reference_month: int or str, defaults to 1
        Integer or string specifying the reference month for rebasing if
        different from the base month. An integer specifies the position while a
        string species the month in the format 'YYYY-MM'.
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
        'carli', 'jevons', 'dutot', 'laspeyres', 'lowe',
        'paasche', 'geom_laspeyres', 'geom_paasche', 'drobish',
        'marshall_edgeworth', 'palgrave', 'fisher', 'tornqvist',
        'walsh', 'sato_vartia', 'geary_khamis_b', 'tpd', 'rothwell'
    }

    if method not in valid_bilateral_methods:
        raise ValueError("Invalid option, please select a valid bilateral method.")

    args = (price_col, quantity_col, date_col, product_id_col)

    periods = sorted(df[date_col].unique())
    no_of_periods = len(periods)

    if isinstance(base_month, str):
        base_month = periods.index(base_month) + 1

    if isinstance(reference_month, str):
        reference_month = periods.index(reference_month) + 1

    # Obtain the base period in the dataframe.
    base_period = periods[base_month-1]

    # Determine product IDs present in the base period.
    df_base_master = df.loc[df[date_col] == base_period]
    keep_ids = df_base_master.loc[:, product_id_col].unique()

    # Filter df to remove product IDs not present in the base period.
    df = df[df[product_id_col].isin(keep_ids)].reset_index(drop=True)

    if groups:
        return (
            df
            .groupby(groups)
            .apply(
                    lambda df_group: bilateral_methods(
                        df_group,
                        *args,
                        method=method,
                        base_month=base_month,
                        reference_month=reference_month,
                        plot=plot,
                    )
            )
            .reset_index()
            .rename({'level_1': 'month'}, axis=1)
        )

    index_vals = np.zeros(no_of_periods)

    if method != 'tpd':
        # Obtain bilateral function for bilateral method.
        func = globals()[method]

    for i in range(no_of_periods):
        # Get data for base and current period.
        df_base = df_base_master
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
                'rothwell', 'lowe'
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
    output_df.sort_index(inplace=True)
    if base_month != reference_month:
        # Rebase the index values to the reference month.
        output_df = output_df / output_df.iloc[reference_month-1]
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
    characteristics: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    method: str = 'all',
    bilateral_method: str = 'tornqvist',
    td_engine: str = 'numpy',
    reference_month: Union[int, str] = 1,
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
        'palgrave', 'fisher', 'tornqvist', 'walsh', 'sato_vartia', 'lowe',
        'geary_khamis_b', 'rothwell'}

        The bilateral method to pair with `method='geks'`.
    td_engine: str, defaults to 'numpy'
        Options: {'numpy', 'statsmodels', 'sklearn', 'pyspark'}

        Engine to use for wls computation with `method='tpd'`.
    reference_month: int or str, defaults to 1
        The month to use as the reference month for the multilateral methods. An
        integer specifies the position while a string species the month in the
        format 'YYYY-MM'.
    plot: bool, defaults to False
        Boolean parameter on whether to plot the resulting timeseries for price
        indices.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the timeseries and index values.
    """
    method, bilateral_method = method.lower(), bilateral_method.lower()

    valid_methods =  {'all', 'geks', 'gk', 'tpd', 'tdh'}
    valid_bilateral_methods = {
        'carli', 'jevons', 'dutot', 'laspeyres', 'lowe',
        'paasche', 'geom_laspeyres', 'geom_paasche', 'drobish',
        'marshall_edgeworth', 'palgrave', 'fisher', 'tornqvist',
        'walsh', 'sato_vartia', 'geary_khamis_b', 'tpd', 'rothwell'
    }

    if method not in valid_methods:
        raise ValueError("Invalid option, please select a valid method.")

    if method in {'all', 'geks'} and bilateral_method not in valid_bilateral_methods:
        raise ValueError("Invalid option, please select a valid bilateral method for GEKS.")

    args = (price_col, quantity_col, date_col, product_id_col)

    # Obtain unique time periods present in the data.
    periods = sorted(df[date_col].unique())

    if isinstance(reference_month, str):
        reference_month = periods.index(reference_month) + 1

    if groups:
        return (
            df
            .groupby(groups)
            .apply(
                    lambda df_group: multilateral_methods(
                        df_group,
                        *args,
                        characteristics=characteristics,
                        method=method,
                        bilateral_method=bilateral_method,
                        td_engine=td_engine,
                        reference_month=reference_month,
                        plot=plot
                    )
            )
            .reset_index()
            .rename({'level_1': 'month'}, axis=1)
        )
    if quantity_col not in df.columns:
        df[quantity_col] = 1
    if bilateral_method not in ('jevons', 'carli', 'dutot'):
        # Calculate weights for each item in each period.
        df = _weights_calc(df, *args)

    if method == 'all':
        index_vals = {
            f'index_value_geks': geks(df, *args, bilateral_method),
            'index_value_gk': geary_khamis(df, *args),
            'index_value_td': time_dummy(df, *args, characteristics, engine=td_engine)
        }
    elif method == 'geks':
        index_vals = geks(df, *args, bilateral_method)
    elif method == 'gk':
        index_vals = geary_khamis(df, *args)
    elif method == 'tpd':
        index_vals = time_dummy(df, *args, None, engine=td_engine)
    elif method == 'tdh':
        if not characteristics:
            raise ValueError("Characteristics required for TDH.")
        else:
            index_vals = time_dummy(df, *args, characteristics, engine=td_engine)
    output_df = (
        pd.DataFrame(
            index_vals,
            index=periods
        )
        .rename({0: 'index_value'}, axis=1)
    )
    output_df.sort_index(inplace=True)
    if reference_month != 1:
        output_df = output_df / output_df.iloc[reference_month - 1]
    if plot:
        sns.set(rc={'figure.figsize':(11, 4)})
        (output_df * 100).plot(linewidth=2)
    return output_df
