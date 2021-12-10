import pandas as pd
import numpy as np

from .helpers import diag

def movement_splice(revisions: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Apply the movement splice revision extension method.

    Parameters
    ----------
    revisions : pandas DataFrame
        A DataFrame of the rolling revisions. The values only need to be
        correct for the size of the moving window, as only the main
        diagonal and values shifted <= window length are used.
    window : int, default 12
        The size of the moving window excluding the base period.

    Returns
    -------
    pandas DataFrame
        The spliced index revisions. 
    """
    # Get movement splice by dividing by the previous revision value and
    # take the cumulative product.
    movement_splice = (
        revisions.div(revisions.shift(1))
        .pipe(diag)
        # The revisions begin after the first full window.
        .iloc[window+1:]
        .cumprod()
    )

    # Multiply by the value at the end of the initial revised window.
    return movement_splice.mul(revisions.iat[window, window])

def hasp(
    revisions: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """Apply the half window splice revision extension method."""
    return wisp(revisions, window, half_window=True)

def wisp(
    revisions: pd.DataFrame,
    window: int = 12,
    half_window: bool = False,
) -> pd.DataFrame:
    """Apply the window splice revision extension method.

    Pass ``half_window=True`` for the half window splice.

    Parameters
    ----------
    %(splice.parameters)s
    half_window : bool, default False
        Computes half window splice when True.

    Returns
    -------
    %(splice.returns)s
    """
    # Need to retain original window len if half window is selected.
    full_window = window
    if half_window:
        window = int(np.floor(window/2))

    # Divide each rolling revision value by the base value.
    window_splice = (
        revisions.div(revisions.shift(window))
        .pipe(diag)
        # The revisions begin after the first full window.
        .iloc[full_window+1:]
        # Take cumprod every n periods where n is window length.
        .pipe(cumprod_over_periods, periods=window)
    )

    # Take full_window+1 as the initial window to include base period.
    initial_window = revisions.iloc[:full_window+1, full_window]

    # Repeat initial window values for the length of the revisions.
    repeated_window = repeat_values(
        # Using -window covers full and half window options.
        initial_window.iloc[-window:],
        # We only want to repeat it from the start of the revisions.
        index=window_splice.index,
    )

    # Multiply by repeated initial window to get revisions.
    return window_splice.mul(repeated_window)


def cumprod_over_periods(df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """Take the cumprod of values that share the same window period."""
    # Bin the values up by period number.
    periods = np.resize(range(periods), len(df))
    period_aligned = df.set_index(periods, append=True).unstack()
    output = period_aligned.cumprod()
    # Return to original shape and type.
    return output.stack().droplevel(-1).astype(df.dtypes)


def repeat_values(
    values,
    index,
) -> pd.DataFrame:
    """Repeat values to length of given index returning as DataFrame."""
    return pd.DataFrame(np.resize(values, len(index)), index=index)