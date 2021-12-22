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
    """Apply the half window splice revision extension method.
    
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

def mean_pub(revisions: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Apply the mean splice revision extension method.

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
    # Since the geometric mean is a product of all values raised to the
    # power of the reciprocal (of the window length), first multiply
    # all the terms together.

    # When considering a full window plus the base period value i.e. n+1
    # values, where n is the window length, take a rolling product of
    # the first n values which includes the base period and divide that
    # by the latest available revised value for that period i.e. the
    # (n+1)th value, raised to the power of n.
    rolling_splice = (
        revisions.pow(window)
        .div(
            revisions.rolling(window).apply(np.prod, raw=True)
            .shift(1)
        )
        .pipe(diag)
    )

    # Remove the datetime name of the initial window and convert to
    # frame so it can be multiplied by the rolling_splice.
    initial_window = (
        revisions.iloc[:window+1, window]
        .rename(None)
        .to_frame()
    )

    # Initialise the final indices with the initial_window.
    indices = initial_window.reindex(revisions.index)

    # The latest revised index value needs to be known for the following
    # steps. No simple vectorised method was found, so loop through
    # appending the latest value each time.
    for _ in range(len(revisions.index) - window):
        # Fill back with the initial window since rolling_splice only
        # has values in the revisions periods, which results in NAs in
        # the initial window after the final index calculation step.
        indices = indices.fillna(initial_window)

        # Get the additional multiplication terms in a similar way, by
        # taking a rolling product of the indices and dividing by the
        # latest revised value to the power of n, where n is the window
        # length.
        index_splice = (
            indices.rolling(window).apply(np.prod, raw=True)
            .div(indices.pow(window))
            # Shift to align with the latest period being calculated.
            .shift(1)
        )

        # Calculate the indices by multiplying all the values together
        # and raising to the power of the reciprocal of the window
        # length (1/n) to get the geometric mean, and multiplying the
        # result by the previous index value.
        indices = (
            rolling_splice.mul(index_splice).pow(1/window)
            .mul(indices.shift(1))
        )

    # Return only the revisions.
    return indices.iloc[window+1:]


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