import pandas as pd

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