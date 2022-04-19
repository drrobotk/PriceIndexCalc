from typing import Sequence, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as SM
from sklearn.linear_model import LinearRegression as LR

from .helpers import _vars_split

__author__ = ['Dr. Usman Kayani']

def wls(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
    engine: str = 'numpy',
) -> pd.Series:
    """Perform weighted least squares regression.

    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.
    engine: str, defaults to 'numpy'
        Options: {'numpy', 'statsmodels', 'sklearn'}

        Engine to use for wls computation.

    Returns
    -------
    pd.Series
        Coefficients for a weighted linear regression model derived from a
        least-squares fit.

    """
    return globals()[f'wls_{engine}'](df, dependent_var, independent_vars)

def wls_numpy(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> pd.Series:
    """
    Weighted least squares with numpy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.

    Returns
    -------
    pd.Series
        Coefficients for a weighted linear regression model derived from a
        least-squares fit.
    """
    # Obtain variables and weight for wls regression.
    X, Y, weights = _get_vars(df, dependent_var, independent_vars)
    W = np.array(weights)

    # Matrix WLS to fit model.
    coeffs = np.linalg.pinv(X.T @ (W[:, None] * X)) @ (X.T @ (W * Y))    
    
    return pd.Series(coeffs, index=X.columns)

def wls_statsmodels(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> pd.Series:
    """
    Weighted least squares with statsmodels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.
    
    Returns
    -------
    pd.Series
        Coefficients for a weighted linear regression model derived from a
        least-squares fit.
    """
    # Obtain variables and weight for wls regression.
    X, Y, weights = _get_vars(df, dependent_var, independent_vars)
    
    # Statsmodels WLS to fit model.
    model = SM.WLS(Y,X, weights=weights).fit(params_only=True)
    return model.params

def wls_sklearn(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> pd.Series:
    """
    Weighted least squares with Sklearn.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.
    
    Returns
    -------
    pd.Series
        Coefficients for a weighted linear regression model derived from a
        least-squares fit.
    """
    # Obtain variables and weight for wls regression.
    X, Y, weights = _get_vars(df, dependent_var, independent_vars)

    # Sklearn WLS to fit model.
    coefficients = (
        LR(fit_intercept=False)
        .fit(X, Y, sample_weight=weights)
        .coef_
    )

    return pd.Series(coefficients, index=X.columns)

def _get_vars(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Get variables for least squares regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.

    Returns
    -------
    tuple of pd.Series
        X, Y, and weights for least squares regression.

    """
    # Split the categoerical and numerical vars.
    categorical_vars, numerical_vars = _vars_split(df[independent_vars[1:]])

    # Convert categorical variables to dummy.
    df_dummy = pd.get_dummies(
        df,
        columns=[independent_vars[0], *categorical_vars],
        drop_first=True
    )

    # Create a column with all values equal to 1 for the constant term in the regression.
    df_dummy['const'] = 1

    # Determine the independent variables.
    vars = df_dummy.columns.difference(df.columns).tolist() + numerical_vars

    return df_dummy[vars], df_dummy[dependent_var], df_dummy['weights']
