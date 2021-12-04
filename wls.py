from typing import Sequence, Union

import pandas as pd
import numpy as np
import statsmodels.api as SM
from sklearn.linear_model import LinearRegression as LR
from pyspark.ml import Pipeline
from pyspark.ml.regression import (
    LinearRegression,
    LinearRegressionModel,
)
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.sql import DataFrame as SparkDF

from helpers import _vars_split

__author__ = ['Dr. Usman Kayani']

def wls(
    df: Union[SparkDF, pd.DataFrame],
    dependent_var: str,
    independent_vars: Sequence[str],
    engine: str = 'numpy',
):
    """Perform weighted least squares regression.

    Parameters
    ----------
    df : SparkDF or pd.DataFrame
        Contains columns for each object in the formula and a weights
        column as a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.
    weights_col : str, defaults to 'weights'
        User-defined weight column name.
    engine: str, defaults to 'numpy'
        Options: {'numpy', 'statsmodels', 'sklearn', 'pyspark'}

        Engine to use for wls computation.

    Returns
    -------
    pyspark LinearRegressionModel
        A weighted linear regression model derived from a least-squares
        fit.

    """
    return globals()[f'wls_{engine}'](df, dependent_var, independent_vars)

def wls_numpy(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> pd.Series:
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
    # Obtain variables and weight for wls regression.
    X, Y, weights = _get_vars(df, dependent_var, independent_vars)

    # Sklearn WLS to fit model.
    coefficients = (
        LR(fit_intercept=False)
        .fit(X, Y, sample_weight=weights)
        .coef_
    )

    return pd.Series(coefficients, index=X.columns)

def wls_pyspark(
    df: SparkDF,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> LinearRegressionModel:
    # Set up stages for pipeline.
    indexers, encoders, vec_cols = [], [], []
    for column in independent_vars:
        # Map the string cols of labels to numeric values.
        indexers.append(
            StringIndexer(
                inputCol=column,
                outputCol=f'{column}_numeric',
                stringOrderType='alphabetDesc',
            )
            .fit(df)
        )

        # Encode the numeric values to dummy variables.
        encoders.append(
            OneHotEncoder(
                inputCol=f'{column}_numeric',
                outputCol=f'{column}_vector',
            )
        )
        vec_cols.append(f'{column}_vector')

    # Transform vectors for dummy vars into a single vector (features).
    assembler = VectorAssembler(inputCols=vec_cols, outputCol="features")
    # Create a pipeline to perform the sequence of stages above.
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    # Fit & transform the dataframe according to pipeline for model.
    model_df = pipeline.fit(df).transform(df)

    # WLS regression with labels, features & weights -> fit model.
    wls_model = LinearRegression(
        labelCol=dependent_var,
        featuresCol='features',
        weightCol='weights',
    )
    return wls_model.fit(model_df)

def _get_vars(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: Sequence[str],
):
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
