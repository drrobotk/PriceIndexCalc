from typing import Sequence

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

__author__ = ['Dr. Usman Kayani']

def wls_pyspark(
    df: SparkDF,
    dependent_var: str,
    independent_vars: Sequence[str],
) -> LinearRegressionModel:
    """Perform weighted least squares regression in PySpark.

    Parameters
    ----------
    df : SparkDF
        Contains columns for each object in the formula and a weights column as
        a minimum.
    dependent_var: str
        The dependent variable for the regression.
    independent_vars: list of str
        The independent variables for the regression.

    Returns
    -------
    LinearRegressionModel
        Weighted linear regression model derived from a least-squares fit.

    """
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