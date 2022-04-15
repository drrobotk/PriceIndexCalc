"""Geary-Khamism (pure) index in PySpark using a native PySpark method.

Provides the following function:

* :func:`geary_khamis`
"""
from typing import (
    Optional,
    Sequence,
    Union,
)

from pyspark.sql import (
    Column as SparkCol,
    DataFrame as SparkDF,
    functions as F,
    Window,
    WindowSpec,
    SparkSession
)

import pandas as pd

def get_window_spec(groups: Optional[Sequence[str]] = None) -> WindowSpec:
    """Return WindowSpec partitioned by levels, defaulting to whole df."""
    return Window.partitionBy(groups) if groups else Window.partitionBy()

def get_weight_shares(
    weights: str,
    levels: Optional[Union[str, Sequence[str]]] = None,
) -> SparkCol:
    """Divide weights by sum of weights for each group."""
    return weights / F.sum(weights).over(get_window_spec(levels))

def geary_khamis_pure(
    df: SparkDF,
    groups: Union[str, Sequence[str]] = None,
    no_of_iterations: int = 5,
    precision: float = None,
    checkpoint: bool = False,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
    product_id_col: str = 'id',
) -> SparkDF:
    """
    Calculate the index values with the Geary-Khamis iterative method using
    native Spark functionality.
    """
    index_groups = [*groups, date_col] if groups else [date_col]
    levels = [*groups, product_id_col] if groups else [product_id_col]

    # Define and price and quantities vectors.
    prices = F.col(price_col)
    quantities = F.col(quantity_col)

    # Initialise price levels as 1's to find the solution with
    # iteration.
    price_levels = F.lit(1)
    if precision:
        df = df.withColumn('price_levels', price_levels)

    # Define windows for sum over groups.
    window_levels = get_window_spec(levels)
    window_id = get_window_spec([product_id_col])
    window_index_groups = get_window_spec(index_groups)
    window_groups = get_window_spec(groups).orderBy(date_col)

    # Calculate the quantity share and turnover for the iteration.
    quantity_share = quantities / F.sum(quantities).over(window_levels)
    turnover = F.sum(prices*quantities).over(window_index_groups)

    # Iterate until we reach the set level of precision, or after a set
    # number of iterations if they do not converge.
    for iteration in range(no_of_iterations):
        # Obtain matrices for iterative calculation.
        deflated_prices = prices / price_levels
        factors = F.sum(deflated_prices * quantity_share).over(window_id)

        # Determine the weighted quantity index for current price
        # levels.
        weighted_quality_index = (
            F.sum(quantities * factors)
            .over(window_index_groups)
        )

        # Calculate new price levels from previous value.
        new_price_levels = turnover / weighted_quality_index
        df = df.withColumn('new_price_levels', new_price_levels)

        if precision:
            pl_diff = F.col('new_price_levels') - F.col('price_levels')
            is_precise = (
                not bool(df.filter(F.abs(pl_diff) > precision).take(1))
            )
            if is_precise:
                print(f'Precision of {precision} in {iteration} iterations.')
                break

        # Otherwise set price level for next iteration. We set this from
        # the column values appended to the df above, rather than the
        # SparkCol object as this works much faster within Spark.
        price_levels = F.col('new_price_levels')
        df = df.withColumn('price_levels', price_levels)

        # todo: add a check for precision?
        if checkpoint and iteration % 10 == 0:
            df = df.checkpoint()

    # Normalize by first period for final output.
    indices = price_levels / F.first(price_levels).over(window_groups)

    return (
        df
        .withColumn('index_value', indices)
        .groupby(index_groups)
        .agg(F.first('index_value').alias('index_value'))
        .orderBy(index_groups)
    )

df = pd.read_csv('tests/test_data/large_input_df.csv')

spark = SparkSession.builder.master('local').getOrCreate()

df = spark.createDataFrame(df)

geary_khamis_pure(df, groups=['group']).show(100)