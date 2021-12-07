"""
Provides the following multilateral methods:

* :func:`time_dummy_pyspark`
* :func:`geary_khamis_pyspark`

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

from typing import List, Sequence, Optional

import numpy as np
from pyspark.mllib.linalg.distributed import (
    IndexedRow,
    IndexedRowMatrix,
    CoordinateMatrix,
    MatrixEntry,
    DenseMatrix,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import (
    DataFrame as SparkDF,
    functions as F,
)
from pyspark import SparkContext

from .weighted_least_squares import wls_pyspark

__author__ = ['Dr. Usman Kayani']

def time_dummy_pyspark(
    df: SparkDF,
    Number_of_periods: int,
    price_col: str = 'price',
    date_col: str = 'month',
    product_id_col: str = 'id',
    characteristics: Optional[Sequence[str]] = None,
) -> List:
    """Obtain the time dummy indices for a given dataframe in PySpark.

    Calculates the time dummy indices using a formula with weighted least
    squares regression.  When passed with characteristics, this function returns
    the Time Dummy Hedonic indices. When passed without it returns the Time
    Product Dummy indices.
    """  
    # Calculate logarithm of the prices for each item.
    df = df.withColumn('log_price', F.log(price_col))

    non_time_vars = characteristics if characteristics else [product_id_col]

    # WLS regression with labels, features & weights -> fit model.
    model = (
        wls_pyspark(
            df,
            dependent_var='log_price',
            independent_vars=[date_col, *non_time_vars],
        )
    )

    # Extracting time dummy coefficients.
    time_dummy_coeff = model.coefficients[:Number_of_periods-1][::-1]
    
    # Get indices from the time dummy coefficients & set first = 1.
    return [1, *np.exp(time_dummy_coeff)]

def geary_khamis_pyspark(
    df: SparkDF,
    sc: SparkContext,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
    product_id_col: str = 'id',
) -> List:
    """Obtain the Geary-Khamis indices for a given dataframe in PySpark.

    Calculates the Geary-Khamis indices using matrix operations.
    """
    pivoted_df = (
        df.groupby(product_id_col)
        .pivot(date_col)
    )

    matrix_dfs = []

    for col in [price_col, quantity_col, 'weights']:
        matrix_dfs.append(
            pivoted_df
            .avg(col)
            .sort(product_id_col)
            .drop(product_id_col)
            .fillna(0)
        )
    cols = matrix_dfs[0].columns
    
    vectors = []
    for mdf in matrix_dfs:
        vectors.append(
            VectorAssembler(inputCols=cols, outputCol='vec')
            .transform(mdf)
            .select('vec')
            .collect()
        )
    
    N = len(vectors[0])
    M = len(cols)

    matrices = []
    for vec in vectors:
        matrices.append(
            IndexedRowMatrix(
                sc
                .range(N)
                .map(lambda i: IndexedRow(i, vec[i][0].array))
            )
        )
    
    qsum_arr = matrices[1].rows.map(lambda row: row.vector.sum()).collect()
    qsum_inv_mat = CoordinateMatrix(
        sc
        .range(N)
        .map(lambda i: MatrixEntry(i, i, 1/qsum_arr[i])), N, N
    )

    C = (
        qsum_inv_mat.toBlockMatrix()
        .multiply(matrices[2].toBlockMatrix())
        .multiply(matrices[1].toBlockMatrix().transpose())
    )


    Identity = (
        CoordinateMatrix(
            sc
            .range(N)
            .map(lambda i: MatrixEntry(i, i, 1.0)), N, N
        )
        .toBlockMatrix()
    )

    R = (
        CoordinateMatrix(
            sc
            .range(N)
            .map(lambda i: MatrixEntry(0, i, 1.0)), N, N
        )
        .toBlockMatrix()
    )

    svd = (
        R.add(Identity).subtract(C)
        .toIndexedRowMatrix()
        .computeSVD(N, True)
    )

    s = svd.s
    V = svd.V
    U = svd.U

    V_ = DenseMatrix(N, N, V.toArray().transpose().reshape(-1))
    invS = DenseMatrix(len(s), len(s), np.diag(1/s).reshape(-1))

    c_ = (
        CoordinateMatrix(
            sc
            .range(N)
            .map(lambda i: MatrixEntry(0, 0, 1/N)), N, 1
        )
        .toBlockMatrix()
        .transpose()
    )

    b = (
        c_.multiply(U.multiply(invS).multiply(V_).toBlockMatrix())
        .toLocalMatrix()
        .toArray()
    )

    mat_prices_ = (
        matrices[0]
        .toBlockMatrix()
        .transpose()
        .toLocalMatrix()
        .toArray()
    )

    mat_quantities_ = (
        matrices[1]
        .toBlockMatrix()
        .transpose()
        .toLocalMatrix()
        .toArray()
    )

    price_levels = []
    for i in range(M):
        val = mat_prices_[i].dot(mat_quantities_[i])/mat_quantities_[i].dot(b[0])
        price_levels.append(val)

    initial_pl = price_levels[0]

    return [pl/initial_pl for pl in price_levels]