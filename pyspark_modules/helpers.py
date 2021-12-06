from pyspark.sql import (
    functions as F,
    Column as SparkCol,
)
from pyspark.sql.window import Window

__author__ = ['Dr. Usman Kayani']

def _weights_calc_pyspark(
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    date_col: str = 'month',
)-> SparkCol:
    """Calculate weights from expenditure shares in PySpark."""
    window = Window.partitionBy(date_col)
    expenditure = F.col(price_col)*F.col(quantity_col)
    return expenditure / F.sum(expenditure).over(window)
  
def _cumprod_over_period(
    col: SparkCol,
    date_col: str = 'period'
) -> SparkCol:
    """Cumulative product of numeric column over a period window."""
    window = (
        Window
        .orderBy(date_col)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return F.exp(F.sum(F.log(col)).over(window))
