"""
Apache Spark Batch Processing Jobs
Computes historical baseline features:
  - Salary delay days
  - Utility payment delay average
  - Discretionary spend trends
  - Demographic & fairness features

Runs on real Spark cluster (Master + Worker in Docker).
"""
import logging
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DecimalType, TimestampType, BooleanType, FloatType,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import SparkConfig, PostgresConfig

logger = logging.getLogger(__name__)

JDBC_URL = f"jdbc:postgresql://{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
JDBC_PROPS = {
    "user": PostgresConfig.USER,
    "password": PostgresConfig.PASSWORD,
    "driver": "org.postgresql.Driver",
}


def get_spark_session(app_name: str = "PDI-BatchProcessor") -> SparkSession:
    """Create SparkSession connected to the Spark cluster."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(SparkConfig.MASTER_URL)
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_table(spark: SparkSession, table_name: str):
    """Read a PostgreSQL table into a Spark DataFrame."""
    return spark.read.jdbc(JDBC_URL, table_name, properties=JDBC_PROPS)


def compute_salary_delay(spark: SparkSession):
    """
    Compute salary delay days for each customer.
    Logic: Compare expected salary day (from customers table) with actual
    salary credit dates from transactions.
    """
    logger.info("[Spark] Computing salary delay features...")

    customers_df = read_table(spark, "customers").select(
        "customer_id", "salary_credit_day", "monthly_salary"
    )

    # Get salary credit transactions
    txns_df = read_table(spark, "transactions").filter(
        (F.col("txn_type") == "salary_credit") & (F.col("status") == "success")
    ).select("customer_id", "amount", "timestamp")

    # Extract month and day from timestamp
    salary_df = txns_df.withColumn(
        "credit_day", F.dayofmonth("timestamp")
    ).withColumn(
        "credit_month", F.date_format("timestamp", "yyyy-MM")
    )

    # Join with expected salary day
    joined = salary_df.join(customers_df, "customer_id", "left")

    # Compute delay: actual_day - expected_day (handle month wraparound)
    delay_df = joined.withColumn(
        "raw_delay", F.col("credit_day") - F.col("salary_credit_day")
    ).withColumn(
        "delay_days", F.when(F.col("raw_delay") < -15, F.col("raw_delay") + 30)
                       .otherwise(F.col("raw_delay"))
    )

    # Average delay per customer (most recent 3 months)
    three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    recent_delay = delay_df.filter(F.col("timestamp") >= three_months_ago)

    result = recent_delay.groupBy("customer_id").agg(
        F.round(F.avg("delay_days"), 0).cast(IntegerType()).alias("salary_delay_days"),
    )

    logger.info(f"[Spark] Computed salary delay for {result.count()} customers")
    return result


def compute_utility_delay(spark: SparkSession):
    """
    Compute average utility payment delay over 3-6 months.
    Logic: Utility bills are typically due on the 1st-5th. Measure how many
    days past the 5th each payment occurs.
    """
    logger.info("[Spark] Computing utility payment delay features...")

    txns_df = read_table(spark, "transactions").filter(
        (F.col("merchant_category") == "utility") &
        (F.col("direction") == "debit")
    ).select("customer_id", "status", "timestamp", "amount")

    # Utility payments typically expected in first 5 days of month
    utility_df = txns_df.withColumn(
        "payment_day", F.dayofmonth("timestamp")
    ).withColumn(
        "payment_month", F.date_format("timestamp", "yyyy-MM")
    )

    # Compute delay from expected due date (day 5)
    delay_df = utility_df.withColumn(
        "utility_delay",
        F.when(F.col("payment_day") > 5, F.col("payment_day") - 5).otherwise(0)
    )

    # Include failed payments as max delay (30 days)
    delay_with_failures = delay_df.withColumn(
        "effective_delay",
        F.when(F.col("status") == "failed", 30).otherwise(F.col("utility_delay"))
    )

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    recent = delay_with_failures.filter(F.col("timestamp") >= six_months_ago)

    result = recent.groupBy("customer_id").agg(
        F.round(F.avg("effective_delay"), 2).alias("utility_payment_delay_avg"),
    )

    logger.info(f"[Spark] Computed utility delay for {result.count()} customers")
    return result


def compute_spend_trends(spark: SparkSession):
    """
    Compute discretionary spend trends.
    Logic: Compare last 7 days' discretionary spend to same period last month.
    Ratio > 1.0 = spending increasing, < 1.0 = decreasing.
    """
    logger.info("[Spark] Computing discretionary spend trend features...")

    discretionary_cats = ["dining", "entertainment", "clothing", "luxury_goods", "travel"]

    txns_df = read_table(spark, "transactions").filter(
        (F.col("merchant_category").isin(discretionary_cats)) &
        (F.col("direction") == "debit") &
        (F.col("status") == "success")
    ).select("customer_id", "amount", "timestamp")

    now = datetime.now()
    current_7d_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    prev_7d_start = (now - timedelta(days=37)).strftime("%Y-%m-%d")
    prev_7d_end = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    # Current period spend
    current_spend = txns_df.filter(
        F.col("timestamp") >= current_7d_start
    ).groupBy("customer_id").agg(
        F.sum("amount").alias("current_spend")
    )

    # Previous period spend (same 7-day window, one month earlier)
    prev_spend = txns_df.filter(
        (F.col("timestamp") >= prev_7d_start) &
        (F.col("timestamp") < prev_7d_end)
    ).groupBy("customer_id").agg(
        F.sum("amount").alias("prev_spend")
    )

    # Compute trend ratio
    trend_df = current_spend.join(prev_spend, "customer_id", "outer").fillna(0)
    result = trend_df.withColumn(
        "discretionary_spend_trend",
        F.when(F.col("prev_spend") > 0,
               F.round(F.col("current_spend") / F.col("prev_spend"), 4))
         .otherwise(1.0)
    ).select("customer_id", "discretionary_spend_trend")

    logger.info(f"[Spark] Computed spend trends for {result.count()} customers")
    return result


def compute_demographics(spark: SparkSession):
    """
    Compute demographic and product-holding features.
    """
    logger.info("[Spark] Computing demographic features...")

    customers_df = read_table(spark, "customers")

    result = customers_df.select(
        "customer_id",
        "credit_score",
        "age",
        "tenure_months",
        "income_bracket",
        "region",
        "gender",
    ).withColumn(
        "product_count",
        F.size(F.col("product_holdings")) if "product_holdings" in customers_df.columns
        else F.lit(1)
    ).withColumn(
        "has_credit_card",
        F.array_contains(F.col("product_holdings"), "credit_card")
        if "product_holdings" in customers_df.columns else F.lit(False)
    ).withColumn(
        "has_personal_loan",
        F.array_contains(F.col("product_holdings"), "personal_loan")
        if "product_holdings" in customers_df.columns else F.lit(False)
    ).withColumn(
        "has_mortgage",
        F.array_contains(F.col("product_holdings"), "home_loan")
        if "product_holdings" in customers_df.columns else F.lit(False)
    )

    logger.info(f"[Spark] Computed demographics for {result.count()} customers")
    return result


def compute_monthly_spend_stats(spark: SparkSession):
    """
    Compute average monthly spend and spend volatility over 3 months.
    """
    logger.info("[Spark] Computing monthly spend statistics...")

    three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    txns_df = read_table(spark, "transactions").filter(
        (F.col("direction") == "debit") &
        (F.col("status") == "success") &
        (F.col("timestamp") >= three_months_ago)
    ).select("customer_id", "amount", "timestamp")

    # Monthly aggregation
    monthly = txns_df.withColumn(
        "month", F.date_format("timestamp", "yyyy-MM")
    ).groupBy("customer_id", "month").agg(
        F.sum("amount").alias("monthly_spend")
    )

    # Compute avg and stddev across months
    result = monthly.groupBy("customer_id").agg(
        F.round(F.avg("monthly_spend"), 2).alias("avg_monthly_spend_3m"),
        F.round(
            F.coalesce(F.stddev("monthly_spend") / F.avg("monthly_spend"), F.lit(0)), 4
        ).alias("spend_volatility_3m"),
    )

    logger.info(f"[Spark] Computed spend stats for {result.count()} customers")
    return result


def run_batch_pipeline():
    """Run the complete batch feature computation pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Engine - Spark Batch Feature Computation")
    print("=" * 70)

    spark = get_spark_session()

    try:
        # Compute all features
        salary_delay = compute_salary_delay(spark)
        utility_delay = compute_utility_delay(spark)
        spend_trends = compute_spend_trends(spark)
        demographics = compute_demographics(spark)
        spend_stats = compute_monthly_spend_stats(spark)

        # Merge all features
        batch_features = demographics
        for df in [salary_delay, utility_delay, spend_trends, spend_stats]:
            batch_features = batch_features.join(df, "customer_id", "left")

        # Fill nulls
        batch_features = batch_features.fillna({
            "salary_delay_days": 0,
            "utility_payment_delay_avg": 0.0,
            "discretionary_spend_trend": 1.0,
            "avg_monthly_spend_3m": 0.0,
            "spend_volatility_3m": 0.0,
        })

        # Write to PostgreSQL (batch_features table)
        batch_features.write.jdbc(
            JDBC_URL,
            "batch_features",
            mode="overwrite",
            properties=JDBC_PROPS,
        )

        count = batch_features.count()
        print(f"\n[Spark] Batch features written for {count} customers")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    run_batch_pipeline()
