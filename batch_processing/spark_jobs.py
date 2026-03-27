# pyre-ignore-all-errors
"""
Apache Spark Batch Processing Jobs
Computes historical baseline features:
  - Salary delay days
  - Utility payment delay average
  - Discretionary spend trends
  - Demographic & fairness features
  - Asset-side: FD premature withdrawal, SIP stoppage, insurance lapse
  - Employer health score
  - Customer segment classification

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

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import SparkConfig, PostgresConfig  # noqa: E402
from config.bank_config import BankProfileLoader  # noqa: E402

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


# ─────────────────────────────────────────────
# M2: Asset-Side Feature Jobs
# ─────────────────────────────────────────────
def compute_fd_premature_withdrawal(spark: SparkSession):
    """
    Detect FD/RD premature closures — strong distress signal.
    A customer breaking a fixed deposit before maturity needs cash urgently.
    """
    logger.info("[Spark] Computing FD premature withdrawal features...")

    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    txns_df = read_table(spark, "transactions").filter(
        (F.col("merchant_category").isin("fixed_deposit", "fd_closure", "recurring_deposit")) &
        (F.col("direction") == "credit") &
        (F.col("timestamp") >= ninety_days_ago)
    ).select("customer_id", "amount", "timestamp", "txn_type")

    result = txns_df.groupBy("customer_id").agg(
        F.count("*").alias("fd_closed_count_90d"),
        F.round(F.sum("amount"), 2).alias("fd_closure_amount_90d"),
    )

    logger.info(f"[Spark] Computed FD closures for {result.count()} customers")
    return result


def compute_sip_stoppage(spark: SparkSession):
    """
    Detect SIP/mutual fund payment stoppage.
    If a customer had regular MF payments for 2+ months then stopped, that's a stress signal.
    """
    logger.info("[Spark] Computing SIP stoppage features...")

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    txns_df = read_table(spark, "transactions").filter(
        (F.col("merchant_category").isin("mutual_fund", "sip", "investment")) &
        (F.col("direction") == "debit") &
        (F.col("status") == "success") &
        (F.col("timestamp") >= six_months_ago)
    ).select("customer_id", "amount", "timestamp")

    # Count MF payments per month per customer
    monthly_mf = txns_df.withColumn(
        "month", F.date_format("timestamp", "yyyy-MM")
    ).groupBy("customer_id", "month").agg(
        F.count("*").alias("mf_count")
    )

    # Count months with MF activity and months without
    months_with_sip = monthly_mf.groupBy("customer_id").agg(
        F.count("*").alias("active_months"),
        F.max("month").alias("last_active_month")
    )

    _current_month = datetime.now().strftime("%Y-%m")
    prev_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m")

    # SIP stopped if had activity in older months but not in last 1 month
    result = months_with_sip.withColumn(
        "sip_stopped_flag",
        F.when(
            (F.col("active_months") >= 2) &
            (F.col("last_active_month") < prev_month),
            F.lit(True)
        ).otherwise(F.lit(False))
    ).withColumn(
        "sip_gaps_3m",
        F.when(F.col("sip_stopped_flag"), F.lit(3) - F.col("active_months"))
         .otherwise(F.lit(0))
    ).select("customer_id", "sip_stopped_flag", "sip_gaps_3m")

    logger.info(f"[Spark] Computed SIP stoppage for {result.count()} customers")
    return result


def compute_insurance_lapse(spark: SparkSession):
    """
    Detect insurance premium lapses — missing or failed insurance payments.
    Missing an insurance premium signals severe cashflow crisis.
    """
    logger.info("[Spark] Computing insurance lapse features...")

    three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    txns_df = read_table(spark, "transactions").filter(
        (F.col("merchant_category").isin("insurance", "lic", "health_insurance")) &
        (F.col("direction") == "debit") &
        (F.col("timestamp") >= three_months_ago)
    ).select("customer_id", "status", "timestamp")

    result = txns_df.groupBy("customer_id").agg(
        F.sum(F.when(F.col("status") == "failed", 1).otherwise(0)).alias("insurance_missed_payments_3m"),
        F.count("*").alias("insurance_total_attempts_3m"),
    ).withColumn(
        "insurance_lapse_flag",
        F.when(F.col("insurance_missed_payments_3m") >= 1, F.lit(True))
         .otherwise(F.lit(False))
    ).select("customer_id", "insurance_lapse_flag", "insurance_missed_payments_3m")

    logger.info(f"[Spark] Computed insurance lapse for {result.count()} customers")
    return result


# ─────────────────────────────────────────────
# M1: Segment Classification (Spark-level)
# ─────────────────────────────────────────────
def compute_customer_segment(spark: SparkSession):
    """
    Assign a segment_type to each customer using rule-based classification.
    """
    logger.info("[Spark] Computing customer segments...")

    customers_df = read_table(spark, "customers").select(
        "customer_id", "age", "employment_type", "industry_sector", "region"
    )

    result = customers_df.withColumn(
        "segment_type",
        F.when(F.col("age") >= 60, F.lit("retiree"))
         .when(F.col("employment_type") == "retired", F.lit("retiree"))
         .when(F.col("industry_sector").isin("Agriculture", "Farming", "Dairy"), F.lit("agricultural"))
         .when(F.col("employment_type").isin("gig_worker", "freelancer", "contract_worker"), F.lit("gig_worker"))
         .when(F.col("employment_type").isin("self_employed", "business_owner", "professional"), F.lit("self_employed"))
         .when(F.col("employment_type").isin("salaried_private", "salaried_govt"), F.lit("salaried"))
         .otherwise(F.lit("salaried"))
    ).select("customer_id", "segment_type")

    logger.info(f"[Spark] Assigned segments for {result.count()} customers")
    return result


def run_batch_pipeline():
    """Run the complete batch feature computation pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Engine - Spark Batch Feature Computation")
    print("  Includes: salary delay, utility delay, spend trends, demographics,")
    print("            FD closures, SIP stoppage, insurance lapse, employer health,")
    print("            customer segment classification")
    print("=" * 70)

    _bank_profile = BankProfileLoader.get_active_profile()  # noqa: F841
    spark = get_spark_session()

    try:
        # ── Original batch features ──
        salary_delay = compute_salary_delay(spark)
        utility_delay = compute_utility_delay(spark)
        spend_trends = compute_spend_trends(spark)
        demographics = compute_demographics(spark)
        spend_stats = compute_monthly_spend_stats(spark)

        # ── M2: Asset-side features ──
        fd_features = compute_fd_premature_withdrawal(spark)
        sip_features = compute_sip_stoppage(spark)
        insurance_features = compute_insurance_lapse(spark)

        # ── M3: Employer health ──
        try:
            from batch_processing.employer_health import EmployerHealthScorer
            employer_scorer = EmployerHealthScorer(spark)
            employer_features = employer_scorer.get_customer_employer_features()
        except Exception as e:
            logger.warning(f"[Spark] Employer health computation skipped: {e}")
            employer_features = None

        # ── M1: Customer segment ──
        segment_features = compute_customer_segment(spark)

        # ── Merge all features ──
        batch_features = demographics
        feature_dfs = [
            salary_delay, utility_delay, spend_trends, spend_stats,
            fd_features, sip_features, insurance_features, segment_features,
        ]
        if employer_features is not None:
            feature_dfs.append(employer_features)

        for df in feature_dfs:
            batch_features = batch_features.join(df, "customer_id", "left")

        # Fill nulls
        batch_features = batch_features.fillna({
            "salary_delay_days": 0,
            "utility_payment_delay_avg": 0.0,
            "discretionary_spend_trend": 1.0,
            "avg_monthly_spend_3m": 0.0,
            "spend_volatility_3m": 0.0,
            "fd_closed_count_90d": 0,
            "fd_closure_amount_90d": 0.0,
            "sip_stopped_flag": False,
            "sip_gaps_3m": 0,
            "insurance_lapse_flag": False,
            "insurance_missed_payments_3m": 0,
            "employer_health_score": 0.0,
            "employer_payroll_delay_avg": 0.0,
            "employer_headcount_change_pct": 0.0,
            # P10: GST invoice-based distress features
            "gst_filing_gap_days": 0,
            "gst_revenue_decline_pct_3m": 0.0,
            "gst_itc_mismatch_flag": False,
            "gst_late_filing_count_6m": 0,
            "gst_nil_return_streak": 0,
            "segment_type": "salaried",
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
