"""
Employer Health Scorer
Computes employer-level payroll stability metrics as a leading indicator
for employee financial stress. Fires 4-8 weeks before salary_delay_days.
"""
import logging
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)

JDBC_URL = f"jdbc:postgresql://{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
JDBC_PROPS = {
    "user": PostgresConfig.USER,
    "password": PostgresConfig.PASSWORD,
    "driver": "org.postgresql.Driver",
}


class EmployerHealthScorer:
    """Computes employer payroll stability as a stress leading indicator."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def _read_table(self, table_name: str) -> DataFrame:
        return self.spark.read.jdbc(JDBC_URL, table_name, properties=JDBC_PROPS)

    def compute_employer_payroll_stability(self) -> DataFrame:
        """
        Compute payroll stability metrics per employer:
        - employer_payroll_delay_avg: average salary delay across all employees
        - employer_headcount_change_pct: month-over-month change in # employees paid
        """
        logger.info("[EmployerHealth] Computing employer payroll stability...")

        customers_df = self._read_table("customers").select(
            "customer_id", "employer_name", "salary_credit_day"
        ).filter(F.col("employer_name").isNotNull())

        # Get salary credits in last 90 days
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        txns_df = self._read_table("transactions").filter(
            (F.col("txn_type") == "salary_credit") &
            (F.col("status") == "success") &
            (F.col("timestamp") >= three_months_ago)
        ).select("customer_id", "amount", "timestamp")

        # Join to get employer name
        joined = txns_df.join(customers_df, "customer_id", "inner")

        # Compute delay per salary payment
        joined = joined.withColumn(
            "credit_day", F.dayofmonth("timestamp")
        ).withColumn(
            "credit_month", F.date_format("timestamp", "yyyy-MM")
        ).withColumn(
            "raw_delay", F.col("credit_day") - F.col("salary_credit_day")
        ).withColumn(
            "delay_days",
            F.when(F.col("raw_delay") < -15, F.col("raw_delay") + 30)
             .otherwise(F.col("raw_delay"))
        )

        # Employer-level avg delay
        employer_delay = joined.groupBy("employer_name").agg(
            F.round(F.avg("delay_days"), 1).alias("employer_payroll_delay_avg"),
            F.countDistinct("customer_id").alias("current_headcount"),
        )

        # Headcount change: compare current month vs 2 months ago
        current_month = datetime.now().strftime("%Y-%m")
        two_months_ago_str = (datetime.now() - timedelta(days=60)).strftime("%Y-%m")

        current_hc = joined.filter(
            F.col("credit_month") == current_month
        ).groupBy("employer_name").agg(
            F.countDistinct("customer_id").alias("hc_current")
        )

        prev_hc = joined.filter(
            F.col("credit_month") == two_months_ago_str
        ).groupBy("employer_name").agg(
            F.countDistinct("customer_id").alias("hc_prev")
        )

        hc_change = current_hc.join(prev_hc, "employer_name", "outer").fillna(0)
        hc_change = hc_change.withColumn(
            "employer_headcount_change_pct",
            F.when(F.col("hc_prev") > 0,
                   F.round((F.col("hc_current") - F.col("hc_prev")) / F.col("hc_prev") * 100, 1))
             .otherwise(0.0)
        )

        # Merge delay + headcount into employer health score
        result = employer_delay.join(
            hc_change.select("employer_name", "employer_headcount_change_pct"),
            "employer_name", "left"
        ).fillna({"employer_headcount_change_pct": 0.0})

        # Compute composite health score (0-1, higher = riskier)
        # delay > 5 days and shrinking headcount both contribute
        result = result.withColumn(
            "employer_health_score",
            F.least(
                F.lit(1.0),
                F.greatest(
                    F.lit(0.0),
                    (F.col("employer_payroll_delay_avg") / 15.0) * 0.6 +
                    F.when(F.col("employer_headcount_change_pct") < -10, 0.4)
                     .when(F.col("employer_headcount_change_pct") < 0, 0.2)
                     .otherwise(0.0)
                )
            )
        )

        logger.info(f"[EmployerHealth] Computed health scores for {result.count()} employers")
        return result

    def get_customer_employer_features(self) -> DataFrame:
        """Join employer health scores back to individual customers."""
        employer_health = self.compute_employer_payroll_stability()

        customers_df = self._read_table("customers").select(
            "customer_id", "employer_name"
        ).filter(F.col("employer_name").isNotNull())

        result = customers_df.join(
            employer_health.select(
                "employer_name", "employer_health_score",
                "employer_payroll_delay_avg", "employer_headcount_change_pct"
            ),
            "employer_name", "left"
        ).fillna({
            "employer_health_score": 0.0,
            "employer_payroll_delay_avg": 0.0,
            "employer_headcount_change_pct": 0.0,
        }).select(
            "customer_id", "employer_health_score",
            "employer_payroll_delay_avg", "employer_headcount_change_pct"
        )

        return result
