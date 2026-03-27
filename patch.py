import sys
import re

file_path = 'c:/Users/JKP/Barc/PreDelinquencyEngine/batch_processing/spark_jobs.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

safe_bad = re.compile(r'from pyspark\.sql\.types import \(\s+"user": PostgresConfig\.USER,\s+"password": PostgresConfig\.PASSWORD,\s+"driver": "org\.postgresql\.Driver",\s+\}')

good_content = """from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DecimalType, TimestampType, BooleanType, FloatType,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import SparkConfig, PostgresConfig  # pyre-ignore[21]
from config.bank_config import BankProfileLoader  # pyre-ignore[21]

logger = logging.getLogger(__name__)

JDBC_URL = f"jdbc:postgresql://{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
JDBC_PROPS = {"""

new_content = safe_bad.sub(good_content, content, count=1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
print("File patched successfully.")
