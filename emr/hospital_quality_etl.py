"""
Healthcare Hospital Quality ETL
Reads CMS Hospital General Information + Timely & Effective Care from S3 raw,
cleans/validates, joins, and writes curated Parquet.

Column names below are confirmed from the actual CMS downloads (Part 7):

Hospital_General_Information.csv:
  Facility ID, Facility Name, Address, City/Town, State, ZIP Code,
  County/Parish, Telephone Number, Hospital Type, Hospital Ownership,
  Emergency Services, Meets criteria for birthing friendly designation,
  Hospital overall rating, Hospital overall rating footnote, ... (MORT/
  Safety/READM/Pt Exp/TE group measure-count columns, unused here)

Timely_and_Effective_Care.csv:
  Facility ID, Facility Name, Address, City/Town, State, ZIP Code,
  County/Parish, Telephone Number, Condition, Measure ID, Measure Name,
  Score, Sample, Footnote, Start Date, End Date
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    upper,
    count,
    avg,
    lit,
    current_timestamp,
)

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("HealthcareHospitalQualityETL")
    .getOrCreate()
)

# ---------------------------------------------------------------------------
# S3 paths
# ---------------------------------------------------------------------------
BUCKET = "healthcare-de-project-20260911-hwz967"

HOSPITAL_INPUT = f"s3://{BUCKET}/raw/cms/hospital_general/"
QUALITY_INPUT = f"s3://{BUCKET}/raw/cms/timely_effective/"

HOSPITAL_OUTPUT = f"s3://{BUCKET}/curated/hospital/"
QUALITY_OUTPUT = f"s3://{BUCKET}/curated/quality_measure/"
SUMMARY_OUTPUT = f"s3://{BUCKET}/curated/hospital_quality_summary/"

# ---------------------------------------------------------------------------
# Read + inspect (Parts 12-14) — run this first, before writing any
# transformation logic below. Confirm real column names from printSchema().
# ---------------------------------------------------------------------------
hospital_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(HOSPITAL_INPUT)
)

hospital_df.printSchema()
hospital_df.show(5, truncate=False)

quality_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(QUALITY_INPUT)
)

quality_df.printSchema()
quality_df.show(5, truncate=False)

# ---------------------------------------------------------------------------
# Part 15 — select + rename hospital columns
# ---------------------------------------------------------------------------
hospital_df = hospital_df.select(
    col("Facility ID").alias("facility_id"),
    col("Facility Name").alias("facility_name"),
    col("City/Town").alias("city"),
    col("State").alias("state"),
    col("ZIP Code").alias("zip_code"),
    col("County/Parish").alias("county"),
    col("Hospital Type").alias("hospital_type"),
    col("Hospital Ownership").alias("hospital_ownership"),
    col("Emergency Services").alias("emergency_services"),
    col("Hospital overall rating").alias("overall_rating"),
)

# ---------------------------------------------------------------------------
# Part 16 — standardize text
# ---------------------------------------------------------------------------
hospital_df = (
    hospital_df
    .withColumn("facility_id", trim(col("facility_id")))
    .withColumn("facility_name", trim(col("facility_name")))
    .withColumn("city", trim(col("city")))
    .withColumn("state", upper(trim(col("state"))))
    .withColumn("county", trim(col("county")))
    .withColumn("hospital_type", trim(col("hospital_type")))
    .withColumn("hospital_ownership", trim(col("hospital_ownership")))
)

# ---------------------------------------------------------------------------
# Part 17 — check NULL facility IDs
# ---------------------------------------------------------------------------
null_facility_count = hospital_df.filter(col("facility_id").isNull()).count()
print(f"NULL facility IDs: {null_facility_count}")

# ---------------------------------------------------------------------------
# Part 18 — check duplicate facility IDs
# ---------------------------------------------------------------------------
duplicate_facilities = (
    hospital_df
    .groupBy("facility_id")
    .count()
    .filter(col("count") > 1)
)
duplicate_count = duplicate_facilities.count()
print(f"Duplicate facility IDs: {duplicate_count}")

# ---------------------------------------------------------------------------
# Part 19 — check hospital ratings
# "Hospital overall rating" is textual in the CMS source (can contain
# "Not Available"), so cast to integer and let non-numeric values become NULL
# rather than raise on read.
# ---------------------------------------------------------------------------
hospital_df = hospital_df.withColumn(
    "overall_rating_numeric",
    col("overall_rating").cast("integer"),
)

invalid_rating_count = (
    hospital_df
    .filter(
        col("overall_rating_numeric").isNotNull()
        & (
            (col("overall_rating_numeric") < 1)
            | (col("overall_rating_numeric") > 5)
        )
    )
    .count()
)
print(f"Invalid ratings: {invalid_rating_count}")

# ---------------------------------------------------------------------------
# Part 20 — split rejected vs valid hospitals
# ---------------------------------------------------------------------------
rejected_hospitals = hospital_df.filter(col("facility_id").isNull())

rejected_hospitals.write.mode("overwrite").parquet(
    f"s3://{BUCKET}/rejected/hospital/"
)

valid_hospitals = hospital_df.filter(col("facility_id").isNotNull())

# ---------------------------------------------------------------------------
# Part 21 — select + rename quality columns
# ---------------------------------------------------------------------------
quality_df = quality_df.select(
    col("Facility ID").alias("facility_id"),
    col("Measure ID").alias("measure_id"),
    col("Measure Name").alias("measure_name"),
    col("Score").alias("score"),
    col("Sample").alias("sample"),
)

quality_df = (
    quality_df
    .withColumn("facility_id", trim(col("facility_id")))
    .withColumn("measure_id", trim(col("measure_id")))
)

# ---------------------------------------------------------------------------
# Part 22 — don't blindly convert scores; "Score" mixes numeric values with
# text like "Not Available" — cast to double and keep the original alongside.
# ---------------------------------------------------------------------------
quality_df.select("score").distinct().show(100, False)

quality_df = quality_df.withColumn(
    "numeric_score",
    col("score").cast("double"),
)

# ---------------------------------------------------------------------------
# Part 23 — join (sanity check only; not written on its own — see summary)
# ---------------------------------------------------------------------------
joined_df = valid_hospitals.join(quality_df, on="facility_id", how="left")
joined_df.show(10, truncate=False)

# ---------------------------------------------------------------------------
# Part 24 — hospital-level quality summary
# ---------------------------------------------------------------------------
quality_summary = (
    quality_df
    .groupBy("facility_id")
    .agg(
        count("*").alias("quality_measure_count"),
        count("numeric_score").alias("numeric_measure_count"),
        avg("numeric_score").alias("average_numeric_score"),
    )
)

hospital_summary = valid_hospitals.join(
    quality_summary, on="facility_id", how="left"
)

# ---------------------------------------------------------------------------
# Part 25 — engineering metadata / lineage
# ---------------------------------------------------------------------------
hospital_summary = (
    hospital_summary
    .withColumn("source_system", lit("CMS"))
    .withColumn("etl_load_timestamp", current_timestamp())
)

# ---------------------------------------------------------------------------
# Part 26 — write curated Parquet
# ---------------------------------------------------------------------------
valid_hospitals.write.mode("overwrite").parquet(HOSPITAL_OUTPUT)
quality_df.write.mode("overwrite").parquet(QUALITY_OUTPUT)
hospital_summary.write.mode("overwrite").parquet(SUMMARY_OUTPUT)

# ---------------------------------------------------------------------------
# Part 27 — basic logging
# ---------------------------------------------------------------------------
print(f"Hospital records: {valid_hospitals.count()}")
print(f"Quality records: {quality_df.count()}")
print(f"Summary records: {hospital_summary.count()}")
print("Healthcare ETL completed successfully.")

spark.stop()
