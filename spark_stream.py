# ─────────────────────────────────────────────────────────────────
#  spark_stream.py  —  Real-time Kafka → Spark Streaming pipeline
#
#  Computations per batch:
#    1. Basic stats      : avg, min, max, count, stddev, last price
#    2. Coefficient of   : (stddev / avg) * 100  — normalized volatility
#       Variation (CV)
#    3. Z-score          : (current_price - avg) / stddev
#                          flags anomaly if |z| > 2.0
#    4. Price momentum   : current_price - avg  (positive = above avg)
#    5. Trend label      : UPTREND / DOWNTREND / SIDEWAYS
#                          based on momentum threshold
#
#  Outputs:
#    → results.json      (live stats, read by Flask every 2s)
#    → stock_data.csv    (every raw tick appended, read by spark_batch.py)
# ─────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, avg, max, min, count, stddev, last
)
from pyspark.sql.types import StructType, StringType, DoubleType
import json
import os
import csv

# Spark Session 
spark = SparkSession.builder \
    .appName("StockAnalytics") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Schema for incoming Kafka JSON messages
schema = StructType() \
    .add("stock",     StringType()) \
    .add("price",     DoubleType()) \
    .add("timestamp", DoubleType())

# Read stream from Kafka
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "stock-data") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON value 
parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Aggregations (over all data seen since stream started) 
aggregated = parsed_df.groupBy("stock").agg(
    avg("price")   .alias("avg_price"),
    max("price")   .alias("max_price"),
    min("price")   .alias("min_price"),
    count("price") .alias("total_ticks"),
    stddev("price").alias("volatility"),       # standard deviation
    last("price")  .alias("current_price")
)

# File paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "results.json")
CSV_FILE     = os.path.join(BASE_DIR, "stock_data.csv")

# Create CSV with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stock", "price", "timestamp"])
    print(f"[Spark] Created CSV: {CSV_FILE}")

# Helper: safe float rounding 
def safe_float(value, decimals=4):
    if value is None:
        return 0.0
    if hasattr(value, "item"):          # numpy scalar
        return round(float(value.item()), decimals)
    return round(float(value), decimals)

# Helper: compute derived metrics on a row dict
def enrich_row(row):
    avg_p  = safe_float(row.get("avg_price"))
    curr   = safe_float(row.get("current_price"))
    vol    = safe_float(row.get("volatility"))

    # 1. Coefficient of Variation  (volatility as % of mean)
    cv = round((vol / avg_p * 100), 4) if avg_p > 0 else 0.0

    # 2. Z-score  — how many std-devs is current price from mean
    z_score = round((curr - avg_p) / vol, 4) if vol > 0 else 0.0

    # 3. Anomaly flag  — True if |z| > 2.0
    is_anomaly = abs(z_score) > 2.0

    # 4. Momentum  — current price minus session average
    momentum = round(curr - avg_p, 4)

    # 5. Trend label
    if   momentum >  1.0:  trend = "UPTREND"
    elif momentum < -1.0:  trend = "DOWNTREND"
    else:                  trend = "SIDEWAYS"

    row["coeff_variation"] = cv
    row["z_score"]         = z_score
    row["is_anomaly"]      = is_anomaly
    row["momentum"]        = momentum
    row["trend"]           = trend
    return row

# Raw tick writer — writes each parsed row to CSV 
def write_raw_ticks(df, epoch_id):
    """
    Writes every individual raw price tick to stock_data.csv.
    This builds the historical dataset that spark_batch.py reads.
    """
    rows = df.select("stock", "price", "timestamp").toPandas()
    if rows.empty:
        return
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for _, r in rows.iterrows():
            writer.writerow([
                r["stock"],
                round(float(r["price"]), 4),
                round(float(r["timestamp"]), 4)
            ])
    print(f"[Spark] Batch {epoch_id} — appended {len(rows)} ticks to CSV")

# Aggregated results writer — writes enriched stats to JSON 
def write_aggregated(df, epoch_id):
    """
    Writes per-stock enriched analytics to results.json.
    Flask reads this file and pushes it to the browser.
    """
    rows = df.toPandas().to_dict(orient="records")

    # Normalise all numeric types first
    cleaned = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if v is None:
                clean[k] = 0
            elif hasattr(v, "item"):
                clean[k] = round(float(v.item()), 4)
            elif isinstance(v, float):
                clean[k] = round(v, 4)
            else:
                clean[k] = v
        cleaned.append(enrich_row(clean))

    with open(RESULTS_FILE, "w") as f:
        json.dump(cleaned, f)

    # Console summary for debugging
    print(f"\n[Spark] -- Batch {epoch_id} ------------------------")
    for r in cleaned:
        anomaly_flag = "ANOMALY" if r["is_anomaly"] else ""
        print(
            f"  {r['stock']:<5} | price=${r['current_price']:<8.2f} "
            f"| avg=${r['avg_price']:<8.2f} | trend={r['trend']:<10} "
            f"| z={r['z_score']:<6.2f} | CV={r['coeff_variation']:.2f}%"
            f"{anomaly_flag}"
        )

# Stream 1: raw ticks → CSV (append mode) \
raw_query = parsed_df.writeStream \
    .outputMode("append") \
    .foreachBatch(write_raw_ticks) \
    .trigger(processingTime="3 seconds") \
    .start()

# Stream 2: aggregated stats → results.json (complete mode) 
agg_query = aggregated.writeStream \
    .outputMode("complete") \
    .foreachBatch(write_aggregated) \
    .trigger(processingTime="3 seconds") \
    .start()

print(f"[Spark] Streaming started.")
print(f"  Live stats  -> {RESULTS_FILE}")
print(f"  Raw ticks   -> {CSV_FILE}")
print(f"  Both queries running. Waiting for Kafka data...\n")

# Keep both queries alive
spark.streams.awaitAnyTermination()