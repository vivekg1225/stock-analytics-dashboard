
#  spark_batch.py  —  Batch analytics + ML prediction on CSV data
#
#  Run this SEPARATELY from spark_stream.py, in its own terminal.
#  It reads stock_data.csv (written by spark_stream.py), performs
#  batch computations, and writes batch_results.json for Flask.
#
#  Run command:
#    python spark_batch.py
#
#  Computations:
#    1. Descriptive stats    : mean, stddev, min, max, skewness,
#                              kurtosis, P25 / P50 / P75 / P95
#    2. Linear Regression    : predict next price using pyspark.ml
#       (price prediction)     trained per stock on tick index
#    3. Bollinger Bands      : upper / lower price bounds
#                              (mean ± 2×stddev)
#    4. Trend classification : slope of last 20 ticks per stock
#                              → UPTREND / DOWNTREND / SIDEWAYS
#    5. Correlation matrix   : pairwise price correlation between
#                              all stocks (AAPL↔GOOG, etc.)
#    6. Session summary      : total ticks, price range, CV per stock
#
#  Output:
#    → batch_results.json    (read by Flask /api/batch endpoint)


import os
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, stddev, min, max, count,
    skewness, kurtosis, percentile_approx,
    monotonically_increasing_id
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature    import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml            import Pipeline

# Spark Session
spark = SparkSession.builder \
    .appName("StockBatchAnalytics") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# File paths 
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_FILE      = os.path.join(BASE_DIR, "stock_data.csv")
OUTPUT_FILE   = os.path.join(BASE_DIR, "batch_results.json")
RUN_INTERVAL  = 10   # seconds between each batch run

# Helper: safe Python scalar from any numeric 
def s(v, decimals=4):
    if v is None: return 0.0
    if hasattr(v, "item"): return round(float(v.item()), decimals)
    return round(float(v), decimals)


#  1. DESCRIPTIVE STATISTICS

def compute_descriptive(stock_df, stock_name):
    """Full descriptive stats for one stock's price series."""
    row = stock_df.agg(
        avg("price")                          .alias("mean"),
        stddev("price")                       .alias("std"),
        min("price")                          .alias("min"),
        max("price")                          .alias("max"),
        count("price")                        .alias("count"),
        skewness("price")                     .alias("skewness"),
        kurtosis("price")                     .alias("kurtosis"),
        percentile_approx("price", 0.25)      .alias("p25"),
        percentile_approx("price", 0.50)      .alias("p50"),
        percentile_approx("price", 0.75)      .alias("p75"),
        percentile_approx("price", 0.95)      .alias("p95"),
    ).collect()[0]

    mean = s(row["mean"])
    std  = s(row["std"])
    cv   = round((std / mean * 100), 4) if mean > 0 else 0.0

    return {
        "stock"      : stock_name,
        "mean"       : mean,
        "std"        : std,
        "min"        : s(row["min"]),
        "max"        : s(row["max"]),
        "count"      : int(row["count"]),
        "skewness"   : s(row["skewness"]),
        "kurtosis"   : s(row["kurtosis"]),
        "p25"        : s(row["p25"]),
        "p50"        : s(row["p50"]),
        "p75"        : s(row["p75"]),
        "p95"        : s(row["p95"]),
        "cv"         : cv,
        "price_range": round(s(row["max"]) - s(row["min"]), 4),
    }


#  2. LINEAR REGRESSION  —  predict next price

def predict_next_price(stock_df, stock_name):
    """
    Trains a LinearRegression model on (tick_index → price).
    Uses the model to predict the next 5 price values.
    Also computes prediction confidence as R² score.
    """
    # Need at least 10 rows to train
    total = stock_df.count()
    if total < 10:
        return {
            "stock"          : stock_name,
            "predicted_next" : 0.0,
            "predictions"    : [],
            "r2"             : 0.0,
            "rmse"           : 0.0,
            "model_status"   : "insufficient_data",
            "data_points"    : total
        }

    # Add sequential index as the feature
    indexed_df = stock_df.withColumn("tick_index",
        monotonically_increasing_id().cast(DoubleType())
    )

    assembler = VectorAssembler(
        inputCols=["tick_index"],
        outputCol="features"
    )

    lr = LinearRegression(
        featuresCol="features",
        labelCol="price",
        maxIter=20,
        regParam=0.01
    )

    pipeline = Pipeline(stages=[assembler, lr])
    model    = pipeline.fit(indexed_df)

    # Training metrics
    lr_model = model.stages[-1]
    summary  = lr_model.summary
    r2   = round(float(summary.r2),   4)
    rmse = round(float(summary.rootMeanSquaredError), 4)

    # Predict next 5 ticks
    last_index = indexed_df.agg(
        max("tick_index").alias("max_idx")
    ).collect()[0]["max_idx"]

    predictions = []
    for i in range(1, 6):
        next_idx = float(last_index) + i
        slope     = float(lr_model.coefficients[0])
        intercept = float(lr_model.intercept)
        pred_price = round(slope * next_idx + intercept, 4)
        predictions.append(pred_price)

    return {
        "stock"          : stock_name,
        "predicted_next" : predictions[0],       # very next price
        "predictions"    : predictions,           # next 5 prices
        "r2"             : r2,
        "rmse"           : rmse,
        "model_status"   : "trained",
        "data_points"    : total
    }


#  3. BOLLINGER BANDS

def compute_bollinger(stock_df, stock_name):
    """
    Bollinger Bands: upper = mean + 2σ, lower = mean - 2σ
    Also flags if current price is outside the bands (breakout).
    """
    row = stock_df.agg(
        avg("price")   .alias("mean"),
        stddev("price").alias("std"),
        max(col("timestamp")).alias("last_ts")   # latest tick
    ).collect()[0]

    mean = s(row["mean"])
    std  = s(row["std"])
    upper = round(mean + 2 * std, 4)
    lower = round(mean - 2 * std, 4)
    band_width = round(upper - lower, 4)

    # Get current (latest) price
    latest_price = s(
        stock_df.orderBy(col("timestamp").desc())
                .select("price")
                .first()["price"]
    )

    # Breakout detection
    if latest_price > upper:
        signal = "BREAKOUT_UP"
    elif latest_price < lower:
        signal = "BREAKOUT_DOWN"
    else:
        signal = "WITHIN_BANDS"

    return {
        "stock"        : stock_name,
        "upper_band"   : upper,
        "lower_band"   : lower,
        "middle_band"  : mean,
        "band_width"   : band_width,
        "current_price": latest_price,
        "signal"       : signal,
    }


#  4. TREND CLASSIFICATION  (slope of last 20 ticks)

def compute_trend(stock_df, stock_name):
    """
    Fits a line through the last 20 price ticks.
    Slope > 0.05  → UPTREND
    Slope < -0.05 → DOWNTREND
    Otherwise     → SIDEWAYS
    """
    # Take last 20 ticks ordered by timestamp
    recent = stock_df.orderBy(col("timestamp").desc()).limit(20)
    recent_count = recent.count()

    if recent_count < 5:
        return { "stock": stock_name, "trend": "SIDEWAYS",
                 "slope": 0.0, "recent_ticks": recent_count }

    # Use pandas for slope calculation (small dataset, safe here)
    pdf = recent.toPandas().sort_values("timestamp").reset_index(drop=True)
    pdf["idx"] = range(len(pdf))

    # Simple linear regression: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    x_mean = pdf["idx"].mean()
    y_mean = pdf["price"].mean()
    numerator   = ((pdf["idx"] - x_mean) * (pdf["price"] - y_mean)).sum()
    denominator = ((pdf["idx"] - x_mean) ** 2).sum()
    slope = round(numerator / denominator, 6) if denominator != 0 else 0.0

    if   slope >  0.05: trend = "UPTREND"
    elif slope < -0.05: trend = "DOWNTREND"
    else:               trend = "SIDEWAYS"

    # Percentage change across those recent ticks
    first_price = round(float(pdf["price"].iloc[0]),  4)
    last_price  = round(float(pdf["price"].iloc[-1]), 4)
    pct_change  = round(((last_price - first_price) / first_price) * 100, 4) \
                  if first_price > 0 else 0.0

    return {
        "stock"        : stock_name,
        "trend"        : trend,
        "slope"        : slope,
        "recent_ticks" : recent_count,
        "first_price"  : first_price,
        "last_price"   : last_price,
        "pct_change"   : pct_change,
    }


#  5. CORRELATION MATRIX

def compute_correlation(full_df, stocks):
    """
    Computes pairwise Pearson correlation between stock prices.
    Aligns by timestamp bucket (5-second windows) so prices are
    comparable across stocks.
    """
    if len(stocks) < 2:
        return {}

    # Bucket timestamps into 5-second windows for alignment
    bucketed = full_df.withColumn(
        "time_bucket",
        (col("timestamp") / 5).cast("long")
    )

    # Pivot: one row per time bucket, columns = stock prices
    pivoted = bucketed.groupBy("time_bucket").pivot("stock").avg("price")
    pivoted = pivoted.dropna()

    row_count = pivoted.count()
    if row_count < 5:
        return { "status": "insufficient_overlap", "pairs": [] }

    # Compute correlation for each pair
    pairs = []
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            s1, s2 = stocks[i], stocks[j]
            # Only compute if both columns exist
            cols = pivoted.columns
            if s1 not in cols or s2 not in cols:
                continue
            try:
                corr_val = pivoted.stat.corr(s1, s2)
                pairs.append({
                    "pair"       : f"{s1}↔{s2}",
                    "stock_a"    : s1,
                    "stock_b"    : s2,
                    "correlation": round(corr_val, 4),
                    "strength"   : "STRONG"   if abs(corr_val) > 0.7
                                 else "MODERATE" if abs(corr_val) > 0.4
                                 else "WEAK",
                    "direction"  : "POSITIVE" if corr_val > 0 else "NEGATIVE"
                })
            except Exception:
                pass

    return { "status": "computed", "pairs": pairs, "rows_used": row_count }


#  MAIN BATCH RUN

def run_batch():
    """Reads CSV, runs all computations, writes batch_results.json."""

    if not os.path.exists(CSV_FILE):
        print(f"[Batch] CSV not found: {CSV_FILE}")
        print("[Batch] Make sure spark_stream.py is running first.")
        return False

    # Load CSV into Spark DataFrame
    full_df = spark.read.csv(CSV_FILE, header=True, inferSchema=True)
    total_rows = full_df.count()

    if total_rows < 5:
        print(f"[Batch] Only {total_rows} rows in CSV. Waiting for more data...")
        return False

    full_df.cache()   # cache since we query it many times
    stocks = [r["stock"] for r in
              full_df.select("stock").distinct().collect()]
    stocks.sort()

    print(f"\n[Batch] ── Running batch on {total_rows} rows, {len(stocks)} stocks ──")

    results = {
        "timestamp"   : round(time.time(), 2),
        "total_rows"  : total_rows,
        "stocks"      : stocks,
        "descriptive" : [],
        "predictions" : [],
        "bollinger"   : [],
        "trends"      : [],
        "correlation" : {},
    }

    for stock in stocks:
        stock_df = full_df.filter(col("stock") == stock).cache()
        tick_count = stock_df.count()
        print(f"  [{stock}] {tick_count} ticks")

        # 1. Descriptive stats
        desc = compute_descriptive(stock_df, stock)
        results["descriptive"].append(desc)
        print(f"    mean=${desc['mean']:.2f}  std={desc['std']:.3f}  "
              f"CV={desc['cv']:.2f}%  skew={desc['skewness']:.3f}")

        # 2. Linear Regression prediction
        pred = predict_next_price(stock_df, stock)
        results["predictions"].append(pred)
        if pred["model_status"] == "trained":
            print(f"    predicted_next=${pred['predicted_next']:.2f}  "
                  f"R²={pred['r2']:.4f}  RMSE={pred['rmse']:.4f}")
        else:
            print(f"    prediction: {pred['model_status']}")

        # 3. Bollinger Bands
        boll = compute_bollinger(stock_df, stock)
        results["bollinger"].append(boll)
        print(f"    Bollinger: [{boll['lower_band']:.2f} — {boll['upper_band']:.2f}]  "
              f"signal={boll['signal']}")

        # 4. Trend
        trend = compute_trend(stock_df, stock)
        results["trends"].append(trend)
        print(f"    Trend: {trend['trend']}  slope={trend['slope']:.5f}  "
              f"pct_change={trend['pct_change']:.2f}%")

        stock_df.unpersist()

    # 5. Correlation matrix
    results["correlation"] = compute_correlation(full_df, stocks)
    if results["correlation"].get("pairs"):
        for p in results["correlation"]["pairs"]:
            print(f"  Correlation {p['pair']}: {p['correlation']:.4f} "
                  f"({p['strength']} {p['direction']})")

    full_df.unpersist()

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Batch] Results written → {OUTPUT_FILE}")
    return True


#  ENTRY POINT  —  runs batch every RUN_INTERVAL seconds

if __name__ == "__main__":
    print(f"[Batch] spark_batch.py started.")
    print(f"  Reading from : {CSV_FILE}")
    print(f"  Writing to   : {OUTPUT_FILE}")
    print(f"  Interval     : every {RUN_INTERVAL}s\n")

    while True:
        try:
            run_batch()
        except Exception as e:
            print(f"[Batch] ERROR: {e}")
        print(f"[Batch] Next run in {RUN_INTERVAL}s...\n")
        time.sleep(RUN_INTERVAL)