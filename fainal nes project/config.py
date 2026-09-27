"""
Central configuration for the Market Spike/Crash/Neutral prediction project.
Edit this file to change tickers, date range, labeling thresholds, and model settings.
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
# Add/remove tickers here. These are the companies your model will know about.
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Fetch the last N years of daily data
YEARS_OF_HISTORY = 2
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * YEARS_OF_HISTORY)

RAW_DATA_DIR = "data/raw"
FEATURES_DATA_DIR = "data/features"
MODEL_DIR = "models"

# ---------------------------------------------------------------------------
# Labeling thresholds (next-day % return -> label)
# ---------------------------------------------------------------------------
# If next day's close moves up by more than SPIKE_THRESHOLD -> "spike"
# If it moves down by more than CRASH_THRESHOLD -> "crash"
# Otherwise -> "neutral"
# Lowered from 3% -> 1.5% so spike/crash are less rare and the model sees
# enough examples of each (with 3% they were only ~8-9% of the data each).
SPIKE_THRESHOLD = 0.015   # +1.5%
CRASH_THRESHOLD = -0.015  # -1.5%

LABELS = ["crash", "neutral", "spike"]

# ---------------------------------------------------------------------------
# Headline sentiment blending (used at prediction time only -- see sentiment.py)
# ---------------------------------------------------------------------------
# Final probability = SENTIMENT_WEIGHT * sentiment-based distribution
#                    + (1 - SENTIMENT_WEIGHT) * model's price-based distribution
# 0 = ignore headline text entirely (old behavior), 1 = ignore price history entirely.
SENTIMENT_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Feature engineering window sizes (in trading days)
# ---------------------------------------------------------------------------
RETURN_WINDOWS = [1, 3, 5, 10]
VOLATILITY_WINDOW = 10
VOLUME_AVG_WINDOW = 10

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = f"{MODEL_DIR}/market_model.joblib"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
