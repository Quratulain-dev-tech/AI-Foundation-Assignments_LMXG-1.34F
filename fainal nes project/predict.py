"""
Step 6 (runtime): Given a news title, predict spike / crash / neutral.

Flow:
  news title
    -> ner_extract.extract_ticker()          find which company it's about
    -> yfinance                               pull that ticker's latest price data
    -> feature_engineering.add_features_for_ticker()   build the same price features used in training
    -> (if the model was trained with news features) recent scraped news
       for this ticker from data/news/news_history.csv -> news_sentiment_mean, news_count
    -> trained model                          combined price(+news)-based probabilities
    -> sentiment.score_sentiment(title)       THIS specific headline's own sentiment
    -> blend the two (config.SENTIMENT_WEIGHT)  final spike/crash/neutral call

Usage:
    python predict.py "Tesla shares jump on record deliveries"
"""

import os
import sys
import joblib
import pandas as pd
import yfinance as yf

import config
from ner_extract import extract_ticker
from feature_engineering import add_features_for_ticker
from train_model import PRICE_FEATURE_COLUMNS, NEWS_FEATURE_COLUMNS, FEATURES_META_PATH
from sentiment import score_sentiment, sentiment_to_distribution

NEWS_PATH = os.path.join("data/news", "news_history.csv")


def get_recent_data(ticker: str, lookback_days: int = 60) -> pd.DataFrame:
    """Pull enough recent history to compute rolling features for `ticker`."""
    df = yf.download(ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Ticker"] = ticker
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    return df


def get_recent_news_features(ticker: str, lookback_days: int = 7) -> dict:
    """Average sentiment + headline count for `ticker` over the last
    `lookback_days` from the scraped news history (0/0 if none available)."""
    if not os.path.exists(NEWS_PATH):
        return {"news_sentiment_mean": 0.0, "news_count": 0}

    news_df = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    recent = news_df[(news_df["Ticker"] == ticker) & (news_df["Date"] >= cutoff)]

    if recent.empty:
        return {"news_sentiment_mean": 0.0, "news_count": 0}

    return {
        "news_sentiment_mean": round(recent["Sentiment"].mean(), 3),
        "news_count": int(len(recent)),
    }


def blend_probabilities(model_proba: dict, sentiment_dist: dict, weight: float) -> dict:
    """Weighted average of the model's probabilities and the sentiment-based
    distribution, then renormalized so they sum to 1."""
    blended = {
        label: (1 - weight) * model_proba.get(label, 0.0) + weight * sentiment_dist.get(label, 0.0)
        for label in config.LABELS
    }
    total = sum(blended.values()) or 1.0
    return {k: round(v / total, 3) for k, v in blended.items()}


def predict_from_title(title: str):
    ticker = extract_ticker(title)
    if ticker is None:
        return {"title": title, "ticker": None, "prediction": None,
                "note": "Could not identify a known company/ticker in this title."}

    raw = get_recent_data(ticker)
    if raw.empty:
        return {"title": title, "ticker": ticker, "prediction": None,
                "note": "No market data returned for this ticker."}

    featured = add_features_for_ticker(raw)
    latest_row = featured.dropna(subset=PRICE_FEATURE_COLUMNS).iloc[[-1]].copy()

    if latest_row.empty:
        return {"title": title, "ticker": ticker, "prediction": None,
                "note": "Not enough recent history to compute features yet."}

    # Which features does the saved model actually expect? (price-only, or
    # price+news if build_dataset.py + train_model.py were run with news data)
    feature_columns = joblib.load(FEATURES_META_PATH) if os.path.exists(FEATURES_META_PATH) else PRICE_FEATURE_COLUMNS
    news_info = {}
    if set(NEWS_FEATURE_COLUMNS).issubset(feature_columns):
        news_info = get_recent_news_features(ticker)
        for col, val in news_info.items():
            latest_row[col] = val

    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    X = scaler.transform(latest_row[feature_columns])
    model_proba = dict(zip(model.classes_, model.predict_proba(X)[0].round(3)))

    sentiment_score = score_sentiment(title)
    sentiment_dist = sentiment_to_distribution(sentiment_score)

    final_proba = blend_probabilities(model_proba, sentiment_dist, config.SENTIMENT_WEIGHT)
    final_pred = max(final_proba, key=final_proba.get)

    result = {
        "title": title,
        "ticker": ticker,
        "as_of_date": str(latest_row["Date"].values[0])[:10],
        "model_prediction": model.predict(X)[0],
        "model_probabilities": model_proba,
        "headline_sentiment_score": round(sentiment_score, 2),
        "final_prediction": final_pred,
        "final_probabilities": final_proba,
    }
    if news_info:
        result["recent_news_used"] = news_info
    return result


if __name__ == "__main__":
    title = " ".join(sys.argv[1:]) or "Tesla shares jump on record deliveries"
    result = predict_from_title(title)
    print("\n=== Prediction ===")
    for k, v in result.items():
        print(f"{k}: {v}")
