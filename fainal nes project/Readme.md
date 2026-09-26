# 📈 Market Spike / Crash Predictor

A news-driven stock movement predictor — give it a headline, and it predicts whether that company's stock is about to **spike 🚀**, **crash 📉**, or stay **neutral 😐** — using both price history and real scraped news sentiment together.

Covers: **AAPL · TSLA · MSFT · GOOGL · AMZN · NVDA**

---

## 🧠 How it works (pipeline)

| Step | File | What it does |
|------|------|---------------|
| 1️⃣ | `data_fetch.py` | Fetches 2 years of daily price data (OHLCV) for each ticker |
| 2️⃣ | `feature_engineering.py` | Builds features — returns, volatility, volume ratio, gap % |
| 3️⃣ | `labeling.py` | Labels each day as `spike` / `crash` / `neutral` |
| 4️⃣ | `ner_extract.py` | Finds which company a headline is talking about |
| 5️⃣ | `sentiment.py` | Scores a headline's sentiment (positive/negative) |
| 6️⃣ | `news_scrape.py` | Scrapes real headlines (yfinance + Google News RSS) |
| 7️⃣ | `build_dataset.py` | Merges price features + news sentiment into one dataset |
| 8️⃣ | `train_model.py` | Trains a Random Forest classifier on everything |
| 9️⃣ | `predict.py` | Takes a headline → predicts spike/crash/neutral 🎯 |

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## ▶️ Run the full pipeline

```bash
python data_fetch.py
python feature_engineering.py
python labeling.py
python news_scrape.py
python build_dataset.py
python train_model.py
python predict.py "Tesla shares jump on record deliveries"
```

---

## 📊 Results

**Overall Accuracy: 51%** on the held-out test set (588 samples) 🎯

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 🟢 Neutral | 0.70 | 0.70 | 0.70 |
| 🚀 Spike | 0.25 | 0.28 | 0.26 |
| 📉 Crash | 0.22 | 0.19 | 0.20 |

**Macro F1: 0.39**

### 🔍 Confusion Matrix (rows = actual, cols = predicted)

|            | Pred: Crash | Pred: Neutral | Pred: Spike |
|------------|:---:|:---:|:---:|
| **Actual: Crash**   | 21  | 42  | 48 |
| **Actual: Neutral** | 40  | 242 | 64 |
| **Actual: Spike**   | 33  | 61  | 37 |

### ⭐ Top Features (by importance)

1. `hl_range_pct` — 0.173
2. `volatility` — 0.167
3. `gap_pct` — 0.113
4. `volume_ratio` — 0.112
5. `return_1d` — 0.112

> 💡 **Note:** News sentiment features (`news_sentiment_mean`, `news_count`) currently contribute almost nothing to the model, since free news sources only expose a short recent window of headlines — not 2 years' worth. As `news_history.csv` keeps growing (run `news_scrape.py` regularly), the news signal should get stronger over time.

---

## 🔁 Keeping it up to date

- Re-run `news_scrape.py` regularly (daily is ideal) to keep building real dated headline coverage
- After scraping more news, re-run `build_dataset.py` → `train_model.py` to retrain

## 🛠️ Customize

- Add more companies → edit `TICKERS` in `config.py`
- Change spike/crash sensitivity → `SPIKE_THRESHOLD` / `CRASH_THRESHOLD` in `config.py`
- Change how much a headline's own wording affects the final call → `SENTIMENT_WEIGHT` in `config.py`

---

## 📝 Notes

- Labels are based on the **next trading day's** return, so the model predicts forward — no data leakage
- Class weights are balanced automatically since "neutral" days are far more common
- Two independent news sources are combined: `yfinance`'s feed + real Google News RSS scraping (no API key needed)

---

✨ *Built as part of an AI/ML foundational program project.*
