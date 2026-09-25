# Market Spike / Crash / Neutral Predictor

A news-title-based stock movement prediction system.
The system takes a news headline, identifies the company mentioned in it, and predicts whether the related stock is expected to **spike, crash, or remain neutral**.

The prediction uses two types of information:

* Historical stock price data
* Real scraped news and sentiment information

## Project Workflow

The complete project follows the pipeline below:

**Data Collection → Feature Engineering → Label Creation → Company/Ticker Identification → Sentiment Analysis → News Collection → Dataset Construction → Model Training → Prediction**

### 1. `data_fetch.py` — Historical Market Data

This script collects approximately **two years of historical OHLCV data** for every ticker specified in `config.py`.

The downloaded information includes date/time and market values and is obtained through `yfinance`.

### 2. `feature_engineering.py` — Creating Market Features

This step converts raw market data into useful numerical features.

The generated features include:

* Daily and multi-day returns
* Rolling volatility
* High-low price range percentage
* Volume ratio for detecting unusual volume activity
* Overnight gap percentage

These features help the model understand recent price behavior and market conditions.

### 3. `labeling.py` — Creating Prediction Classes

The target variable is created from the **next trading day's return**.

Each record is assigned one of three classes:

* **Spike**
* **Crash**
* **Neutral**

The return threshold can be changed easily, and alternative labeling methods can also be implemented.

### 4. `ner_extract.py` — Identifying the Company

When a news headline is provided, this component determines which company the headline is referring to.

It first uses predefined company aliases and then uses **spaCy NER** as a fallback method.

After identifying the company, it maps the company name to its corresponding stock ticker.

### 5. `sentiment.py` — Headline Sentiment

This module calculates the sentiment of the individual news headline.

It uses a keyword-based sentiment approach and is mainly applied to the specific headline entered during prediction.

### 6. `news_scrape.py` — Collecting Real News

News headlines are collected for each ticker from two different sources:

1. `yfinance` built-in news feed
2. Google News RSS through real web scraping

The scraped headlines are stored in:

`data/news/news_history.csv`

No API key is required for the Google News RSS scraping component.

### 7. `build_dataset.py` — Combining Price and News Data

This step combines the historical market features with the collected news information.

The datasets are matched using:

`(Ticker, Date)`

Two additional news-based features are created:

* `news_sentiment_mean`
* `news_count`

The resulting dataset is saved as:

`data/features/labeled_features_with_news.csv`

### 8. `train_model.py` — Model Training

The model uses a **Random Forest classifier**.

When the news-enhanced dataset is available, the model learns from:

* Historical price features
* News-related features

If the news dataset is unavailable, the system can fall back to using price features only.

The trained model, scaler, and exact feature list are saved so that the prediction stage uses the same features that were used during training.

### 9. `predict.py` — Final Prediction

This is the main prediction stage.

The overall process is:

**News headline → Company identification → Ticker → Fresh market features → Recent news → Model probabilities → Headline sentiment → Final prediction**

For the identified ticker, the system obtains fresh price features and checks recent scraped news from the previous seven days.

The model's predicted probabilities are then combined with the sentiment of the **exact headline entered by the user**.

The influence of the headline sentiment is controlled through:

`config.SENTIMENT_WEIGHT`

---

## Installation and Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Running the Complete Pipeline

For the first complete run, execute the scripts in the following order:

```bash
python data_fetch.py
```

Fetches the historical price data.

```bash
python feature_engineering.py
```

Generates the required market features.

```bash
python labeling.py
```

Creates the spike, crash, and neutral target labels.

```bash
python news_scrape.py
```

Collects news from the available sources.

```bash
python build_dataset.py
```

Combines the price and news information into a single training dataset.

```bash
python train_model.py
```

Trains the Random Forest model using the available features.

Finally, provide a news headline for prediction:

```bash
python predict.py "Tesla shares jump on record deliveries"
```

This final command performs the company identification, sentiment processing, and prediction steps.

---

## Keeping the Dataset Updated

To continuously improve the news coverage, run:

```bash
python news_scrape.py
```

regularly. Running it daily is suitable for maintaining updated headline information.

After collecting additional news, rebuild the dataset and retrain the model:

```bash
python build_dataset.py
python train_model.py
```

The `predict.py` script automatically retrieves fresh price information and uses the current contents of:

`data/news/news_history.csv`

Therefore, prediction does not require retraining unless you want the newly collected news data to become part of the trained model.

---

## Customization

The project can be modified through the configuration files.

### Add More Companies

To include additional companies:

* Add their tickers to `TICKERS` in `config.py`
* Add company-name mappings to `TICKER_ALIASES` in `ner_extract.py`
* Add the relevant company names to `COMPANY_NAMES` in `web_scraper.py`

### Change Historical Data Period

The amount of historical price data can be controlled through:

`YEARS_OF_HISTORY`

inside `config.py`.

### Modify Spike / Crash Thresholds

The threshold that determines whether a movement is considered a spike or crash can be changed using:

* `SPIKE_THRESHOLD`
* `CRASH_THRESHOLD`

The current configuration uses approximately **+1.5% for spike** and **-1.5% for crash**.

### Adjust Headline Sentiment Influence

The contribution of the specific headline's sentiment can be controlled using:

`SENTIMENT_WEIGHT`

The range works conceptually as follows:

* `0` → headline sentiment is ignored
* `1` → model output is ignored in favor of the headline sentiment
* Default value → `0.35`

---

## Important Notes for Report / Viva

### Forward-Looking Labels

The target is based on the **next trading day's return**.

This means the model is designed to predict future movement rather than simply describe the previous day's movement.

The `next_return` value is shifted by `-1` for each ticker before the final unlabeled row is removed.

### Class Imbalance

Neutral market days generally occur much more frequently than spike or crash days.

To handle this imbalance, class weights are automatically balanced during training.

### Company Identification / NER

The company extraction process is intentionally simple and explainable.

It uses:

**Alias Matching → spaCy ORG Entity Detection as Fallback**

For a more advanced implementation, the NER component could later be improved using a fine-tuned HuggingFace NER model or an existing encoder-based news classifier.

### News Data Availability

The project obtains news through two sources:

* `yfinance` news feed
* Google News RSS web scraping

These free sources mainly provide recent headlines and do not provide a complete two-year historical news archive.

Because of this, many older historical records may initially have:

* `news_sentiment_mean = 0`
* `news_count = 0`

This is expected behavior.

As the scraper continues running and more headlines are collected, the amount of available news information will gradually increase.

### Two Sentiment Signals

At prediction time, the system uses two different sentiment signals.

**First:** the trained model has already learned from the available historical news features, including aggregate news sentiment.

**Second:** the exact headline entered by the user is analyzed separately in real time using `sentiment.py`.

The two signals are then combined using:

`config.SENTIMENT_WEIGHT`

Therefore, the final prediction considers both:

* Patterns learned by the trained model
* The wording and sentiment of the specific headline being tested
