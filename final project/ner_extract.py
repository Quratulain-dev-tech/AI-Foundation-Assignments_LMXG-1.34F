"""
Step 5: Given a news title, find which company/ticker it's about.

Approach (in priority order):
  1. Direct company-name match against TICKER_ALIASES (case-insensitive,
     whole-word). This is the primary, most reliable method for the
     companies you're actually tracking in config.TICKERS.
  2. Direct bare ticker symbol mention, e.g. "$AAPL" or standalone "AAPL".
  3. spaCy NER fallback: pulls ORG entities and fuzzy-matches them against
     TICKER_ALIASES. This only helps for company names NOT already in your
     alias list -- the small spaCy model (en_core_web_sm) is often too weak
     to reliably tag single-word company names in terse headlines (e.g. it
     misses "Tesla" in "Tesla shares jump on record deliveries" entirely),
     so alias matching above is what actually carries this project.

Install spaCy model once:
    python -m spacy download en_core_web_sm

Usage:
    python ner_extract.py "Apple unveils new iPhone, shares jump"
"""

import re
import sys
import spacy

import config

# Map company name variants -> ticker. Extend this as you add more tickers.
TICKER_ALIASES = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "microsoft": "MSFT",
    "microsoft corp": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "nvidia": "NVDA",
}

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_ticker(title: str):
    """Return the best-guess ticker symbol for a news title, or None."""

    title_lower = title.lower()
    known_tickers = set(config.TICKERS)

    # 1. Direct company-name alias match (primary method -- reliable for the
    #    fixed set of companies this project tracks). Word-boundary match so
    #    "google" doesn't match inside some unrelated longer word.
    for alias, ticker in TICKER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", title_lower):
            return ticker

    # 2. Direct ticker symbol mention, e.g. "$AAPL" or standalone "AAPL"
    for token in re.findall(r"\$?[A-Z]{2,5}", title):
        token = token.lstrip("$")
        if token in known_tickers:
            return token

    # 3. NER fallback: look for organization names and match against aliases
    nlp = get_nlp()
    doc = nlp(title)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            key = ent.text.strip().lower()
            if key in TICKER_ALIASES:
                return TICKER_ALIASES[key]
            # loose match: alias contained in entity text or vice versa
            for alias, ticker in TICKER_ALIASES.items():
                if alias in key or key in alias:
                    return ticker

    return None


if __name__ == "__main__":
    title = " ".join(sys.argv[1:]) or "Apple unveils new iPhone, shares jump"
    ticker = extract_ticker(title)
    print(f"Title:  {title}")
    print(f"Ticker: {ticker}")
