"""
Simple keyword-based sentiment scorer for news headlines.

Why keyword-based and not a trained model: this project's training data
(data_fetch.py -> features.csv -> labeled_features.csv) only contains price
history, not historical news headlines matched to dates. Without real
historical headlines tied to each date/ticker, we cannot properly *train* a
model to use headline text as a feature (there's nothing to learn from).

Instead, this module scores ANY headline directly using a positive/negative
word list, and predict.py blends that score with the price-based model's
output at prediction time (see config.SENTIMENT_WEIGHT). This is a
transparent, rule-based signal -- not a trained model -- but it's an honest
way to make the headline's actual content affect the final answer.

If you later collect real historical headlines (e.g. from a news API) tied
to dates/tickers, you could replace this with a proper trained sentiment/
text model and use it as a real training feature instead of a blend.
"""

import re

POSITIVE_WORDS = {
    "jump", "jumps", "surge", "surges", "soar", "soars", "rally", "rallies",
    "record", "high", "highs", "beat", "beats", "strong", "growth", "gain",
    "gains", "profit", "profits", "upgrade", "upgraded", "breakthrough",
    "boost", "boosts", "win", "wins", "expand", "expands", "expansion",
    "deal", "partnership", "outperform", "bullish", "recovery", "recovers",
    "exceeds", "optimistic", "success", "successful", "innovative",
    "milestone", "positive", "rise", "rises", "rising", "climbs", "climb",
}

NEGATIVE_WORDS = {
    "drop", "drops", "fall", "falls", "falling", "crash", "crashes",
    "plunge", "plunges", "recall", "recalls", "lawsuit", "lawsuits", "weak",
    "miss", "misses", "decline", "declines", "layoffs", "cut", "cuts",
    "loss", "losses", "downgrade", "downgraded", "bug", "bugs", "fraud",
    "investigation", "probe", "sues", "sued", "bearish", "slump", "slumps",
    "warns", "warning", "concern", "concerns", "worried", "worry", "risk",
    "risks", "shutdown", "delay", "delays", "delayed", "scandal", "fine",
    "fined", "penalty", "resign", "resigns", "resignation", "strike",
    "fired", "bankruptcy", "default", "crisis",
}


def score_sentiment(title: str) -> float:
    """Return a sentiment score in [-1, 1]. 0 means neutral/no signal found."""
    words = re.findall(r"[a-zA-Z']+", title.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)

    if pos == 0 and neg == 0:
        return 0.0

    return (pos - neg) / (pos + neg)


def sentiment_to_distribution(score: float) -> dict:
    """
    Turn a sentiment score into a pseudo probability distribution over
    {crash, neutral, spike}, used for blending with the model's output.
    """
    if score > 0:
        spike = score
        crash = 0.0
    elif score < 0:
        crash = -score
        spike = 0.0
    else:
        spike = crash = 0.0

    neutral = 1.0 - spike - crash
    return {"crash": crash, "neutral": neutral, "spike": spike}


if __name__ == "__main__":
    import sys
    title = " ".join(sys.argv[1:]) or "Tesla shares jump on record deliveries"
    s = score_sentiment(title)
    print(f"Title:     {title}")
    print(f"Sentiment: {s:+.2f}")
    print(f"Distribution: {sentiment_to_distribution(s)}")
