"""Offline Phase 6 fixtures: ClassifiedArticle builder (synthetic sentiment, real keyword classifier)."""

from services.event_classifier import KeywordEventClassifier
from services.event_features import ClassifiedArticle
from services.finbert_sentiment import INFERENCE_VERSION, LABELS, SentimentResult, build_sentiment_text, text_hash
from services.sentiment_features import ScoredArticle
from tests.sentiment_fakes import news

POS = (0.7, 0.1, 0.2)      # (positive, negative, neutral)
NEG = (0.1, 0.8, 0.1)
NEU = (0.2, 0.1, 0.7)

_CLASSIFIER = KeywordEventClassifier()


def sentiment_for(article, probs) -> SentimentResult:
    pos, neg, neu = probs
    by_label = {"positive": pos, "negative": neg, "neutral": neu}
    label = max(LABELS, key=lambda lbl: (by_label[lbl], -LABELS.index(lbl)))
    return SentimentResult(inference_version=INFERENCE_VERSION, model_name="ProsusAI/finbert", model_revision="rev",
                           label=label, positive_probability=pos, negative_probability=neg,
                           neutral_probability=neu, sentiment_score=pos - neg,
                           input_text_hash=text_hash(build_sentiment_text(article)))


def classified(i, when, headline, probs=NEU, symbols=("AAPL",), updated=None) -> ClassifiedArticle:
    article = news(i, when, headline, symbols=symbols, updated=updated)
    scored = ScoredArticle(article, sentiment_for(article, probs))
    return ClassifiedArticle(scored, _CLASSIFIER.classify_text(build_sentiment_text(article)))
