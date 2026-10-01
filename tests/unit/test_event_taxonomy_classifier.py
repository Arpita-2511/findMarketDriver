"""Phase 6: central taxonomy and the rules_v1 keyword classifier (offline)."""

from pathlib import Path

import pytest

from services.event_classifier import (
    RULES,
    ConstantEventClassifier,
    EventPrediction,
    KeywordEventClassifier,
)
from services.event_features import EVENT_TYPE_COLUMNS
from services.event_taxonomy import (
    EVENT_DESCRIPTIONS,
    EVENT_IMPACTS,
    EVENT_TYPES,
    NO_EVENT,
    EventTaxonomyError,
    impact_from_sentiment_label,
    type_feature_name,
    validate_event_impact,
    validate_event_type,
)

ROOT = Path(__file__).resolve().parents[2]
CLF = KeywordEventClassifier()


# ==========================================
# Taxonomy
# ==========================================


def test_taxonomy_is_valid_and_complete():
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES)) == 15
    assert EVENT_TYPES[-1] == NO_EVENT == "OTHER"
    assert set(EVENT_DESCRIPTIONS) == set(EVENT_TYPES)
    assert set(RULES) == set(EVENT_TYPES) - {NO_EVENT}
    for t in EVENT_TYPES:
        assert validate_event_type(t) == t


@pytest.mark.parametrize("bad", ["earnings", "PRODUCT_LAUNCH", "", "SENTIMENT"])
def test_invalid_event_types_rejected(bad):
    with pytest.raises(EventTaxonomyError):
        validate_event_type(bad)


def test_impacts():
    assert EVENT_IMPACTS == ("POSITIVE", "NEGATIVE", "NEUTRAL")
    assert [impact_from_sentiment_label(x) for x in ("positive", "negative", "neutral")] == list(EVENT_IMPACTS)
    with pytest.raises(EventTaxonomyError):
        impact_from_sentiment_label("bullish")
    with pytest.raises(EventTaxonomyError):
        validate_event_impact("MIXED")


def test_feature_columns_derive_from_taxonomy():
    assert type_feature_name("M_AND_A") == "m_and_a_event_count"
    assert EVENT_TYPE_COLUMNS == tuple(f"{t.lower()}_event_count" for t in EVENT_TYPES if t != NO_EVENT)


def test_taxonomy_is_defined_in_exactly_one_module():
    defining = [p.name for p in (ROOT / "services").glob("*.py")
                if "EVENT_TYPES: tuple" in p.read_text(encoding="utf-8")
                or "\nEVENT_TYPES = " in p.read_text(encoding="utf-8")]
    assert defining == ["event_taxonomy.py"]


# ==========================================
# Classifier on real sample headlines
# ==========================================


@pytest.mark.parametrize("headline, expected", [
    ("Apple Reports Q1 EPS $3.36 vs $3.22 Est., Sales $78.4B vs $77.38B Est.", "EARNINGS"),
    ("Appple Sees Q2 Rev. $51.5-$53.5B vs. Est. $53.9B", "GUIDANCE"),
    ("Guggenheim Initiates Coverage On Apple at Buy, Announces $140.00 Target", "ANALYST_RATING"),
    ("Wells Fargo Cuts Apple Estimates", "ANALYST_RATING"),
    ("Apple Says It Is Suing Qualcomm Over Licensing Practices -DJ", "LEGAL"),
    ("Regulators Open Antitrust Investigation Into Apple App Store", "REGULATORY"),
    ("Option Alert: Apple Mar $120 Call; 4062 @Bid @ $2.38", "MARKET_ACTIVITY"),
    ("13F from Berkshire Hathaway Shows Increased Apple Stake from ~15.2M Shares to ~57M Shares", "OWNERSHIP"),
    ("Hearing Book for Apple's $10B Bond Offer Reaches $38B", "CAPITAL_ACTION"),
    ("Apple Holding Annual Shareholder Mtg Today", "MANAGEMENT_GOVERNANCE"),
    ("Apple Said to be Partnering with Carl Zeiss on AR glasses for 2018 Launch -AppleInsider", "PARTNERSHIP"),
    ("99 Problems And Time For Investment? Sprint Acquires 33% Of Jay Z's Tidal", "M_AND_A"),
    ("What If Trump Brands China A Currency Manipulator?", "MACRO"),
    ("Apple to Launch Three New iPads in Spring 2017; No iPad Mini", "PRODUCT"),
    ("Jabil Circuit to be Supplier for Next-Gen iPhone -DigiTimes", "OPERATIONS"),
    ("Quanta to Manufacture Next-gen Apple Watch -DigiTimes", "OPERATIONS"),
    ("Inside The Hidden Economy Of Pawn Shops", "OTHER"),
])
def test_classifier_on_sample_headlines(headline, expected):
    p = CLF.classify_text(headline)
    assert p.event_type == expected
    assert 0.0 <= p.event_confidence <= 1.0
    assert all(rule.split(":")[0] in EVENT_TYPES for rule in p.matched_rules)


def test_confidence_is_share_of_rule_evidence():
    p = CLF.classify_text("Appple Sees Q2 Rev. $51.5-$53.5B vs. Est. $53.9B")
    assert p.matched_rules == ("EARNINGS:quarter", "GUIDANCE:sees_quarter")
    assert p.event_confidence == pytest.approx(3 / (3 + 1))


def test_no_evidence_is_other_with_zero_confidence():
    p = CLF.classify_text("Inside The Hidden Economy Of Pawn Shops")
    assert (p.event_type, p.event_confidence, p.matched_rules) == ("OTHER", 0.0, ())


def test_ties_break_in_taxonomy_order():
    p = CLF.classify_text("Fed Meeting Starts Today, But Focus Remains On Key Earnings Data")   # 2 vs 2
    assert p.event_type == "EARNINGS" and p.event_confidence == pytest.approx(0.5)


def test_deterministic_and_case_insensitive():
    text = "Apple Reports Q1 EPS beat"
    assert CLF.classify_text(text) == CLF.classify_text(text) == CLF.classify_text(text.upper())
    assert CLF.classify_texts([text, "x"]) == [CLF.classify_text(text), CLF.classify_text("x")]


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_text_rejected(bad):
    with pytest.raises(ValueError, match="empty"):
        CLF.classify_text(bad)


@pytest.mark.parametrize("kwargs, message", [
    ({"event_type": "NEWS", "event_confidence": 0.5, "matched_rules": ()}, "unknown event type"),
    ({"event_type": "LEGAL", "event_confidence": 1.5, "matched_rules": ()}, r"\[0, 1\]"),
    ({"event_type": "LEGAL", "event_confidence": 1, "matched_rules": ()}, "float"),
    ({"event_type": "OTHER", "event_confidence": 0.3, "matched_rules": ()}, "0.0"),
])
def test_prediction_schema_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EventPrediction(**kwargs)


def test_constant_baseline():
    assert ConstantEventClassifier().classify_texts(["a", "b"]) == [EventPrediction("OTHER", 0.0, ())] * 2
    assert ConstantEventClassifier("EARNINGS").classify_texts(["a"])[0].event_type == "EARNINGS"
    with pytest.raises(EventTaxonomyError):
        ConstantEventClassifier("NOPE")
