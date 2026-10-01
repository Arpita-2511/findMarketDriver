"""
Phase 6 evaluation TOOLING. Labels below are SYNTHETIC test fixtures used only
to check the metric code - they are not human labels and say nothing about
the classifier's real accuracy.
"""

import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from services.event_evaluation import (
    EventEvaluationError,
    evaluate_against_human_labels,
    export_validation_template,
    load_human_labels,
    select_validation_sample,
)
from services.event_taxonomy import EVENT_TYPES
from tests.event_fakes import classified

NY = ZoneInfo("America/New_York")
T0 = datetime(2024, 7, 8, 9, 0, tzinfo=NY)

TEXTS = ["Apple Reports Q1 EPS beat", "Apple Says It Is Suing Qualcomm", "Wells Fargo Cuts Apple Estimates",
         "Inside The Hidden Economy Of Pawn Shops"]
ITEMS = [classified(i, T0 + timedelta(hours=i), f"{TEXTS[i % 4]} #{i}") for i in range(20)]


def write_labels(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["provider", "provider_article_id", "headline", "summary", "human_event_type", "notes"])
        w.writerows(rows)
    return path


def test_sample_is_deterministic_stratified_and_bounded():
    a, b = select_validation_sample(ITEMS, 8), select_validation_sample(list(reversed(ITEMS)), 8)
    assert [c.key for c in a] == [c.key for c in b]
    assert len(a) == 8 and {c.event_type for c in a} == {"EARNINGS", "LEGAL", "ANALYST_RATING", "OTHER"}
    assert len(select_validation_sample(ITEMS, 500)) == 20 and select_validation_sample([], 5) == []


def test_template_hides_predictions_and_lists_taxonomy(tmp_path):
    path = export_validation_template(ITEMS, 6, tmp_path / "labels.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert len(rows) == 6 and all(r["human_event_type"] == "" for r in rows)
    assert "predicted" not in ",".join(rows[0].keys()).lower()                      # no anchoring
    instructions = path.with_suffix(".instructions.txt").read_text(encoding="utf-8")
    assert all(t in instructions for t in EVENT_TYPES)
    with pytest.raises(FileExistsError):
        export_validation_template(ITEMS, 6, path)                                  # may hold labels


def test_no_human_labels_means_no_accuracy(tmp_path):
    path = write_labels(tmp_path / "empty.csv", [["alpaca", "1", "h", "", "", ""]])
    with pytest.raises(EventEvaluationError, match="formal classification accuracy cannot be established"):
        load_human_labels(path)


def test_invalid_human_label_rejected(tmp_path):
    path = write_labels(tmp_path / "bad.csv", [["alpaca", "1", "h", "", "earnings call", ""]])
    with pytest.raises(EventEvaluationError, match="not a taxonomy event type"):
        load_human_labels(path)


def test_metrics_with_synthetic_labels(tmp_path):
    # synthetic: articles 0..7 labelled; article 2 (an ANALYST prediction) labelled LEGAL on purpose
    rows = []
    for c in ITEMS[:8]:
        label = "LEGAL" if c.article.provider_article_id == "2" else c.event_type
        rows.append(["alpaca", c.article.provider_article_id, "h", "", label.lower(), ""])
    labels = load_human_labels(write_labels(tmp_path / "labels.csv", rows))
    report = evaluate_against_human_labels(ITEMS, labels)

    clf = report["classifier"]
    assert report["n_human_labelled"] == 8 and report["label_source"].startswith("human")
    assert clf["accuracy"] == pytest.approx(7 / 8)
    assert clf["per_class"]["EARNINGS"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}
    assert clf["per_class"]["LEGAL"]["recall"] == pytest.approx(2 / 3)
    assert clf["per_class"]["ANALYST_RATING"]["precision"] == pytest.approx(1 / 2)
    assert 0 < clf["macro_f1"] < 1 and 0 < clf["weighted_f1"] < 1
    cm = clf["confusion_matrix"]
    i_legal, i_analyst = cm["rows_true_cols_pred"].index("LEGAL"), cm["rows_true_cols_pred"].index("ANALYST_RATING")
    assert cm["matrix"][i_legal][i_analyst] == 1
    assert report["baseline_always_other"]["accuracy"] == pytest.approx(2 / 8)
    assert report["baseline_majority_human_label"]["label"] == "LEGAL"
    assert "no statistical claims" in report["warning"]


def test_labels_for_unknown_articles_rejected(tmp_path):
    path = write_labels(tmp_path / "x.csv", [["alpaca", "999", "h", "", "LEGAL", ""]])
    with pytest.raises(EventEvaluationError, match="not in the event dataset"):
        evaluate_against_human_labels(ITEMS, load_human_labels(path))
