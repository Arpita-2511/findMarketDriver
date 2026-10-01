"""
Event-classifier evaluation against HUMAN labels (Phase 6).

There is no labelled event dataset in this project. Classifier output and
its event_confidence are NOT accuracy. Supervised metrics are computed only
from labels a person has entered:

    1. export_validation_template  -> CSV with headline/summary and an EMPTY
                                      human_event_type column (predictions are
                                      deliberately NOT shown, to avoid anchoring)
    2. a person fills human_event_type (one of EVENT_TYPES) for some rows
    3. evaluate_against_human_labels -> precision / recall / F1 per class,
                                      macro & weighted F1, confusion matrix,
                                      class support, compared with baselines:
                                        - always OTHER
                                        - majority human label (an optimistic
                                          reference: it peeks at the labels)

Small samples cannot support statistical claims; the report says so.
Template files contain licensed headlines and live in the git-ignored
data/processed/events/validation/ directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.event_features import ClassifiedArticle  # noqa: E402
from services.event_taxonomy import EVENT_DESCRIPTIONS, EVENT_TYPES, NO_EVENT, EventTaxonomyError, validate_event_type  # noqa: E402

VALIDATION_DIR = PROJECT_ROOT / "data" / "processed" / "events" / "validation"
TEMPLATE_COLUMNS = ("provider", "provider_article_id", "headline", "summary", "human_event_type", "notes")
SMALL_SAMPLE = 100


class EventEvaluationError(ValueError):
    """No usable human labels or malformed label file."""


def _stable_rank(c: ClassifiedArticle) -> str:
    return hashlib.sha256(f"{c.article.provider}:{c.article.provider_article_id}".encode()).hexdigest()


def select_validation_sample(classified: list[ClassifiedArticle], n: int) -> list[ClassifiedArticle]:
    """
    Deterministic, roughly stratified sample: round-robin over predicted types
    (taxonomy order), each type's articles ordered by a hash of their id.
    Predicted types are used only to spread the sample, never shown.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    groups = {t: sorted((c for c in classified if c.event_type == t), key=_stable_rank) for t in EVENT_TYPES}
    sample, i = [], 0
    while len(sample) < min(n, len(classified)):
        for t in EVENT_TYPES:
            if i < len(groups[t]) and len(sample) < n:
                sample.append(groups[t][i])
        i += 1
    return sample


def export_validation_template(classified: list[ClassifiedArticle], n: int, path: Path) -> Path:
    """Write a labelling template (+ instructions file). Refuses to overwrite a template that may hold labels."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists (it may contain human labels); choose another file name")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TEMPLATE_COLUMNS)
        for c in select_validation_sample(classified, n):
            a = c.article
            writer.writerow([a.provider, a.provider_article_id, a.headline, a.summary or "", "", ""])
    instructions = path.with_suffix(".instructions.txt")
    instructions.write_text(
        "HUMAN LABELLING TEMPLATE - fill column human_event_type with exactly one of:\n"
        + "".join(f"  {t}: {EVENT_DESCRIPTIONS[t]}\n" for t in EVENT_TYPES)
        + "Leave it empty to skip a row. Judge only the headline/summary text. "
          "Do not copy classifier output - it is intentionally not shown.\n",
        encoding="utf-8")
    return path


def load_human_labels(path: Path) -> dict[tuple[str, str], str]:
    """{(provider, provider_article_id): human label} for non-empty rows; labels must be in the taxonomy."""
    labels = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("provider", "provider_article_id", "human_event_type") if c not in (reader.fieldnames or [])]
        if missing:
            raise EventEvaluationError(f"label file is missing columns {missing}")
        for row_no, row in enumerate(reader, start=2):
            value = (row["human_event_type"] or "").strip().upper()
            if not value:
                continue
            try:
                validate_event_type(value)
            except EventTaxonomyError:
                raise EventEvaluationError(f"line {row_no}: {value!r} is not a taxonomy event type") from None
            labels[(row["provider"], str(row["provider_article_id"]))] = value
    if not labels:
        raise EventEvaluationError("no human labels found - formal classification accuracy cannot be established")
    return labels


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = [t for t in EVENT_TYPES if t in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "labels": labels,
        "per_class": {lbl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, lbl in enumerate(labels)},
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "confusion_matrix": {"rows_true_cols_pred": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def evaluate_against_human_labels(classified: list[ClassifiedArticle], labels: dict[tuple[str, str], str]) -> dict:
    by_key = {c.key: c for c in classified}
    unknown = [k for k in labels if k not in by_key]
    if unknown:
        raise EventEvaluationError(f"{len(unknown)} labelled article(s) are not in the event dataset, e.g. {unknown[0]}")
    keys = sorted(labels)
    y_true = [labels[k] for k in keys]
    y_pred = [by_key[k].event_type for k in keys]
    majority = Counter(y_true).most_common(1)[0][0]
    report = {
        "n_human_labelled": len(keys),
        "label_source": "human (manual validation sample)",
        "class_distribution_human": dict(Counter(y_true)),
        "classifier": _metrics(y_true, y_pred),
        "baseline_always_other": _metrics(y_true, [NO_EVENT] * len(keys)),
        "baseline_majority_human_label": {"label": majority, **_metrics(y_true, [majority] * len(keys))},
        "warning": None,
    }
    if len(keys) < SMALL_SAMPLE:
        report["warning"] = (f"only {len(keys)} labelled articles - descriptive only; "
                             "no statistical claims about classifier quality")
    return report


def main(argv: list[str] | None = None) -> int:
    from services.historical_events import load_event_dataset

    parser = argparse.ArgumentParser(description="Human-labelled evaluation of the event classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--events", type=Path, required=True)
        p.add_argument("--sentiment", type=Path, required=True)
        p.add_argument("--news", type=Path, required=True)
    sub.choices["export"].add_argument("--n", type=int, default=60)
    sub.choices["export"].add_argument("--out", type=Path, default=VALIDATION_DIR / "event_labels.csv")
    sub.choices["evaluate"].add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)

    classified, _ = load_event_dataset(args.events, args.sentiment, args.news)
    if args.command == "export":
        path = export_validation_template(classified, args.n, args.out)
        print(f"Template written: {path}  (fill human_event_type, then run the 'evaluate' command)")
    else:
        print(json.dumps(evaluate_against_human_labels(classified, load_human_labels(args.labels)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
