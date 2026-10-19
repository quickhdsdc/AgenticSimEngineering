"""Retrieval scoring: label parsing, tier metrics, rejection curve, margin.

Every test here runs on hand-built vectors, so the arithmetic is checkable and
nothing calls an embedding endpoint. That separation is the point: the numbers
this module reports go into decisions about published thresholds, and a metric
that is wrong in a way no test can see would move those thresholds for a reason
nobody could reconstruct.

The label file is validated strictly for the same reason. A case whose tier and
gold disagree, or two cases sharing an id, produce a report that looks fine and
cannot be compared against the previous one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from a4s.aasctl.index import Index, Node
from a4s.experiment.retrieval import (
    LABELS_DIR,
    TIERS,
    Case,
    LabelError,
    LabelSet,
    Outcome,
    evaluate,
    margin_separation,
    rejection_curve,
    render,
    tier_metrics,
    to_json,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def node(path: str, parent: str | None = None) -> Node:
    return Node(path=path, id_short=path.rsplit(".", 1)[-1], model_type="Property", parent=parent)


def one_hot_index(*paths: str) -> Index:
    """An index whose vectors are the standard basis, one axis per node.

    Cosine similarity then reduces to "which axis did the query point along",
    which makes every expected ranking in this file derivable by hand instead
    of being whatever the embedding happened to produce.
    """
    import numpy as np

    return Index(
        submodel_id="urn:x:sm:1",
        id_short="Test",
        model="fake-embedding",
        nodes=[node(p) for p in paths],
        vectors=np.eye(len(paths), dtype="float32"),
        enriched=0,
    )


def pointing_at(order: Sequence[int], size: int) -> Any:
    """A query vector whose similarity to axis i decreases along `order`."""
    import numpy as np

    vector = np.zeros(size, dtype="float32")
    for rank, axis in enumerate(order):
        vector[axis] = 1.0 - rank * 0.1
    return vector


def embedder(vectors: Sequence[Any]):
    def embed(texts: Sequence[str], model: str | None = None) -> Any:  # noqa: ARG001
        assert len(texts) == len(vectors), "one vector per case"
        return list(vectors)

    return embed


def labels(*cases: dict, submodel: str = "urn:x:sm:1") -> LabelSet:
    return LabelSet(
        submodel=submodel, cases=[Case.parse(c) for c in cases]
    )


# ---------------------------------------------------------------------------
# label parsing
# ---------------------------------------------------------------------------

def test_a_negative_case_may_not_name_a_gold_path() -> None:
    """The two ways of saying "unanswerable" must not be able to disagree.

    A case tagged negative that also names a target would be counted as a
    rejection success and a retrieval failure at the same time.
    """
    with pytest.raises(LabelError, match="negative"):
        Case.parse({"id": "x", "query": "q", "tier": "negative", "gold": ["A"]})


def test_an_answerable_case_must_name_a_gold_path() -> None:
    with pytest.raises(LabelError, match="no gold path"):
        Case.parse({"id": "x", "query": "q", "tier": "paraphrase"})


def test_an_unknown_tier_is_refused() -> None:
    with pytest.raises(LabelError, match="unknown tier"):
        Case.parse({"id": "x", "query": "q", "tier": "invented", "gold": ["A"]})


def test_reused_case_ids_are_refused(tmp_path: Path) -> None:
    """Ids attribute a regression to a case across runs; duplicates break that."""
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "submodel": "urn:x:sm:1",
                "cases": [
                    {"id": "same", "query": "a", "gold": ["A"]},
                    {"id": "same", "query": "b", "gold": ["B"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LabelError, match="reuses case ids"):
        LabelSet.load(path)


def test_an_acceptable_path_counts_as_a_hit() -> None:
    """Genuine domain ambiguity must not be scored as a failure.

    Two of the seed cases have more than one defensible target -- an order code
    and an article number answer the same question. Forcing a single gold path
    would measure the labeller's preference rather than the retrieval.
    """
    case = Case.parse(
        {"id": "x", "query": "q", "gold": ["A"], "acceptable": ["B"], "tier": "paraphrase"}
    )
    assert case.hit("A") and case.hit("B")
    assert not case.hit("C")
    assert not case.hit(None)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_rank_is_the_position_of_the_first_hit() -> None:
    index = one_hot_index("A", "B", "C")
    outcomes = evaluate(
        index,
        labels({"id": "c1", "query": "q", "gold": ["C"], "tier": "paraphrase"}),
        embed=embedder([pointing_at([1, 2, 0], 3)]),   # ranks B, C, A
    )
    assert outcomes[0].ranked[:2] == ["B", "C"]
    assert outcomes[0].rank == 2
    assert not outcomes[0].top1
    assert outcomes[0].top5
    assert outcomes[0].reciprocal_rank == 0.5


def test_a_gold_path_outside_the_retrieved_list_leaves_rank_unset() -> None:
    """`None` rather than a large number: absent and last are different facts.

    MRR treats them the same, which is exactly why the raw rank is kept -- a
    submodel where the answer is never retrieved at all needs a different fix
    from one where it ranks fourth.
    """
    index = one_hot_index("A", "B", "C", "D", "E", "F", "G")
    outcomes = evaluate(
        index,
        labels({"id": "c1", "query": "q", "gold": ["G"], "tier": "paraphrase"}),
        embed=embedder([pointing_at([0, 1, 2, 3, 4], 7)]),   # G is never ranked
        top_k=5,
    )
    assert outcomes[0].rank is None
    assert outcomes[0].reciprocal_rank == 0.0
    assert "G" not in outcomes[0].ranked


def test_at_least_five_results_are_retrieved_whatever_top_k_asks_for() -> None:
    """recall@5 has to be measurable, so the search depth has a floor.

    Retrieving three and then reporting a hit at rank 5 would count a position
    that was never returned. `top_k` is a request, not a cap on what the metric
    is allowed to see.
    """
    index = one_hot_index("A", "B", "C", "D", "E", "F")
    outcomes = evaluate(
        index,
        labels({"id": "c1", "query": "q", "gold": ["E"], "tier": "paraphrase"}),
        embed=embedder([pointing_at([0, 1, 2, 3, 4], 6)]),
        top_k=1,
    )
    assert len(outcomes[0].ranked) >= 5
    assert outcomes[0].rank == 5
    assert outcomes[0].top5


def test_margin_is_the_gap_between_the_first_two_hits() -> None:
    index = one_hot_index("A", "B", "C")
    outcomes = evaluate(
        index,
        labels({"id": "c1", "query": "q", "gold": ["A"], "tier": "paraphrase"}),
        embed=embedder([pointing_at([0, 1, 2], 3)]),
    )
    assert outcomes[0].margin == pytest.approx(
        outcomes[0].scores[0] - outcomes[0].scores[1], abs=1e-4
    )
    assert outcomes[0].margin > 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _outcome(tier: str, rank: int | None, top: float, margin: float = 0.05) -> Outcome:
    return Outcome(
        case_id=f"{tier}-{rank}-{top}", tier=tier, query="q", gold=["A"],
        ranked=["A"], scores=[top, top - margin], rank=rank, margin=margin,
    )


def test_tier_metrics_count_only_their_own_tier() -> None:
    rows = [_outcome("paraphrase", 1, 0.6), _outcome("paraphrase", 3, 0.5),
            _outcome("paraphrase", None, 0.4)]
    m = tier_metrics(rows)
    assert m["n"] == 3
    assert m["recall@1"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["recall@5"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3, abs=1e-3)


def test_the_rejection_curve_reports_what_a_threshold_costs() -> None:
    """Refusing everything scores perfectly on negatives, and the curve says so.

    Reported as a pair at every theta because the two errors trade against each
    other, and a single "rejection rate" hides the trade entirely.
    """
    rows = rejection_curve(
        [
            _outcome("paraphrase", 1, 0.60),
            _outcome("paraphrase", 1, 0.35),
            _outcome("negative", None, 0.40),
            _outcome("negative", None, 0.20),
        ],
        (0.3, 0.5),
    )
    at_03, at_05 = rows
    assert (at_03["negatives_rejected"], at_03["correct_hits_lost"]) == (1, 0)
    assert (at_05["negatives_rejected"], at_05["correct_hits_lost"]) == (2, 1)


def test_margin_separation_reports_overlap_as_not_separated() -> None:
    """Open question 7's test, and the shape of the answer it actually got.

    Comparing medians is not enough: distributions with clearly different
    centres can still overlap, and an overlapping pair admits no threshold. The
    seed set lands here -- answerable margins reach down to 0.0126 and
    unanswerable ones up to 0.0383 -- so the margin is not the better band
    signal that one earlier run suggested.
    """
    overlapping = margin_separation(
        [
            _outcome("paraphrase", 1, 0.6, margin=0.02),
            _outcome("paraphrase", 1, 0.6, margin=0.20),
            _outcome("negative", None, 0.3, margin=0.05),
        ]
    )
    assert overlapping["comparable"]
    assert overlapping["separated"] is False

    clean = margin_separation(
        [
            _outcome("paraphrase", 1, 0.6, margin=0.20),
            _outcome("negative", None, 0.3, margin=0.01),
        ]
    )
    assert clean["separated"] is True


def test_margin_separation_needs_both_kinds_of_case() -> None:
    assert not margin_separation([_outcome("paraphrase", 1, 0.6)])["comparable"]


def test_the_report_names_every_answerable_query_that_missed() -> None:
    """A miss is a diagnosis, not a decimal. The report has to name it."""
    index = one_hot_index("A", "B", "C", "D", "E", "F", "G")
    label_set = labels({"id": "c1", "query": "the missing one", "gold": ["G"],
                        "tier": "paraphrase"})
    outcomes = evaluate(
        index, label_set, embed=embedder([pointing_at([0, 1, 2, 3, 4], 7)])
    )

    text = render(index, label_set, outcomes)
    assert "the missing one" in text
    assert "no gold path in the top 5" in text


# ---------------------------------------------------------------------------
# the shipped label set
# ---------------------------------------------------------------------------

def test_the_shipped_label_set_parses() -> None:
    """It is data, and data that fails to load fails at the worst moment.

    Kept as a test rather than a startup check because the file is edited by
    hand: the point is to fail in CI when someone adds a case, not to fail on
    an operator's machine when they run the evaluation.
    """
    files = sorted(LABELS_DIR.glob("*.json"))
    assert files, "no label sets shipped"
    for path in files:
        label_set = LabelSet.load(path)
        # `*` is the repository-wide index, which is not an AAS Identifier and
        # deliberately cannot collide with one.
        assert label_set.submodel == "*" or label_set.submodel.startswith(("http", "urn"))
        assert len(label_set.cases) >= 10


def test_the_shipped_label_set_covers_every_tier() -> None:
    """A set without negatives cannot measure the failure that matters most.

    A similarity search does not usually fail by missing the target; it fails
    by confidently returning the nearest thing to a target that is not there.
    Only the negative tier sees that, so its absence would make the whole
    report misleading rather than merely incomplete.
    """
    label_set = LabelSet.load(LABELS_DIR / "battery-passport-527.json")
    for tier in TIERS:
        assert label_set.by_tier(tier), f"no {tier} cases"
    negatives = label_set.by_tier("negative")
    assert len(negatives) / len(label_set.cases) >= 0.15, (
        f"only {len(negatives)} negatives out of {len(label_set.cases)}"
    )


def test_the_json_report_carries_enough_to_rebuild_the_table() -> None:
    """A report that only holds the summary cannot be re-read at another theta.

    Re-embedding to change a threshold costs money and produces a run that is
    not comparable to the first for a reason unrelated to what changed.
    """
    index = one_hot_index("A", "B")
    label_set = labels(
        {"id": "c1", "query": "q", "gold": ["A"], "tier": "paraphrase"},
        {"id": "n1", "query": "n", "gold": [], "tier": "negative"},
    )
    outcomes = evaluate(
        index, label_set,
        embed=embedder([pointing_at([0, 1], 2), pointing_at([1, 0], 2)]),
    )

    payload = to_json(index, label_set, outcomes)
    assert payload["embedding_model"] == "fake-embedding"
    assert set(payload["tiers"]) == set(TIERS)
    assert len(payload["outcomes"]) == 2
    assert all("scores" in o and "ranked" in o for o in payload["outcomes"])
