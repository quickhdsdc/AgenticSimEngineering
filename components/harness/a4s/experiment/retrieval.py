"""Measuring whether retrieval finds the right element.

`aas index` and `aas find` have 29 tests between them and none of them asks the
only question that matters: **given a query, is the right element in the top
k?** What is tested is the machinery -- descriptor construction, cosine
ranking, boundary expansion, disk round-trip, slug naming. All of that can be
correct while the search is useless.

The consequence is not academic. `aas match` splits on two thresholds that are
reported in the paper, θ_high = 0.8 and θ_low = 0.5, and `PROGRESS.md` open
question 6 records that a later change to the descriptor moved the similarity
distribution underneath them -- so the auto-accept band no longer fires, and
nobody can say where to put it instead. Re-tuning 0.8 until it lights up again
would be fitting a published parameter to whatever is at hand. What is needed
is a labelled set, and there was not one.

There was *almost* one. `Node.descriptor` carries a table comparing four
descriptor variants on "10 query/target pairs drawn from a real
battery-passport submodel". Those ten pairs exist nowhere in the repository:
the conclusion survived and the evidence did not, so the experiment cannot be
repeated, extended, or checked. This module and the files in `labels/` are
that evidence, written down.

Four tiers, reported separately
-------------------------------
Aggregating them would hide the only number that means anything.

``verbatim``
    The query is the element's own idShort or preferred name. Near-circular:
    `descriptor()` puts the preferred name into the embedded text, so a
    verbatim query is matching a string against itself. Kept as a **floor** --
    a configuration that fails here is broken, and one that passes here has
    demonstrated nothing.

``paraphrase``
    Natural language that deliberately avoids the element's name. "how much
    charge does it hold" against `RatedCapacity`. **This is the measurement.**

``multilingual``
    The same, in a language the descriptors are not written in. The deployment
    is German and the operators here are Chinese-speaking; an English-only
    retrieval surface is a limit worth knowing about rather than discovering.

``negative``
    A query the submodel genuinely cannot answer, with an empty ``gold``. The
    common failure of a similarity search is not missing the target, it is
    confidently returning the nearest thing to a target that does not exist,
    and only negatives measure that.

What is reported
----------------
``recall@1`` and ``recall@5`` bound the two bands `aas match` uses: nothing can
be auto-accepted that is not rank 1, and nothing can be arbitrated that is not
in the top k. If recall@5 is low, **a better arbitration model cannot help** --
the right answer was never in the list.

``margin`` is rank-1 minus rank-2, recorded because open question 7 observed it
separating entities with a real target (0.098-0.200) from entities the submodel
cannot hold (0.003-0.006) an order of magnitude more sharply than the absolute
score does. Every plan `aas match` writes already carries it and nothing reads
it. Here it is read.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..aasctl.index import Index, search

# The tiers, in the order they are reported: floor first, then the real
# measurement, then the two that probe specific weaknesses.
TIERS = ("verbatim", "paraphrase", "multilingual", "negative")

LABELS_DIR = Path(__file__).parent / "labels"


class LabelError(ValueError):
    """The label file cannot be read as asked."""


@dataclass
class Case:
    """One query and the paths that would answer it."""

    id: str
    query: str
    gold: List[str] = field(default_factory=list)
    # Paths that are defensible but not the intended answer. Counted as a hit
    # so that a genuinely ambiguous query does not punish a correct system --
    # "internal resistance" really does have three legitimate targets here.
    acceptable: List[str] = field(default_factory=list)
    tier: str = "paraphrase"
    note: str = ""
    # The structural constraints an agent could reasonably have derived from
    # the question before searching -- "a Property", "in a Nameplate", "in kW".
    # Declared per case rather than globally so the same set can be scored
    # twice, with and without them, and the difference attributed.
    filter: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        return not self.gold

    def as_filter(self) -> Any:
        from ..aasctl.index import Filter

        return Filter(**self.filter) if self.filter else None

    def hit(self, path: Optional[str]) -> bool:
        return bool(path) and (path in self.gold or path in self.acceptable)

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "Case":
        if "query" not in raw:
            raise LabelError(f"case has no query: {json.dumps(raw)[:120]}")
        tier = raw.get("tier", "paraphrase")
        if tier not in TIERS:
            raise LabelError(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
        gold = list(raw.get("gold") or [])
        if tier == "negative" and gold:
            raise LabelError(f"case {raw.get('id')!r} is tier negative but names a gold path")
        if tier != "negative" and not gold:
            raise LabelError(f"case {raw.get('id')!r} has no gold path and is not tier negative")
        return cls(
            id=str(raw.get("id") or raw["query"][:24]),
            query=str(raw["query"]),
            gold=gold,
            acceptable=list(raw.get("acceptable") or []),
            tier=tier,
            note=str(raw.get("note") or ""),
            filter=dict(raw.get("filter") or {}),
        )


@dataclass
class LabelSet:
    submodel: str
    cases: List[Case]
    note: str = ""

    @classmethod
    def load(cls, path: Path) -> "LabelSet":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LabelError(f"no label file at {path}") from exc
        except json.JSONDecodeError as exc:
            raise LabelError(f"{path} is not JSON: {exc}") from exc
        if "submodel" not in raw:
            raise LabelError(f"{path} does not name a submodel")

        cases = [Case.parse(c) for c in raw.get("cases") or []]
        if not cases:
            raise LabelError(f"{path} has no cases")
        seen: Dict[str, int] = {}
        for case in cases:
            seen[case.id] = seen.get(case.id, 0) + 1
        duplicates = [i for i, n in seen.items() if n > 1]
        if duplicates:
            # Ids are how a regression is attributed to a case across runs. Two
            # cases sharing one make the comparison silently wrong.
            raise LabelError(f"{path} reuses case ids: {', '.join(duplicates)}")
        return cls(submodel=str(raw["submodel"]), cases=cases, note=str(raw.get("note") or ""))

    def by_tier(self, tier: str) -> List[Case]:
        return [c for c in self.cases if c.tier == tier]


@dataclass
class Outcome:
    """What one query produced, kept in full so a report can be re-derived."""

    case_id: str
    tier: str
    query: str
    gold: List[str]
    ranked: List[str]                  # paths, best first
    scores: List[float]
    rank: Optional[int] = None         # 1-based position of the first hit, None if absent
    margin: float = 0.0                # score[0] - score[1]
    candidates: int = 0                # rows the filter left for ranking to choose among

    @property
    def top1(self) -> bool:
        return self.rank == 1

    @property
    def top5(self) -> bool:
        return self.rank is not None and self.rank <= 5

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0


def evaluate(
    index: Index,
    labels: LabelSet,
    *,
    top_k: int = 5,
    embed: Optional[Any] = None,
    use_filters: bool = False,
) -> List[Outcome]:
    """Run every case against the index. No thresholds applied here.

    Thresholds are a reporting decision, not a retrieval one: the same run has
    to be readable at several θ without re-embedding, which costs money and
    would make two reports incomparable for a reason unrelated to what changed.
    """
    from ..aasctl import index as index_mod

    embed_fn = embed or index_mod.embed
    vectors = embed_fn([c.query for c in labels.cases], model=index.model)

    # recall@5 is reported, so at least five results have to be retrieved
    # whatever the caller asked for. Retrieving fewer and then reporting
    # "recall@5" would be counting positions that were never returned.
    depth = max(top_k, 5)

    from ..aasctl.index import rows_matching

    outcomes: List[Outcome] = []
    for case, vector in zip(labels.cases, vectors):
        where = case.as_filter() if use_filters else None
        hits = search(index, vector, top_k=depth, where=where)
        ranked = [h.node.path for h in hits]
        scores = [float(h.score) for h in hits]

        rank: Optional[int] = None
        for position, path in enumerate(ranked, start=1):
            if case.hit(path):
                rank = position
                break

        outcomes.append(
            Outcome(
                case_id=case.id,
                tier=case.tier,
                query=case.query,
                gold=list(case.gold),
                # Stored whole rather than truncated: the ranked list is the
                # diagnosis when a case misses, and `rank` is only meaningful
                # against the list it was computed from.
                ranked=ranked,
                scores=scores,
                rank=rank,
                margin=round(scores[0] - scores[1], 4) if len(scores) > 1 else 0.0,
                candidates=len(rows_matching(index, where)),
            )
        )
    return outcomes


def compare(
    index: Index,
    labels: LabelSet,
    *,
    top_k: int = 5,
    embed: Optional[Any] = None,
) -> Dict[str, List[Outcome]]:
    """Score the same set twice, with the structural filters off and on.

    The whole argument for filtering rests on a difference, so the difference
    is what gets measured. Running one arm alone would leave the question of
    what filtering bought to an anecdote.
    """
    return {
        "unfiltered": evaluate(index, labels, top_k=top_k, embed=embed, use_filters=False),
        "filtered": evaluate(index, labels, top_k=top_k, embed=embed, use_filters=True),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def tier_metrics(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """recall@1, recall@5 and MRR over one tier. Meaningless for negatives."""
    if not outcomes:
        return {"n": 0}
    n = len(outcomes)
    return {
        "n": n,
        "recall@1": round(sum(o.top1 for o in outcomes) / n, 3),
        "recall@5": round(sum(o.top5 for o in outcomes) / n, 3),
        "mrr": round(sum(o.reciprocal_rank for o in outcomes) / n, 3),
        "median_top_score": round(statistics.median(o.top_score for o in outcomes), 4),
        "median_margin": round(statistics.median(o.margin for o in outcomes), 4),
    }


def rejection_curve(
    outcomes: Sequence[Outcome], thresholds: Sequence[float]
) -> List[Dict[str, Any]]:
    """How many negatives are correctly refused at each θ, and what it costs.

    Reported as a curve rather than a number because the two errors trade
    against each other and only the caller knows the exchange rate. Refusing
    everything scores perfectly on negatives; the paired column says what it
    threw away to get there.
    """
    positives = [o for o in outcomes if o.tier != "negative"]
    negatives = [o for o in outcomes if o.tier == "negative"]
    rows: List[Dict[str, Any]] = []
    for theta in thresholds:
        rejected = sum(1 for o in negatives if o.top_score < theta)
        lost = sum(1 for o in positives if o.top1 and o.top_score < theta)
        rows.append(
            {
                "theta": theta,
                "negatives_rejected": rejected,
                "negatives_total": len(negatives),
                "correct_hits_lost": lost,
                "correct_hits_total": sum(1 for o in positives if o.top1),
            }
        )
    return rows


def margin_separation(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """Open question 7, answered from data instead of from one run's anecdote.

    If the margin distributions for answerable and unanswerable queries do not
    overlap, the margin is a better band signal than the absolute score, and
    `aas match` should be splitting on it.
    """
    hits = [o.margin for o in outcomes if o.tier != "negative" and o.top1]
    misses = [o.margin for o in outcomes if o.tier == "negative"]
    if not hits or not misses:
        return {"comparable": False}
    return {
        "comparable": True,
        "hit_margin_median": round(statistics.median(hits), 4),
        "hit_margin_min": round(min(hits), 4),
        "miss_margin_median": round(statistics.median(misses), 4),
        "miss_margin_max": round(max(misses), 4),
        # The honest test. Overlapping ranges mean no threshold on margin
        # separates the two, however different the medians look.
        "separated": min(hits) > max(misses),
    }


def to_json(index: Index, labels: LabelSet, outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    return {
        "submodel": labels.submodel,
        "embedding_model": index.model,
        "nodes_indexed": len(index.nodes),
        "enriched_from_concept_descriptions": index.enriched,
        "tiers": {t: tier_metrics([o for o in outcomes if o.tier == t]) for t in TIERS},
        "rejection_curve": rejection_curve(outcomes, (0.3, 0.4, 0.5, 0.6, 0.7)),
        "margin": margin_separation(outcomes),
        "outcomes": [asdict(o) for o in outcomes],
    }


def render(index: Index, labels: LabelSet, outcomes: Sequence[Outcome]) -> str:
    lines: List[str] = [
        f"{len(outcomes)} queries against {index.id_short or labels.submodel}",
        f"  {len(index.nodes)} nodes, {index.enriched} enriched from concept "
        f"descriptions, model {index.model}",
        "",
        f"{'tier':<14}{'n':>4}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}"
        f"{'med score':>11}{'med margin':>12}",
    ]
    for tier in TIERS:
        rows = [o for o in outcomes if o.tier == tier]
        if not rows:
            continue
        m = tier_metrics(rows)
        if tier == "negative":
            # recall is undefined without a target; the score is the whole point.
            lines.append(
                f"{tier:<14}{m['n']:>4}{'-':>10}{'-':>10}{'-':>8}"
                f"{m['median_top_score']:>11.4f}{m['median_margin']:>12.4f}"
            )
        else:
            lines.append(
                f"{tier:<14}{m['n']:>4}{m['recall@1']:>10.3f}{m['recall@5']:>10.3f}"
                f"{m['mrr']:>8.3f}{m['median_top_score']:>11.4f}{m['median_margin']:>12.4f}"
            )

    lines += ["", "rejecting negatives by score threshold:",
              f"  {'theta':>7}{'negatives refused':>20}{'correct hits lost':>20}"]
    for row in rejection_curve(outcomes, (0.3, 0.4, 0.5, 0.6, 0.7)):
        lines.append(
            f"  {row['theta']:>7.1f}"
            f"{row['negatives_rejected']:>13}/{row['negatives_total']:<6}"
            f"{row['correct_hits_lost']:>13}/{row['correct_hits_total']:<6}"
        )

    margin = margin_separation(outcomes)
    lines += ["", "margin (rank1 - rank2):"]
    if not margin.get("comparable"):
        lines.append("  not comparable -- need both answerable and negative cases")
    else:
        lines += [
            f"  answerable   median {margin['hit_margin_median']:.4f}  "
            f"min {margin['hit_margin_min']:.4f}",
            f"  unanswerable median {margin['miss_margin_median']:.4f}  "
            f"max {margin['miss_margin_max']:.4f}",
            f"  separated by a threshold: {'yes' if margin['separated'] else 'no'}",
        ]

    searched = [o.candidates for o in outcomes if o.candidates]
    if searched:
        lines += ["", f"candidates ranked per query: median "
                      f"{int(statistics.median(searched))}, max {max(searched)}"]

    misses = [o for o in outcomes if o.tier != "negative" and not o.top5]
    if misses:
        lines += ["", f"{len(misses)} answerable queries with no gold path in the top 5:"]
        for outcome in misses[:12]:
            lines.append(f"  [{outcome.tier[:4]}] {outcome.query[:44]:<44} "
                         f"got {outcome.ranked[0][:44] if outcome.ranked else '(nothing)'}")
    return "\n".join(lines)


def render_delta(before: Sequence[Outcome], after: Sequence[Outcome]) -> str:
    """What the structural filter bought, per tier and per case.

    Reported as a difference rather than as two tables because the claim being
    tested is comparative. A tier that does not move is as informative as one
    that does: the filter costing nothing where it cannot help is what makes it
    safe to apply by default.
    """
    paired_after = {o.case_id: o for o in after}
    lines = [
        "filtering: what it changed",
        "",
        f"{'tier':<14}{'n':>4}{'recall@1':>20}{'recall@5':>20}{'candidates':>20}",
    ]

    for tier in TIERS:
        rows = [o for o in before if o.tier == tier]
        if not rows:
            continue
        mirror = [paired_after[o.case_id] for o in rows]
        b, a = tier_metrics(rows), tier_metrics(mirror)
        cand = "{} -> {}".format(
            int(statistics.median(o.candidates or 0 for o in rows)),
            int(statistics.median(o.candidates or 0 for o in mirror)),
        )
        if tier == "negative":
            lines.append(f"{tier:<14}{len(rows):>4}{'-':>20}{'-':>20}{cand:>20}")
            continue
        r1 = "{:.3f} -> {:.3f}".format(b["recall@1"], a["recall@1"])
        r5 = "{:.3f} -> {:.3f}".format(b["recall@5"], a["recall@5"])
        lines.append(f"{tier:<14}{len(rows):>4}{r1:>20}{r5:>20}{cand:>20}")

    was_top1 = {o.case_id: o.top1 for o in before}
    fixed = [o for o in after if o.tier != "negative" and o.top1 and not was_top1[o.case_id]]
    broken = [
        o for o in before
        if o.tier != "negative" and o.top1 and not paired_after[o.case_id].top1
    ]

    lines += ["", f"moved to rank 1 by filtering: {len(fixed)}"]
    lines += [f"  + {o.query[:62]}" for o in fixed[:8]]
    lines += [f"lost from rank 1 by filtering: {len(broken)}"]
    lines += [f"  - {o.query[:62]}" for o in broken[:8]]

    neg_before = [o.scores[0] for o in before if o.tier == "negative" and o.scores]
    neg_after = [o.scores[0] for o in after if o.tier == "negative" and o.scores]
    if neg_before and neg_after:
        # Stated because it is the limit of the whole approach: a filter the
        # query happens to satisfy still leaves the wrong answers rankable, so
        # nothing here helps a question the repository cannot answer.
        lines += [
            "",
            "highest score on an unanswerable query: "
            "{:.4f} -> {:.4f}".format(max(neg_before), max(neg_after)),
            "  filtering cannot fix a negative -- rejection still needs a threshold",
        ]
    return "\n".join(lines)

