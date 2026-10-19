"""Two ablations over retrieval: what goes into the descriptor, and what
counts as the same element.

Both questions were raised the same way -- by someone asking why the current
answer was the answer -- and neither had evidence behind it.

**How much of the path belongs in the descriptor.** One ancestor level was
chosen when the corpus was one submodel of 46 elements. Across 3,789 elements
in 116 submodels the same short name recurs under deeper structures: seven
`Payload` properties differing only by which link they hang from, and one level
cannot separate them. Whether more levels help, or dilute the name with
boilerplate that every sibling shares, is measurable and was not measured.

**What makes two hits the same hit.** Deduplication currently keys on
`(path, semanticId, modelType)`, and `semanticId` is a bad thing to depend on:
31% of elements here carry none at all, and 281 of those that do carry an
ECLASS IRDI -- `0173-1#02-AAO677#002` -- which is an opaque identifier with no
semantics in the string. An element without one would never collapse, so the
duplicate rows the key exists to remove would come back for exactly the
elements least able to afford it.

The alternatives are not obviously better either. `idShort` alone is far too
coarse: 1,035 distinct values over 3,789 elements, and `X` occurs 72 times.
The full path is stronger (1,465 distinct, worst case 14) and is what an agent
would read anyway. So: measure.

Re-embedding is what makes this cost anything -- roughly 3 cents and 20 seconds
per descriptor variant over the whole repository. The dedupe ablation is free,
because the key is applied at search time over vectors that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..aasctl.index import Index, Node, embed
from .retrieval import LabelSet, Outcome, evaluate, tier_metrics


@dataclass
class Variant:
    """One configuration, its label, and how it differs from the default."""

    name: str
    note: str = ""


# ---------------------------------------------------------------------------
# descriptor: how much ancestry to embed
# ---------------------------------------------------------------------------

DESCRIPTOR_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("no parent", {"parent_depth": 0}),
    ("1 level (current)", {"parent_depth": 1}),
    ("2 levels", {"parent_depth": 2}),
    ("full path", {"parent_depth": -1}),
    ("full path + submodel", {"parent_depth": -1, "include_submodel": True}),
    # `--with-values` has been a flag since the module was written and nobody
    # has ever measured what it buys. The `Link00` against `Link06` failure is
    # an ordinal problem the schema half of the descriptor cannot express, and
    # the stored value is the only field that differs between those seven
    # elements -- so if anything separates them it is this.
    ("1 level + values", {"parent_depth": 1, "with_values": True}),
    ("full path + values", {"parent_depth": -1, "with_values": True}),
]


def reembed(index: Index, **descriptor_kwargs: Any) -> Index:
    """A copy of `index` whose vectors come from a different descriptor.

    The nodes are shared rather than copied: nothing here mutates them, and a
    deep copy of 3,789 dataclasses per variant is real time spent to protect
    against a bug that would show up immediately anyway.
    """
    import dataclasses

    kwargs = dict(descriptor_kwargs)
    with_values = kwargs.pop("with_values", index.with_values)
    texts = [n.descriptor(with_values, **kwargs) for n in index.nodes]
    return dataclasses.replace(index, vectors=embed(texts, model=index.model))


def descriptor_ablation(
    index: Index,
    labels: LabelSet,
    *,
    top_k: int = 5,
    variants: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
    on_variant: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[Outcome]]:
    results: Dict[str, List[Outcome]] = {}
    for name, kwargs in variants or DESCRIPTOR_VARIANTS:
        if on_variant is not None:
            on_variant(name)
        rebuilt = reembed(index, **kwargs)
        results[name] = evaluate(rebuilt, labels, top_k=top_k, use_filters=False)
    return results


# ---------------------------------------------------------------------------
# deduplication: what counts as the same element
# ---------------------------------------------------------------------------

# Each returns the key two nodes must share to be collapsed into one hit.
DEDUPE_KEYS: Dict[str, Callable[[Node], Any]] = {
    "none": lambda n: object(),                       # never collapses
    "idShort": lambda n: (n.id_short,),
    "path": lambda n: (n.path,),
    "path + type": lambda n: (n.path, n.model_type),
    "path + semanticId + type": lambda n: (n.path, n.semantic_id, n.model_type),
    "semanticId only": lambda n: (n.semantic_id,) if n.semantic_id else object(),
}


def dedupe_ablation(
    index: Index,
    labels: LabelSet,
    *,
    top_k: int = 5,
    keys: Optional[Dict[str, Callable[[Node], Any]]] = None,
) -> Dict[str, List[Outcome]]:
    """Score each key over one set of vectors. No re-embedding, so it is free.

    `semanticId only` is included because it is the tempting simplification and
    it is worth seeing it fail: an element with no semanticId gets a key that
    can never match, so 31% of this corpus silently stops collapsing.
    """
    import a4s.aasctl.index as index_mod

    original = index_mod._dedupe_key
    results: Dict[str, List[Outcome]] = {}
    try:
        for name, key in (keys or DEDUPE_KEYS).items():
            index_mod._dedupe_key = key
            results[name] = evaluate(index, labels, top_k=top_k, use_filters=False)
    finally:
        index_mod._dedupe_key = original
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def render_ablation(results: Dict[str, List[Outcome]], title: str) -> str:
    """One row per variant, tiers as columns, so a regression is visible.

    Reported per tier rather than aggregated because the tiers do not move
    together: a change that helps paraphrase can cost multilingual, and an
    average would hide the trade rather than surface it.
    """
    tiers = ("paraphrase", "multilingual", "verbatim")
    lines = [
        title,
        "",
        f"{'variant':<24}" + "".join(f"{t[:12]:>26}" for t in tiers) + f"{'negatives':>12}",
        f"{'':<24}" + "".join(f"{'r@1 / r@5':>26}" for t in tiers) + f"{'max score':>12}",
    ]
    for name, outcomes in results.items():
        row = f"{name:<24}"
        for tier in tiers:
            rows = [o for o in outcomes if o.tier == tier]
            if not rows:
                row += f"{'-':>26}"
                continue
            m = tier_metrics(rows)
            cell = "{:.3f} / {:.3f}".format(m["recall@1"], m["recall@5"])
            row += f"{cell:>26}"
        negatives = [o.scores[0] for o in outcomes if o.tier == "negative" and o.scores]
        row += "{:>12}".format("{:.4f}".format(max(negatives)) if negatives else "-")
        lines.append(row)

    lines += ["", "negatives: the highest score any unanswerable query reached. "
                  "Lower is better -- it is the headroom a rejection threshold has."]
    return "\n".join(lines)
