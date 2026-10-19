"""The HTTP surface the existing web UI talks to.

The harness is a command surface because its principal caller is a language
model: one that re-reads the interface on every call, pays tokens to do it, and
has a nonzero error rate per token. That argument is about **LLM ergonomics**
and it does not generalise. A browser is not an LLM and this server is not an
agent, so it imports the library directly rather than composing command lines
and parsing stdout -- a programmer reads the interface once and encodes what
they learned, which is exactly the cost model `aas` exists to avoid paying.

The one exception is `/ws/chat`, which runs a real `AgentLoop`. Inside it the
agent reaches the repository the way it always does: `shell`, then `aas`.

**What this is a reimplementation of, and what it is not.** The predecessor's
`app/api.py` is 2,785 lines. The Vue frontend, now vendored at `frontend/`,
calls 25 of its routes, and most map onto verbs that already exist here:

    /api/aas-list                -> client.list_shells
    /api/parse-aas/{id}          -> index.walk
    /api/aas-properties/{id}     -> the same walk, grouped by parent
    /api/templates               -> submodels with kind=Template
    /api/instantiate-aas         -> smt.instantiate
    /api/upload + extraction     -> extract
    /api/matching-results        -> match
    /api/save-matching-results   -> match.apply

`aas_routes` reads and `write_routes` writes, and the split is not cosmetic:
the writing half runs no approval gate -- decided 2026-08-27, that gate is a
control on the agent -- and appends every mutation to the ledger instead.

So this is an adapter, not a port. Where the predecessor cached a NetworkX
graph to a temp file and served that, this walks the submodel -- the graph file
was a cache with no invalidation, and `/api/aas-properties` returning 404
because nobody had run `parse-aas` first is a failure mode worth not
reproducing.
"""

from __future__ import annotations

from .app import build_app, main

__all__ = ["build_app", "main"]
