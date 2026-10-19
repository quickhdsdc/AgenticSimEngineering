# Delegated operator: FactoryFlow modeling pilot

Run the FactoryFlow case with the existing A4S Harness. Author both models
from the original S1/S2 descriptions returned by prepare, then score once.
No live factory data are available. Time is a reference simulation time unit,
not an established equipment unit mapping.

Submission: JSON object with task IDs S1 and S2. Each model has `nodes` and
`buffers`. Nodes have `id`, `kind` (source/machine/sink); source has
`inter_arrival_time`; machine has `processing_delay` and integer
`work_capacity`; sink has no extra fields. Buffers have `id`, `from`, `to`,
integer `capacity`. Defaults: machine work_capacity=1, FIFO buffers with
zero transfer delay; source nonblocking, machines blocking, deterministic
arrival/processing delays. These defaults are delegated reference context.

Keep upstream identifiers. Clarification for the S2 description:
the sink ID is `sink`, and its final buffer is `B_2_sink` (the prose calls it
B_2_3). This explicit source discrepancy follows the supplied upstream model.
All nodes start without items. Do not invent initial WIP or measurements.

Scoring compares topology/parameters and received-count/cumulative-cycle-time
at 25 and 50 time units against the retained original native reference models.
This small deterministic pilot does not test routing, reliability or optimization.
