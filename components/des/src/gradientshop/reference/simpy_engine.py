"""Discrete ground-truth reference models in SimPy (proposal D1).

The DDES re-expresses a shopfloor model as a tensor recursion so gradients
flow; but that recursion is only *useful* if it reproduces the behaviour of a
bona-fide event-driven engine. SimPy is that independent oracle: a real
process-based DES with an event heap and no notion of differentiation.

These functions consume the *same* precomputed inter-arrival and service arrays
the DDES uses, so the two engines see an identical sample path and their
departure times must agree to floating-point tolerance. This is the validation
harness behind :func:`tests.test_reference`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import simpy

__all__ = ["simpy_single_server", "simpy_tandem_line"]


def simpy_single_server(interarrival: Sequence[float],
                        service: Sequence[float]) -> np.ndarray:
    """Event-driven FIFO single server. Returns per-job departure times ``[N]``."""
    inter = np.asarray(interarrival, dtype=float)
    srv = np.asarray(service, dtype=float)
    n = len(inter)
    depart = np.empty(n, dtype=float)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    def job(i: int, arrive_at: float):
        yield env.timeout(arrive_at - env.now)
        with server.request() as req:
            yield req
            yield env.timeout(srv[i])
            depart[i] = env.now

    arrivals = np.cumsum(inter)
    for i in range(n):
        env.process(job(i, float(arrivals[i])))
    env.run()
    return depart


def simpy_tandem_line(interarrival: Sequence[float],
                      service: np.ndarray) -> np.ndarray:
    """Event-driven FIFO tandem line. ``service`` is ``[N, M]``.

    Returns the ``[N, M]`` matrix of departure times from each station. FIFO is
    enforced per station by requesting each station's resource in job order,
    matching the DDES max-plus recursion.
    """
    srv = np.asarray(service, dtype=float)
    n, m = srv.shape
    inter = np.asarray(interarrival, dtype=float)
    depart = np.empty((n, m), dtype=float)

    env = simpy.Environment()
    stations = [simpy.Resource(env, capacity=1) for _ in range(m)]
    # Per-station FIFO: job i may seize station k only after job i-1 has seized
    # it, so the event heap cannot reorder equal-time contenders.
    order_gate = [[env.event() for _ in range(n)] for _ in range(m)]

    def job(i: int, arrive_at: float):
        yield env.timeout(arrive_at - env.now)
        for k in range(m):
            if i > 0:
                yield order_gate[k][i - 1]
            with stations[k].request() as req:
                yield req
                yield env.timeout(float(srv[i, k]))
                depart[i, k] = env.now
            order_gate[k][i].succeed()

    arrivals = np.cumsum(inter)
    for i in range(n):
        env.process(job(i, float(arrivals[i])))
    env.run()
    return depart
