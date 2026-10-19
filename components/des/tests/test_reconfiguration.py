"""Phase 5.5 — the slow scale, and whether it is genuinely sequential.

``PLAN.md`` asserts that re-solving the global model every period churns the
configuration, and that reinforcement learning therefore has an edge at T2.
That assertion is what this phase tests rather than assumes, so the tests come
in two kinds:

* **the environment is a real MDP** — events land before the action is scored,
  the cost depends on the state, and the accounting closes;
* **the claim is measured** — a policy that charges itself for moving beats one
  that does not, and by how much is a number rather than a story.
"""

from __future__ import annotations

import pytest

from intralogistics.instances import build_network_instance
from intralogistics.reconfiguration import (
    Episode,
    Event,
    FrozenPolicy,
    MyopicReoptimise,
    ReconfigurationAware,
    ReconfigurationCost,
    SupplyChainMDP,
    _place_newcomers,
    _with_demand_shift,
    _with_new_item,
    build_episode,
    run_episode,
)
from intralogistics.heuristic import PlanWorld, SupplyPlan
from opsworld.world import Capability, SequentialWorldModel


def _base(seed: int = 1):
    return build_network_instance(n_items=4, n_resources=2, n_forward=2, seed=seed)


# ----------------------------------------------------------------------
# the world can change
# ----------------------------------------------------------------------
def test_a_newcomer_is_a_real_item_everywhere():
    """Instance construction validates its own tables, so a newcomer missing
    from any of them would be rejected rather than silently mis-priced."""
    base = _base()
    grown = _with_new_item(base, "NEW1", base.I[0])

    assert grown.I == [*base.I, "NEW1"]
    assert any("NEW1" in grown.items_of_product(p) for p in grown.P)
    for r in grown.R:
        assert grown.has_param("demand_item_resource", "NEW1", r)
    for b in grown.B:
        assert grown.has_param("carrier_capacity", "NEW1", b)
    for l1, l2 in grown.U_edges:
        assert grown.has_param("relocation_cost", "NEW1", l1, l2,
                               grown.B[0], grown.B[0])
    # the base instance is untouched
    assert "NEW1" not in base.I


def test_a_demand_shift_moves_inventory_with_it():
    """Scaling demand while holding average inventory fixed would quietly be a
    change of turnover ratio -- a different scenario, and a confound."""
    base = _base()
    item = base.I[0]
    shifted = _with_demand_shift(base, {item: 2.0})

    for r in base.R:
        assert shifted.param("demand_item_resource", item, r) == pytest.approx(
            2.0 * base.param("demand_item_resource", item, r)
        )
    assert shifted.param("avg_inventory_total", item) == pytest.approx(
        2.0 * base.param("avg_inventory_total", item)
    )
    # untouched items really are untouched
    other = base.I[1]
    assert shifted.param("avg_inventory_total", other) == base.param(
        "avg_inventory_total", other
    )


# ----------------------------------------------------------------------
# what a change costs
# ----------------------------------------------------------------------
def test_doing_nothing_is_free():
    base = _base()
    plan = PlanWorld(base).decision_space().initial()
    assert ReconfigurationCost().between(plan, plan.copy(), base.I) == 0.0


def test_each_kind_of_change_is_charged_once():
    base = _base()
    costs = ReconfigurationCost(
        open_location=40.0, close_location=15.0, change_pick=8.0, change_carrier=25.0
    )
    item, other = base.I[0], base.I[1]
    warehouse = base.L_prod[0]
    elsewhere = base.L_prod[-1] if len(base.L_prod) > 1 else base.L_log[0]
    resource = base.R[0]

    before = SupplyPlan(
        pick={(item, resource): warehouse},
        carrier={(item, warehouse): base.B[0]},
    )
    # moving the pick opens one location, closes another, and changes the pick
    after = SupplyPlan(
        pick={(item, resource): elsewhere},
        carrier={(item, elsewhere): base.B[0]},
    )
    assert costs.between(before, after, [item, other]) == pytest.approx(
        40.0 + 15.0 + 8.0
    )

    # swapping only the carrier costs only that
    swapped = SupplyPlan(
        pick={(item, resource): warehouse},
        carrier={(item, warehouse): base.B[1]},
    )
    assert costs.between(before, swapped, [item, other]) == pytest.approx(25.0)


# ----------------------------------------------------------------------
# it is an MDP
# ----------------------------------------------------------------------
def test_the_mdp_declares_and_satisfies_c1():
    """The first sequential world model on this side of the framework."""
    mdp = SupplyChainMDP(build_episode(_base(), horizon=3, seed=0))
    assert mdp.capabilities() == {Capability.EVALUATE, Capability.SEQUENTIAL}
    assert isinstance(mdp, SequentialWorldModel)


def test_events_land_before_the_action_is_scored():
    """Acting before the event would be clairvoyance -- and would hand the
    lookahead policy an advantage it is not entitled to."""
    base = _base()
    episode = Episode(
        base=base,
        events=(Event(1, "new_item", ("NEW1", base.I[0])),),
        horizon=2,
    )
    mdp = SupplyChainMDP(episode)
    state = mdp.reset()
    assert state.period == 0
    assert "NEW1" not in state.instance.I

    upcoming = mdp.advance_world(state)
    assert "NEW1" in upcoming.I

    action = _place_newcomers(upcoming, state.plan)
    nxt, reward, done = mdp.step(state, action)
    assert nxt.period == 1
    assert "NEW1" in nxt.instance.I
    assert reward < 0.0  # cost, reported as negative reward
    assert done is False


def test_placing_newcomers_leaves_no_demand_unserved():
    base = _base()
    grown = _with_new_item(base, "NEW1", base.I[0])
    plan = _place_newcomers(grown, PlanWorld(base).decision_space().initial())
    for r in grown.R:
        if float(grown.param("demand_item_resource", "NEW1", r)) > 0:
            assert ("NEW1", r) in plan.pick


def test_episode_accounting_closes():
    """Total must be the two components and nothing else, so a policy cannot
    hide churn in the operating figure."""
    episode = build_episode(_base(), horizon=4, n_new_items=1,
                            shift_strength=0.3, seed=1)
    result = run_episode(SupplyChainMDP(episode), FrozenPolicy())
    assert len(result.operating) == episode.horizon
    assert len(result.reconfiguration) == episode.horizon
    assert result.total == pytest.approx(
        sum(result.operating) + sum(result.reconfiguration)
    )


# ----------------------------------------------------------------------
# the claim under test
# ----------------------------------------------------------------------
def test_standing_still_is_not_an_option():
    """The scenario has to be genuinely dynamic for any of this to matter: a
    configuration that never moves must fall over as the world shifts."""
    episode = build_episode(_base(), horizon=8, n_new_items=2,
                            shift_strength=0.5, seed=0)
    result = run_episode(SupplyChainMDP(episode), FrozenPolicy())
    assert result.infeasible_periods > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_bad_warm_start_does_not_trap_the_aware_policy(seed):
    """Regression: warm-starting from the standing plan is what makes this
    policy calm, but a plan an event has just made infeasible can trap the
    local search where a fresh construction would not. Measured at horizon 4
    before the fix: the configuration stayed infeasible and the episode cost a
    penalty, i.e. the policy lost to the one it exists to beat purely because
    of where it began.
    """
    episode = build_episode(_base(seed=seed), horizon=4, n_new_items=2,
                            shift_strength=0.3, seed=seed)
    aware = run_episode(SupplyChainMDP(episode), ReconfigurationAware())
    myopic = run_episode(SupplyChainMDP(episode), MyopicReoptimise())

    assert aware.infeasible_periods <= myopic.infeasible_periods
    assert aware.total <= myopic.total


def test_charging_yourself_for_moving_beats_ignoring_the_cost():
    """The measured version of ``PLAN.md``'s churn claim.

    Same environment, same search, same budget; the only difference is whether
    the move cost is inside the objective. If this ever ties, the sequential
    story is wrong and Phase 5.5 should shrink rather than grow an RL agent.
    """
    episode = build_episode(_base(seed=1), horizon=8, n_new_items=2,
                            shift_strength=0.3, seed=1)

    myopic = run_episode(SupplyChainMDP(episode), MyopicReoptimise())
    aware = run_episode(SupplyChainMDP(episode), ReconfigurationAware())

    assert myopic.infeasible_periods == 0
    assert aware.infeasible_periods == 0
    assert aware.total < myopic.total
    assert sum(aware.reconfiguration) < sum(myopic.reconfiguration)
    assert aware.locations_touched < myopic.locations_touched
    # and it does not buy the calm by running a much worse configuration
    assert sum(aware.operating) <= 1.05 * sum(myopic.operating)
