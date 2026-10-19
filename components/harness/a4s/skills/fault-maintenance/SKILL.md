---
name: fault-maintenance
description: Investigate station fault evidence, retrieve asset knowledge, and prepare approved MES maintenance and scheduling actions
triggers: maintenance investigation, fault transition detected, maintenance plan, maintenance event, 故障分析, 维护计划
---

# Fault investigation and maintenance planning

Follow this procedure for a fault notification and subsequent approval/follow-up.
Do not infer root cause from an alarm bit or from the wording of the notification.
Equipment actions (start, reset, modes, pressure settings) are outside this workflow.

1. **Resolve the notified asset and evidence.** Use the exact AAS repository and
   asset ID in the notification. Read its referenced OperationalData,
   MaintenanceInstructions and ProductionContext, not similarly named templates.
   The read-only helper is:
   `python -m a4s.machines.festo_cp_lab.fault_context --asset ASSET_ID --endpoint AAS_URL`
   It returns source submodel IDs and contents, not a diagnosis. Alternatively use
   `aas shell show` and AAS read commands (inspect `aas --help`). Missing models or
   references are blockers to unsupported actions, not permission to invent IDs.
2. **Interpret the transition using those semantics.** Compare before/after,
   units and quality. A communication failure is not a device fault or recovery.
   Ready is not authorization to start. Never apply another instance's xSF01
   mapping. Check pressure, flow and emergency-stop evidence when defined.
3. **Retrieve documents before diagnosing.** Run
   `python -m a4s.machines.festo_cp_lab.knowledge --query 'SIGNAL NAMES AND OBSERVED SYMPTOMS' --limit 4`.
   Read returned PDF passages; cite filename/page in the assessment. Follow the
   retrieved document's scope and provenance. If evidence is insufficient, run a
   narrower query or ask the operator. Distinguish observations, likely cause,
   alternatives and confirmation needed. Do not present a hypothesis as a repair.
4. **Assess business impact.** Compare the ProductionContext MES endpoint with the
   configured trusted MES. Resolve its stable `MESLineCode` against current MES
   lines, then read current orders whose MES `line_id` matches that line. Also
   inspect the configured backup line for orders moved there. Use `mes auth whoami`,
   `mes line show CODE`, `mes order ls --line CODE`, and `mes order show ID` for
   live values. Never infer an order ID from the AAS or copy MES state into it.
   For the reverse lookup, compare the order's MES `line_id` with the resolved
   line and then match that line's `code` to an instance's `MESLineCode`.
   Inspect existing `mes maintenance ls --line ID` records for this event ID before proposing a new
   one. A previously approved/created event must not be created again.
5. **Prepare a concrete maintenance plan.** State the inspection scope, qualified
   operator requirement, priority, proposed start (current time with timezone),
   and planning allowance from the manual, labelled estimate. Query current time
   and MES timezone; never use historical demonstration dates. Preview:
   `mes maintenance create --title TITLE --event-type corrective --line LINE_ID --scheduled-at ISO_TIME --description DESCRIPTION --trace-id EVENT_ID --dry-run`
   Include AAS ID, event ID, evidence citations and expected inspection duration
   in description. Do not invent a dedicated duration field or employee ID.
   If ProductionContext contains ReplanningPolicy, inspect the integrated plan:
   `python -m a4s.machines.festo_cp_lab.replanning --asset ASSET_ID --endpoint AAS_URL`.
   Use its proposed maintenance_start and a 60-minute MES reservation. Explain
   that the manual's 30-minute inspection estimate is within that reservation;
   OpenMES defaults to 60 minutes when the API event has no scheduled_end_at.
   Run one MES command per shell call. Quote text arguments and use plain text
   without shell operators (especially semicolons, pipes, ampersands or newlines)
   inside descriptions: the conservative command gate rejects those too.
   Present the proposed action and obtain explicit approval. After approval,
   recheck current time: if an immediate-start proposal is now in the past,
   explain the delay and re-preview a new time for renewed approval. Do not
   silently change the approved payload or claim a past slot is an immediate plan.
   execute the exact agreed command without --dry-run through the existing gate;
   read back `mes maintenance show ID` and report the real ID/status. On uncertain
   timeout inspect existing records before retrying. Never say created on preview.
6. **Planning follows maintenance, not implicit side effects.** A Maintenance
   Event does not hold or reschedule an order. Read `mes schedule --help`, current
   capacity, order state, routing and compatibility. Propose a move/resize only
   when supported. Use `--dry-run`, separate approval, execution, readback and
   `mes schedule changes`. If no feasible option exists, report it. Do not force
   capacity conflicts, resume batches or resolve issues without authorization.
   Zero available hours do not prove missing calendar configuration. Report the
   returned capacity and inability to confirm a feasible slot, and identify which
   calendar/capacity data need checking. Do not say scheduling is unsupported.
   For a ProductionContext with ReplanningPolicy, this step is REQUIRED after
   maintenance creation: run the replanning helper again with --maintenance-id ID.
   Present the affected lots, current and proposed lines/times, and reasons.
   Obtain approval for the proposed moves, run each returned mes schedule move
   command separately with --trace-id EVENT_ID, and verify each result. Query
   mes schedule changes afterwards. Report partial progress if a move fails;
   do not replay successful moves. Do not finish at maintenance creation while
   replanning remains pending: continue to its proposal/approval or explain a
   concrete blocker. Apply the same process for both input sources.

Communicate the investigation as it happens, not only after a business write:
- With the first read calls, give one short assistant sentence stating the
  observed change and that you are checking asset information and manuals.
- After retrieving evidence, give a short assistant update alongside the next
  tool calls: observed pressure/flow changes, likely cause versus uncertainty,
  and the supporting manual page. Do not expose private reasoning or raw JSON.
- Before approval, explain the proposed inspection and production impact in
  plain language. Keep the ask_user question focused on the exact proposed action.
- Use finish.summary for the final user-facing outcome: what happened, likely
  cause, what was actually done, what remains blocked and the next operator step.
  Do not emit a long final assistant report and then repeat it in finish.
Keep trace IDs, execution IDs, hashes and chunk identifiers in tool evidence and
MES records, not normal chat prose. A maintenance ID and PDF/page citation are
useful to the operator. Respect simulation provenance, but do not repeatedly
prefix every finding with test/disclaimer wording. Never imply physical repair.
