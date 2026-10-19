"""Authoring source for generated demonstration PDFs; never used as a retrieval fallback."""

DOCUMENTS = [
    {"id": "hardware-layout", "title": "Two modules: pressing and turning over",
     "origin": "REFERENCE", "source": "Anleitung - Festo CP Lab.pptx, slides 4-6; Festo CP & Module.pptx, slide 1",
     "text": "The CP Lab has two connected application modules: a press and a gripper that turns "
     "the workpiece over by 180 degrees. Turning means reorientation, not lathe machining. "
     "Conveyors, carrier stoppers, RFID and position sensors support transport. The press "
     "application uses a Revolution Pi; the turning-over module uses a microcontroller. "
     "Both communicate with a Siemens PLC. This reference is not a current connectivity test."},
    {"id": "opc-evidence", "title": "OPC interfaces and limits of observations",
     "origin": "REFERENCE", "source": "Anleitung - Festo CP Lab.pptx, slides 8-9; a4s/machines/festo_cp_lab/model.py",
     "text": "The model names Turning PLC, Press PLC and Press RevPi OPC servers. Application "
     "flags include xStart, xReady, xBusy and xError; station collective fault xSF is a separate "
     "signal. Previous project tests found xError could stay false during a station fault. "
     "RevPi Pressure_1/2 and Flow_1/2 are uncalibrated raw readings in this integration. "
     "Never compare these raw counts to the fictional bar thresholds in the demo. "
     "A connection timeout means unavailable evidence, not a physical stop."},
    {"id": "readiness-reference", "title": "Readiness depends on application communication",
     "origin": "REFERENCE", "source": "Anleitung - Festo CP Lab.pptx, slides 17-18",
     "text": "The commissioning reference mentions compressed-air supply and application "
     "communication readiness. At power-up the PLC may initially lack a connection to the "
     "RevPi or microcontroller. A qualified operator must use approved site procedures. "
     "These observations are possible checks, not proof of an air leak and not permission "
     "for an agent to reset, start or change physical equipment."},
    {"id": "phone-housing-route", "title": "Demo process: smartphone housing",
     "origin": "DEMO_FICTION", "source": "A4S authored scenario v1; not manufacturer process data",
     "text": "For this exhibition, A4S-DEMO-PART represents a smartphone housing. A carrier "
     "enters a stamping/pressing step, then the turning-over module reorients it 180 degrees "
     "for a fictional downstream inspection handoff. Order A is affected; upcoming order B "
     "can be placed on the explicitly simulated A4S-BACKUP. Existing seeded orders are "
     "ACCEPTED/planned, not evidence of running batches, accepted finished parts or real capacity."},
    {"id": "pneumatic-leak-scenario", "title": "Demo fault: pressure loss / suspected pneumatic leak",
     "origin": "DEMO_FICTION", "source": "A4S authored scenario v1; all numeric values synthetic",
     "text": "The synthetic air-pressure signal starts at 6.0 bar. After 8 seconds it drops "
     "by 0.13 bar per second, down to 3.8 bar. Warning is below 5.2 bar; a fault is raised "
     "after pressure remains below 4.8 bar for 3 seconds. The fictional failure mechanism "
     "is a leaking pneumatic connector: available pressure falls, the press cannot remain "
     "ready, the synthetic collective fault becomes true, and order A requires a maintenance "
     "hold. Turning-over readiness stays true and conveyor speed is 0 in the fault state. "
     "This is a simulated hypothesis, not a measured diagnosis, calibrated threshold or PLC alarm."},
    {"id": "maintenance-playbook", "title": "Demo maintenance review and business response",
     "origin": "DEMO_FICTION", "source": "A4S scenario v1 / MaintenanceInstructions application profile based on IDTA 02018",
     "text": "On the simulated pressure alarm, retrieve technical context and preserve the "
     "signal history. Report suspected air leakage as the scenario hypothesis. A qualified "
     "operator assesses the pneumatic connection under approved site isolation and maintenance "
     "procedures; the agent does not issue physical repair instructions or operate the PLC. "
     "Read OpenMES orders, dry-run a blocking TOOL_FAILURE issue on A, then compare moving "
     "the not-yet-started B to the simulated backup. Each business mutation requires approval "
     "and read-back. The demo planning allowance for operator review is 30 minutes; it is "
     "an invented scheduling assumption, not a predicted or guaranteed repair time."},
    {"id": "recovery-boundaries", "title": "Recovery criteria and false conclusions to avoid",
     "origin": "PROJECT_CONTRACT", "source": "a4s/mesctl/domains; docs/A4S-Harness-架构拆解.md",
     "text": "MES owns business rules, status and persistence. AAS holds semantic references, "
     "not copied order status. PLC interlocks stay authoritative for physical safety. "
     "Closing an issue does not repair a leak. Resolving a blocking issue recalculates order "
     "status and may return an accepted no-batch order to PENDING; read order_after rather "
     "than blindly resuming. A write timeout has unknown effects; read before retrying. "
     "The monitor stops when the operator ends the simulation; this does not prove real recovery."},
]

FAULT_DOCUMENTS = [
    {"id": "pneumatic-diagnostic-matrix", "title": "Pneumatic supply alarm: pressure, flow and safety differentiation",
     "origin": "ENGINEERING_SCENARIO", "source": "AgenticMOM - Demo.pdf p.4; authored diagnostic profile for Festo runtime simulation",
     "text": "Scope: this engineering-authored guide applies to the runtime simulation profile for the selected Festo instance, not the physical 381 PLC alarm mapping. Press.Fault01 indicates low supply pressure for this profile. Read Press.PressureBar, Press.FlowLMin and Safety.EmergencyStop together. Normal reference is 6.0 bar and 12 litres/minute at idle. Below 4.8 bar with flow above 30 litres/minute and EmergencyStop false supports an open vent valve or pneumatic leakage hypothesis. It does not distinguish an intentionally opened valve from a damaged connector. Low pressure with low flow instead suggests insufficient upstream supply. EmergencyStop true requires a safety-stop assessment; do not interpret it as proof of leakage. Press.Ready can change independently of Turning.Ready. Alarm bits alone cannot establish root cause. Values are authored scenario engineering assumptions, not manufacturer limits."},
    {"id": "pneumatic-maintenance-plan", "title": "Pneumatic fault inspection and maintenance planning",
     "origin": "ENGINEERING_SCENARIO", "source": "AgenticMOM - Demo.pdf pp.5-6; authored maintenance planning procedure",
     "text": "Preserve alarm timestamps, pressure/flow values and the affected asset identity. Ask qualified onsite personnel to assess the air supply, valve state and connections using the site's approved isolation procedure. Do not remotely manipulate valves, reset interlocks or start machines. Reserve an initial 30-minute inspection allowance; actual duration and spare parts depend on inspection findings. Create a corrective Maintenance Event for the MES line referenced by ProductionContext, citing the fault event and this guide. Propose a start time relative to the current clock, not the presentation date. Preview and obtain approval before creating; verify the returned event ID by reading it back. An event is a business record, not proof of physical repair. Examine affected/upcoming orders and capacity; preview any feasible rescheduling separately and obtain approval, then verify schedule and change list. Never assume maintenance creation automatically blocks orders or changes scheduling."},
]
