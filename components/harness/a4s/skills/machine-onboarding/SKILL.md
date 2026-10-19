---
name: machine-onboarding
description: Give a new machine an AAS with a Capability submodel so it can be orchestrated
triggers: onboard, new machine, register asset, capability, delegation, OPC UA
---

# Onboarding a machine

A machine the orchestrator cannot see is a machine it cannot schedule. Making
it visible means giving it a shell, describing what it can do as Operations,
and binding each Operation to the endpoint that performs it.

The mapping to worry about, because it is the one that makes delegation work:

| Concept | AAS construct |
|---|---|
| What the machine can do | an Operation in its Capability submodel |
| Where the request goes | that Operation's `invocationDelegation` qualifier |
| What it needs to be told | the Operation's input variables |

## 1. Confirm the endpoint answers before modelling it

Model second, verify first. An Operation bound to a dead URL fails at the
worst moment -- during a production run, not during onboarding.

```bash
curl -sS -m 5 http://edge-389.plant.local:5040/health
```

If the machine speaks OPC UA rather than HTTP, the delegation target is its
proxy, not the OPC UA endpoint. Find the proxy first.

## 2. Create the shell

```bash
AAS=urn:plant:aas:printer-389
aas shell create "$AAS" --id-short Printer389 \
    --global-asset-id urn:plant:asset:printer-389 \
    --description "LPBF metal printer, cell 3"
```

## 3. Nameplate first, capability second

Nameplate is what a human uses to confirm they are looking at the right
machine. It is cheap and it prevents a whole class of mistake.

```bash
SM_NP=urn:plant:sm:printer-389:nameplate
aas sm create "$SM_NP" --id-short Nameplate \
    --semantic-id https://admin-shell.io/zvei/nameplate/2/0/Nameplate \
    --attach-to "$AAS"
aas el add "$SM_NP" '' --type Property --id-short ManufacturerName --value "EOS"
aas el add "$SM_NP" '' --type Property --id-short SerialNumber --value "M290-4471"
```

## 4. Describe the capabilities

One Operation per thing the machine can be asked to do. Name the inputs the
service actually requires -- an Operation whose inputs do not match its
endpoint is worse than no Operation, because the orchestrator will try it.

```bash
SM_CAP=urn:plant:sm:printer-389:capability
aas sm create "$SM_CAP" --id-short Capability --attach-to "$AAS"

aas el add "$SM_CAP" '' --type Operation --id-short PrintJob \
    --delegate-to http://edge-389.plant.local:5040/print \
    --input jobId=xs:string \
    --input material=xs:string \
    --input layerHeight=xs:double \
    --output estimatedDuration=xs:double
```

If the machine runs its own agent, add the agent endpoint as a property so the
orchestrator can delegate the whole task rather than a single call:

```bash
aas el add "$SM_CAP" '' --type Property --id-short AgentEndpoint \
    --value http://edge-389.plant.local:5041/sse --value-type xs:anyURI
```

## 5. Verify the loop closes

Discovery and invocation, in that order. `op ls` is what the orchestrator
itself will run, so if it does not show the Operation, nothing else matters.

```bash
aas op ls "$SM_CAP"
aas op invoke "$SM_CAP" PrintJob --arg jobId=probe-1 --arg material=IN718 \
    --arg layerHeight=0.03 --dry-run
```

Check the dry run resolves to the endpoint you expect. Only then invoke for
real, and only with a job the operator has agreed to run.

## 6. Validate and report

```bash
aas validate "$SM_CAP"
```

Report the shell id, the operations registered with their delegation targets,
and anything the machine can do that you could not model -- an unmodelled
capability is invisible to scheduling, and someone needs to know.
