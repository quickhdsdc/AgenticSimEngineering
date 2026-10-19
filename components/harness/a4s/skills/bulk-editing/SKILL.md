---
name: bulk-editing
description: Read, filter and write or delete many submodel elements in one pass instead of one at a time
triggers: bulk, batch, prune, delete every, delete all, remove every, remove all, empty slots, for every row, one at a time, many elements
---

# Editing many elements at once

The `aas` command is a program, so several elements can be handled in one
shell call. Doing them one at a time still works and costs a round trip each.

This procedure exists because of a measurement, not a preference. On a task
that deleted five empty slots from a submodel, an agent with this command
surface took **8 steps and 21,566 input tokens** reading and deleting one
element at a time, against **3 steps and 7,808** for an agent with one tool per
operation that issued its calls in parallel. The surface made a pipeline
possible; nothing made it happen.

## The shape

Three stages, one shell call each, and often the first two collapse into one:

```
read the structure  →  decide which elements  →  act on all of them
   aas sm tree           jq / grep / a loop        aas set | aas el rm
    --json
```

## 1. Read once, in JSON

Never read elements one at a time to find out what is there. `--json` is on the
top-level command, before the verb.

```bash
aas --json sm tree <submodel-id> --depth 5      # structure, no values
aas --json el ls <submodel-id> '' --depth 3     # children with their values
```

## 2. Filter with the shell, not with turns

Every candidate you evaluate in a separate tool call is a round trip you did
not need. `jq` decides in the same call that fetched the data.

```bash
# every Property at the root whose value is empty
aas --json el ls <sm> '' --depth 1 \
  | jq -r '.elements[] | select(.modelType=="Property" and (.value // "")=="") | .idShort'
```

If the output shape is not what you expected, print it once and look, rather
than guessing twice.

## 3. Act on all of them in one call

```bash
# delete every empty Property
aas --json el ls <sm> '' --depth 1 \
  | jq -r '.elements[] | select(.modelType=="Property" and (.value // "")=="") | .idShort' \
  | while read -r name; do aas el rm <sm> "$name"; done

# write thirty values from a CSV
tail -n +2 values.csv | while IFS=, read -r name value; do
    aas el add <sm> '' --type Property --id-short "$name" \
        --value "$value" --value-type xs:double
done
```

Then verify in one more call rather than reading each element back:

```bash
aas --json el ls <sm> '' --depth 1 | jq '[.elements[] | select(.value != null)] | length'
```

## Deleting from a SubmodelElementList: go backwards

Deleting `Items[1]` **renumbers everything after it** -- this is measured, not
assumed, and it is pinned in `tests/integration/test_list_semantics.py`. The
loop an agent writes first is wrong:

```bash
for i in 0 1 2; do aas el rm <sm> "Items[$i]"; done   # WRONG: removes 0, 2, 4
for i in 2 1 0; do aas el rm <sm> "Items[$i]"; done   # right: highest index first
```

Elements addressed by idShort are unaffected -- only list positions shift.

## What not to batch

**Anything that moves a machine.** `aas op invoke` in a loop asks an operator
to approve each invocation separately, or trips the gate repeatedly. Invoke
Operations one at a time and read the result of each.

**Anything you cannot verify in one call afterwards.** If checking the result
costs one call per element, the pipeline saved nothing and you have lost the
ability to say which write failed.

## Check before a bulk create

`aas el add` refuses an idShort that already exists among its siblings, because
a duplicate makes *both* copies unreadable by path. In a loop that means the
run stops partway. List what is there first:

```bash
aas --json el ls <sm> '' --depth 1 | jq -r '.elements[].idShort'
```

If a submodel is already holding duplicates -- from an import, or from another
client -- `aas validate` reports them as AASd-022 and `aas sm repair --apply`
removes them.
