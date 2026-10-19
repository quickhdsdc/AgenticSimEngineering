---
name: dpp-generation
description: Build a Digital Product Passport as a conformant AAS from supplier documents
triggers: DPP, product passport, datasheet, battery regulation, IDTA template
---

# Generating a Digital Product Passport

A DPP is an AAS whose submodels instantiate IDTA templates and satisfy the
regulation that applies to the product class. Building one is four decisions
and a lot of mechanical writing. Get the decisions right and the writing is a
loop.

## 1. Decide which templates apply

Read the source document first, then choose. Do not choose from the product
name alone -- a "battery module" datasheet may carry cell-level data that
belongs in a different template than the one the title suggests.

```bash
aas smt list                                    # every template in the repository
aas smt list --of-shell <collection-aas-id>     # only the ones a collection curates
aas smt show <template-id>                      # slot outline, one line per slot
```

Templates live as `kind: Template` submodels in the internal environment.
The semanticId is what identifies the template; the idShort is a label and
may differ between vendors.

`aas smt show` prints a cardinality against every slot -- `1..1` must be
filled, `0..1` may be omitted, `1..*` repeats. Read it before extracting:
knowing which slots are mandatory tells you what the document has to yield,
and a mandatory slot you cannot source is a finding for the report, not a
value to invent.

For an EU battery passport the usual set is Nameplate, TechnicalData,
CarbonFootprint, and the battery-specific template. Confirm against the
document rather than assuming.

## 2. Extract entities from the source

Extract to a file first, then read it. Two reasons: extraction is expensive
and you may need several passes over the same data, and a file survives a
context compression that a tool result does not.

```bash
a4s extract --pdf Attachments/datasheet.pdf --out temp/entities.json
a4s extract --table Attachments/specs.xlsx --out temp/entities.json
a4s extract --image Attachments/nameplate.jpg --intent "type plate" --out temp/entities.json
```

Each entity carries the value, the unit as its own field, and where in the
document it was found (`datasheet.pdf p.3`). The unit is separate for a reason:
step 4 checks it against the unit the target element declares, and a value of
`"24 VDC"` cannot be checked against anything. Provenance is not optional
either -- a passport whose values cannot be traced to a source is not evidence
of anything.

Check the count of entities without a value or without a unit before going on.
A page that yielded nothing usually means the render was too low-resolution;
raise `--dpi`.

## 3. Create the shell and its submodels

Create the shell once, then instantiate one submodel per template, attaching
as you go.

```bash
AAS=urn:example:aas:cell-4711
aas shell create "$AAS" --id-short Cell4711 --global-asset-id urn:example:asset:cell-4711

aas smt instantiate https://admin-shell.io/idta/SubmodelTemplate/DigitalNameplate/2/0 \
    --as urn:example:sm:cell-4711:nameplate --id-short Nameplate --attach-to "$AAS"
```

Instantiate rather than `aas sm create`. The template carries the semanticIds
that make each slot mean something, and reproducing them by hand is where
conformance is usually lost. `smt instantiate` copies the structure, switches
`kind` to `Instance`, and removes the template-authoring qualifiers that
AASd-129 forbids on an instance.

Three flags decide the shape of the result:

- **`--include-optional`** creates the `0..1` and `0..*` slots too. Off by
  default: an empty optional slot is not an error, but neither is it evidence,
  so it is not created speculatively. Turn it on when the document actually
  covers those slots.
- **`--seed-examples`** writes the template's `ExampleValue` qualifiers into
  the slots. Demo fixtures only. An example value traces to no source, and a
  passport field that traces to no source is exactly what a passport exists to
  rule out.
- **`--attach-to`** saves a separate `aas shell attach`. Use it.

Then confirm what you actually got, since the optional slots are not there:

```bash
aas sm tree urn:example:sm:cell-4711:nameplate
```

## 4. Populate, matching entity to element

This is the mechanical part and the part worth scripting. For each extracted
entity, decide which path it belongs at, then write it.

Do not decide the path by eye. An IDTA submodel names things the way the
metamodel spells them, not the way a datasheet does: "nominal capacity" is
stored as `RatedCapacity`, and the ConceptDescription is what says so. Index
once, then search per entity:

```bash
aas index "$SM"
aas find "nominal capacity" --in "$SM" --top-k 5
```

Each hit comes back inside its boundary — its parent collection and that
parent's real children. Use it. The failure this prevents is pairing a hit
from one collection with a field remembered from a different template, which
produces a path that looks plausible and does not exist.

```bash
aas el add "$SM" '' --type Property --id-short ManufacturerName \
    --value "$(jq -r '.manufacturer' temp/entities.json)" \
    --value-type xs:string \
    --semantic-id 0173-1#02-AAO677#002
```

A `MultiLanguageProperty` is not a string. Its `$value` is a list of
`{language: text}` objects, and BaSyx answers 500 -- not 400 -- when a plain
string arrives instead, so the status code will not tell you what went wrong.
Use `--lang`:

```bash
aas set "$SM" ManufacturerName 'Fraunhofer IPA@en' --lang
aas set "$SM" ManufacturerProductDesignation 'Battery cell@en|Batteriezelle@de' --lang
```

When a template calls for repeated structures, the container is a
`SubmodelElementList` and its children are addressed by position:

```bash
aas el add "$SM" '' --type SubmodelElementList --id-short Markings \
    --type-value-list-element SubmodelElementCollection
aas el add "$SM" Markings --type SubmodelElementCollection
aas set "$SM" 'Markings[0].MarkingName' CE
```

Write a loop rather than issuing thirty separate calls. One shell script is
one turn; thirty tool calls are thirty chances to lose track of where you
were.

### Let the matcher decide, then read what it decided

Deciding paths by hand is fine for three entities and hopeless for thirty. For
a whole extraction, match once and review the plan:

```bash
aas match --entities temp/entities.json --into "$SM" --out temp/plan.json
```

Every entity lands in one of three bands, on similarity to its best candidate:

- **at or above 0.8** — taken outright. Retrieval is reliable enough here that
  asking a model costs tokens and adds noise.
- **below 0.5** — never written. No candidate this far away is worth arguing
  over; the entity is either created in a collection that can host it, or
  reported unplaced.
- **in between** — handed to a model, which sees each candidate inside its
  parent collection. This is the only genuinely uncertain band, and the reason
  the plan exists as a file: the decision is recorded with its reason.

Read the plan before applying it. Two things to look for:

- **`!! unit mismatch`** — the document's unit disagrees with the element's.
  This is the trap: 8 with unit `a` written into a property declared in
  `month` is eight months, and nothing downstream will catch it.
- **`create`** rather than `write` — no existing element matched, so a new one
  would be added. Check that the chosen collection is really where it belongs.

Then apply:

```bash
aas populate --plan temp/plan.json --dry-run     # see it first
aas populate --plan temp/plan.json --skip-warnings
```

`--skip-warnings` leaves the flagged slots empty rather than wrong. An empty
mandatory slot is a finding for the report; a slot filled from a mismatched
unit is a false passport. Resolve those by hand, or fix the unit in the
entities file and match again.

If no entity reaches 0.8 and the command says so, do not shrug it off — it
means every decision went to arbitration, including ones retrieval already had
right. Report it.

### On entities with no home

Do not invent an element to hold something the template does not cover. Record
it in the report as unmapped. A passport with a missing field is incomplete; a
passport with a fabricated field is false.

## 5. Validate before you finish

```bash
aas validate "$SM"
```

Fix every finding. `AASd-002` (illegal idShort) and `AASd-120` (an idShort on
a list child) are the two that make a submodel unreadable to strict consumers
while still looking fine in a browser.

## 6. Check completeness, then report

`aas validate` checks the metamodel; it cannot tell you whether the passport is
*complete*. Cardinality is declared by qualifiers that AASd-129 forbids on an
instance, so the instance no longer knows which of its slots were mandatory.
Completeness is a comparison against the template it came from:

```bash
aas smt diff <template-id> "$SM"                     # unfilled mandatory slots
aas smt diff <template-id> "$SM" --include-optional  # and the optional ones
```

It separates `missing` (no element — needs `aas el add`) from `empty` (element
present, no value — needs `aas set`), and carries each slot's semanticId so a
gap can be matched against a source document later.

Then state: which templates were instantiated, how many elements were written,
which mandatory slots remain unfilled and why the document could not supply
them, which extracted entities went unmapped, and any value you had to judge
rather than read directly.

A gap is a finding, not a failure. A passport with a missing field is
incomplete; a passport with a fabricated field is false.
