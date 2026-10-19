<template>
  <div class="extraction-stage">
    <h2 class="section-title">Data Extraction</h2>

    <div v-if="data.isExtracting" class="progress-indicator">
      <div class="spinner"></div>
      <h3>Extracting data...</h3>
      
    </div>

    <div v-else-if="data.extractedEntities.length > 0">
      <div class="alert alert-success">
        Successfully extracted {{ data.extractedEntities.length }} properties from the data source.
        Review and select the properties you want to use for matching.
      </div>

      <!-- A datasheet describes a product family; this AAS describes one
           asset. Until a configuration is chosen, several values compete for
           every slot and the matcher drops all of them. -->
      <div v-if="configurations.length > 1" class="config-picker">
        <h4>This document describes {{ configurations.length }} configurations</h4>
        <p>
          The values below do not all describe the same machine. Choose the one
          this asset is; the others stay in the table, unticked, and are not
          matched.
        </p>
        <div class="config-options">
          <button
            v-for="name in configurations"
            :key="name"
            class="config-option"
            :class="{ 'config-option-active': data.chosenConfiguration === name }"
            @click="chooseConfiguration(name)"
          >
            {{ name }}
            <span class="config-count">{{ countFor(name) }} values</span>
          </button>
        </div>
        <p v-if="data.chosenConfiguration" class="config-chosen">
          Keeping <strong>{{ universalCount }}</strong> values that apply to every
          configuration and <strong>{{ countFor(data.chosenConfiguration) }}</strong>
          specific to <strong>{{ data.chosenConfiguration }}</strong>.
          <strong>{{ data.extractedEntities.length - universalCount - countFor(data.chosenConfiguration) }}</strong>
          describe other configurations and are not selected.
        </p>
      </div>

      <!-- What the extraction could not do cleanly. Reported rather than
           swallowed: a value silently dropped is the failure this prevents. -->
      <div v-if="data.extractionWarnings.length > 0" class="extraction-warnings">
        <h4>
          {{ data.extractionWarnings.length }} things to check before matching
          <button class="warn-toggle" @click="warningsOpen = !warningsOpen">
            {{ warningsOpen ? 'hide' : 'show' }}
          </button>
        </h4>
        <ul v-if="warningsOpen">
          <li v-for="(line, i) in data.extractionWarnings" :key="i">{{ line }}</li>
        </ul>
      </div>

      <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #666; font-size: 14px;">
          {{ selectedCount }} of {{ data.extractedEntities.length }} properties selected
        </span>
        <div>
          <button class="btn btn-secondary btn-sm" @click="selectAll" style="margin-right: 8px;">
            Select All
          </button>
          <button class="btn btn-secondary btn-sm" @click="deselectAll">
            Deselect All
          </button>
        </div>
      </div>

      <HelpTip text="Click row to select/deselect Double-click cell to edit" />

      <table class="entity-table">
        <thead>
          <tr>
            <th style="width: 8%">
              <input
                type="checkbox"
                class="row-checkbox"
                :checked="allSelected"
                @change="toggleAll"
                title="Select/Deselect All"
              />
            </th>
            <th style="width: 22%">Property Name</th>
            <th style="width: 18%">Value</th>
            <th v-if="configurations.length > 0" style="width: 16%">Configuration</th>
            <th style="width: 36%">Description</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, index) in data.extractedEntities"
            :key="index"
            :class="{ 'row-deselected': (row as SelectableEntity).selected === false }"
            @click="toggleEntity(index)"
            style="cursor: pointer;"
          >
            <td @click.stop>
              <input
                type="checkbox"
                class="row-checkbox"
                :checked="(row as SelectableEntity).selected !== false"
                @change="toggleEntity(index)"
              />
            </td>
            <!-- Name cell. The field is `entity`: that is what the extractor
                 writes, what `Entity` declares, and what the matcher reads
                 back. This asked for `name`, which nothing ever set, so the
                 column was blank and editing it wrote to a field the matcher
                 ignores. -->
            <td @click.stop @dblclick="startEditing(index, 'entity')">
              <input
                v-if="isEditing(index, 'entity')"
                ref="editInput"
                type="text"
                v-model="editValue"
                class="edit-input"
                @blur="saveEdit"
                @keyup.enter="saveEdit"
                @keyup.escape="cancelEdit"
                @click.stop
              />
              <strong v-else>{{ row.entity }}</strong>
            </td>
            <!-- Value cell -->
            <td @click.stop @dblclick="startEditing(index, 'value')">
              <input
                v-if="isEditing(index, 'value')"
                ref="editInput"
                type="text"
                v-model="editValue"
                class="edit-input"
                @blur="saveEdit"
                @keyup.enter="saveEdit"
                @keyup.escape="cancelEdit"
                @click.stop
              />
              <span v-else>{{ row.value }}</span>
            </td>
            <!-- Which configuration this value belongs to. Blank means it
                 holds for every configuration, which is the common case. -->
            <td v-if="configurations.length > 0" style="color: #666; font-size: 13px">
              <span v-if="row.variant" class="variant-tag">{{ row.variant }}</span>
              <span v-else style="color: #aaa">all</span>
            </td>
            <!-- Description cell -->
            <td @click.stop @dblclick="startEditing(index, 'description')" style="color: #666">
              <input
                v-if="isEditing(index, 'description')"
                ref="editInput"
                type="text"
                v-model="editValue"
                class="edit-input"
                @blur="saveEdit"
                @keyup.enter="saveEdit"
                @keyup.escape="cancelEdit"
                @click.stop
              />
              <span v-else>{{ row.description }}</span>
            </td>
          </tr>
        </tbody>
      </table>

      <HelpTip text="The selected properties will be automatically matched to the AAS submodel template in the next stage. The agent will use semantic matching to find the best corresponding properties in the template." />
    </div>

    <div v-else class="progress-indicator">
      <div style="font-size: 48px; margin-bottom: 20px">⚠️</div>
      <h3>No data extracted yet</h3>
      
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, type Ref, type ComputedRef } from 'vue'
import type { WizardData, Entity } from '@/types'

// Extended entity type with selection state. There is no `name` alias --
// there never was one, and declaring it is what let the table read a field
// nothing writes.
interface SelectableEntity extends Entity {
  selected?: boolean
}

// Editing cell state
interface EditingCell {
  index: number
  field: keyof SelectableEntity
}

// Props
interface Props {
  data: WizardData
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  update: [key: keyof WizardData, value: any]
}>()

// Editing state
const editingCell: Ref<EditingCell | null> = ref(null)
const editValue = ref('')

const warningsOpen = ref(true)

/**
 * Configurations the document distinguishes.
 *
 * From the extraction when it reported them, otherwise read off the rows -- a
 * file written before the extractor reported configurations still marks them
 * on its entities.
 */
const configurations: ComputedRef<string[]> = computed(() => {
  if (props.data.extractionConfigurations.length > 0) {
    return props.data.extractionConfigurations
  }
  const seen: string[] = []
  for (const e of props.data.extractedEntities) {
    const name = (e as SelectableEntity).variant
    if (name && !seen.includes(name)) seen.push(name)
  }
  return seen
})

const universalCount: ComputedRef<number> = computed(() => {
  return props.data.extractedEntities.filter(e => !(e as SelectableEntity).variant).length
})

const countFor = (name: string): number => {
  return props.data.extractedEntities.filter(e => (e as SelectableEntity).variant === name).length
}

/**
 * Choosing a configuration is expressed as a selection, not a new mechanism.
 *
 * The rows for other configurations are unticked rather than removed: `save-
 * edited-entities` already drops unticked rows, the user can see exactly what
 * was set aside and tick one back, and nothing is deleted from the extraction
 * the model produced.
 */
const chooseConfiguration = (name: string): void => {
  const entities = props.data.extractedEntities.map(e => {
    const row = e as SelectableEntity
    // A value with no configuration holds for all of them, so it stays.
    return { ...row, selected: !row.variant || row.variant === name }
  })
  emit('update', 'chosenConfiguration', name)
  emit('update', 'extractedEntities', entities)
}

const sourceTypeLabel: ComputedRef<string> = computed(() => {
  if (props.data.dataSourceType === 'url') return 'website'
  if (props.data.dataSourceType === 'pdf') return 'PDF document'
  if (props.data.dataSourceType === 'text') return 'text description'
  return 'data source'
})

const selectedCount: ComputedRef<number> = computed(() => {
  return props.data.extractedEntities.filter(e => (e as SelectableEntity).selected !== false).length
})

const allSelected: ComputedRef<boolean> = computed(() => {
  return props.data.extractedEntities.every(e => (e as SelectableEntity).selected !== false)
})

const isEditing = (index: number, field: keyof SelectableEntity): boolean => {
  return editingCell.value?.index === index && editingCell.value?.field === field
}

const startEditing = async (index: number, field: keyof SelectableEntity): Promise<void> => {
  const entity = props.data.extractedEntities[index] as SelectableEntity
  editingCell.value = { index, field }
  editValue.value = (entity[field] as string) || ''

  // Focus input after Vue updates the DOM
  await nextTick()
  const inputs = document.querySelectorAll<HTMLInputElement>('.edit-input')
  if (inputs.length > 0) {
    inputs[inputs.length - 1].focus()
    inputs[inputs.length - 1].select()
  }
}

const saveEdit = (): void => {
  if (!editingCell.value) return

  const { index, field } = editingCell.value
  const entities = [...props.data.extractedEntities] as SelectableEntity[]

  // Only update if value changed
  if (entities[index][field] !== editValue.value) {
    entities[index] = {
      ...entities[index],
      [field]: editValue.value
    }
    emit('update', 'extractedEntities', entities)
  }

  editingCell.value = null
  editValue.value = ''
}

const cancelEdit = (): void => {
  editingCell.value = null
  editValue.value = ''
}

const toggleEntity = (index: number): void => {
  // Don't toggle if we're editing
  if (editingCell.value) return

  const entities = [...props.data.extractedEntities] as SelectableEntity[]
  entities[index] = {
    ...entities[index],
    selected: entities[index].selected === false ? true : false
  }
  emit('update', 'extractedEntities', entities)
}

const toggleAll = (): void => {
  const newValue = !allSelected.value
  const entities = props.data.extractedEntities.map(e => ({
    ...e,
    selected: newValue
  }))
  emit('update', 'extractedEntities', entities)
}

const selectAll = (): void => {
  const entities = props.data.extractedEntities.map(e => ({
    ...e,
    selected: true
  }))
  emit('update', 'extractedEntities', entities)
}

const deselectAll = (): void => {
  const entities = props.data.extractedEntities.map(e => ({
    ...e,
    selected: false
  }))
  emit('update', 'extractedEntities', entities)
}
</script>

<style scoped>
.config-picker {
  margin-bottom: 16px;
  padding: 16px;
  background: #fff8e6;
  border-left: 4px solid #f0ad4e;
  border-radius: 6px;
}

.config-picker h4 {
  margin: 0 0 6px;
  color: #333;
  font-size: 15px;
}

.config-picker p {
  margin: 0 0 12px;
  color: #666;
  font-size: 13px;
  line-height: 1.5;
}

.config-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.config-option {
  padding: 8px 14px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
  text-align: left;
  line-height: 1.35;
}

.config-option:hover {
  border-color: #179C7D;
}

.config-option-active {
  border-color: #179C7D;
  background: #eafaf5;
  font-weight: 600;
}

.config-count {
  display: block;
  color: #888;
  font-size: 11px;
  font-weight: 400;
}

.config-chosen {
  margin: 12px 0 0;
  font-size: 13px;
  color: #444;
}

.extraction-warnings {
  margin-bottom: 16px;
  padding: 14px 16px;
  background: #fdf3f3;
  border-left: 4px solid #d9534f;
  border-radius: 6px;
}

.extraction-warnings h4 {
  margin: 0;
  font-size: 14px;
  color: #333;
  display: flex;
  align-items: center;
  gap: 10px;
}

.extraction-warnings ul {
  margin: 10px 0 0;
  padding-left: 20px;
  color: #555;
  font-size: 13px;
  line-height: 1.6;
}

.warn-toggle {
  border: 1px solid #d9d9d9;
  background: #fff;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 12px;
  cursor: pointer;
  color: #666;
}

.variant-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  background: #eef2f7;
  color: #445;
  font-size: 12px;
}

.row-deselected {
  opacity: 0.5;
  background: #f5f5f5;
}

.row-deselected td {
  text-decoration: line-through;
}

.btn-sm {
  padding: 4px 12px;
  font-size: 12px;
}

.table-hint {
  display: flex;
  gap: 20px;
  margin-bottom: 8px;
  font-size: 12px;
  color: #888;
}

.table-hint span::before {
  content: "•";
  margin-right: 6px;
}

.edit-input {
  width: 100%;
  padding: 4px 8px;
  border: 2px solid #667eea;
  border-radius: 4px;
  font-size: 14px;
  font-family: inherit;
  background: #fff;
  box-sizing: border-box;
}

.edit-input:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

/* Make editable cells show cursor hint */
td:not(:first-child) {
  position: relative;
}

td:not(:first-child):hover::after {
  content: "";
  position: absolute;
  bottom: 2px;
  right: 4px;
  width: 12px;
  height: 12px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23999' stroke-width='2'%3E%3Cpath d='M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7'/%3E%3Cpath d='M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z'/%3E%3C/svg%3E");
  background-size: contain;
  opacity: 0.5;
  pointer-events: none;
}

.row-checkbox {
  width: 18px;
  height: 18px;
  cursor: pointer;
}
</style>
