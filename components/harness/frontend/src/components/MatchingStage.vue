<template>
  <div class="matching-stage">
    <h2 class="section-title">Property Matching</h2>

    <div v-if="data.isMatching" class="progress-indicator">
      <div class="spinner"></div>
      <h3>Matching properties...</h3>
      
    </div>

    <div v-else-if="data.matchedProperties.length > 0">
      <div class="alert alert-success">
        Successfully processed <strong>{{ entityCount }}</strong> properties.
        <strong>{{ placedEntityCount }}</strong> placed in the AAS,
        <strong>{{ entityCount - placedEntityCount }}</strong> with no home found.
        <HelpTip text="Each property is checked against each submodel. A property can be matched in one submodel and unmatched in another." />
      </div>

      <div style="margin-bottom: 20px; padding: 16px; background: #f0fdf7; border-radius: 6px; border-left: 4px solid #179C7D">
        <details><summary>Match types</summary>
        <div style="display: flex; gap: 20px; flex-wrap: wrap">
          <div style="display: flex; align-items: center; gap: 8px">
            <span class="match-type-badge match-property">Property</span>
            <span style="font-size: 13px; color: #666">Direct property match</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px">
            <span class="match-type-badge match-collection">Collection</span>
            <span style="font-size: 13px; color: #666">Matched to collection</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px">
            <span class="match-type-badge match-none">No Match</span>
            <span style="font-size: 13px; color: #666">No suitable match found</span>
          </div>
        </div>
      </details></div>

      <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #666; font-size: 14px;">
          {{ selectedCount }} of {{ data.matchedProperties.length }} matches selected for population
        </span>
        <div>
          <button class="btn btn-secondary btn-sm" @click="selectAllMatched" style="margin-right: 8px;">
            Select Matched
          </button>
          <button class="btn btn-secondary btn-sm" @click="deselectAll">
            Deselect All
          </button>
        </div>
      </div>

      <HelpTip text="Click row to select/deselect Click AAS Property to change match (multi-select supported)" />

      <table class="entity-table">
        <thead>
          <tr>
            <th style="width: 5%">
              <input
                type="checkbox"
                :checked="allMatchedSelected"
                @change="toggleAllMatched"
                title="Select/Deselect All Matched"
              />
            </th>
            <th style="width: 20%">Extracted Property</th>
            <th style="width: 12%">Value</th>
            <th style="width: 38%">AAS Property</th>
            <th style="width: 12%">Type</th>
            <th style="width: 13%">Confidence</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(match, index) in data.matchedProperties"
            :key="index"
            :class="{
              'row-unmatched': getMatchType(match) === 'none',
              'row-deselected': match.selected === false && getMatchType(match) !== 'none'
            }"
          >
            <td @click="toggleMatch(index)">
              <input
                type="checkbox"
                :checked="match.selected !== false"
                :disabled="getMatchType(match) === 'none'"
                @change="toggleMatch(index)"
              />
            </td>
            <td @click="toggleMatch(index)" style="cursor: pointer;">
              <strong>{{ match.extractedEntity }}</strong>
            </td>
            <td @click="toggleMatch(index)" style="cursor: pointer;" :title="match.extractedValue">
              {{ truncateValue(match.extractedValue) }}
            </td>
            <td class="aas-property-cell" @click.stop="openDropdown(index, $event)">
              <div class="property-selector" :class="{ 'is-open': activeDropdown === index }">
                <div class="selected-property">
                  <!-- Display multiple properties -->
                  <div v-if="getSelectedProperties(match).length > 0" class="selected-props-list">
                    <span
                      v-for="(prop, pIndex) in getSelectedProperties(match)"
                      :key="pIndex"
                      class="selected-prop-tag"
                      :title="prop.idShort"
                    >
                      <span class="tag-icon">{{ getPropTypeIcon(prop.type) }}</span>
                      {{ prop.idShort }}
                    </span>
                  </div>
                  <span v-else style="color: #999; font-style: italic; font-size: 14px;">— Click to Select —</span>
                  <span class="dropdown-arrow">✎</span>
                </div>
              </div>
            </td>

            <td>
              <span :class="getMatchTypeBadgeClass(getMatchType(match))">
                {{ formatMatchType(getMatchType(match)) }}
              </span>
              <span v-if="getSelectedProperties(match).length > 1" class="multi-badge">
                +{{ getSelectedProperties(match).length - 1 }}
              </span>
            </td>
            <td>
              <span v-if="getMatchType(match) !== 'none'" :class="getConfidenceBadgeClass(match.confidence)">
                {{ (match.confidence * 100).toFixed(0) }}%
              </span>
              <span v-else style="color: #999">—</span>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="unmatchedCount > 0" style="margin-top: 20px; padding: 16px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107; position: relative; z-index: 1;">
        <h4 style="margin-bottom: 8px; color: #333">Unmatched Properties</h4>
        <p style="color: #666; line-height: 1.6; font-size: 14px">
          {{ entityCount - placedEntityCount }} properties could not be matched to the AAS template.
          Click on "— Select —" to manually assign matches (multiple selections allowed).
        </p>
      </div>

      <HelpTip text="Review and adjust property matches as needed. You can select multiple AAS properties for each extracted entity. Click &quot;Done&quot; to confirm your selections. Only selected matches will be populated in the final AAS." />
    </div>

    <div v-else class="progress-indicator">
      <div style="font-size: 48px; margin-bottom: 20px">⚠️</div>
      <h3>No matches found yet</h3>
      
    </div>
    <!-- Property Selector Modal -->
    <div v-if="activeDropdown !== null" class="modal-overlay" @click.self="closeDropdown">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Select AAS Property</h3>
          <button class="close-btn" @click="closeDropdown">✕</button>
        </div>
        
        <div class="modal-body">
           <div class="current-selection-info" v-if="activeMatch">
              <p><strong>Extracted:</strong> {{ activeMatch.extractedEntity }}</p>
              <p><strong>Value:</strong> {{ truncateValue(activeMatch.extractedValue) }}</p>
           </div>
           
           <AASPropertyTree
             v-if="activeMatch"
             :properties="availableProperties"
             :selected-properties="getSelectedProperties(activeMatch)"
             @select="onPropertySelect"
           />
        </div>
        
        <div class="modal-footer" v-if="activeMatch">
            <div class="selection-summary">
                {{ getSelectedProperties(activeMatch).length }} properties selected
            </div>
            <button class="btn btn-secondary" @click="clearAllSelections(activeDropdown!)">Clear Selection</button>
            <button class="btn btn-primary" @click="closeDropdown">Done</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick, type Ref, type ComputedRef } from 'vue'
import { useEnvStore } from '@/stores/environmentStore'
import { aasService } from '@/services/aasService'
import { ApiError } from '@/services/api'
import type { WizardData, MatchedProperty } from '@/types'
import AASPropertyTree from './AASPropertyTree.vue'

// Property type from AAS
interface AASProperty {
  idShort: string
  type: string
  description?: string
  semanticPath?: string
  apiPath?: string
  semanticId?: string
}

// Extended matched property with selection
interface ExtendedMatchedProperty extends MatchedProperty {
  aasProperties?: AASProperty[]
  aasProperty?: string
  aasType?: string
  matchType?: string
  originalMatch?: string
  originalType?: string
  originalConfidence?: number
  reason?: string
  selected?: boolean
  semanticPath?: string
  apiPath?: string
  semanticId?: string
}

// Dropdown position
interface DropdownPosition {
  top: number
  left: number
  width: number
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

// Environment store
const environmentStore = useEnvStore()

// Refs
const activeDropdown: Ref<number | null> = ref(null)
const searchQuery = ref('')
const availableProperties: Ref<AASProperty[]> = ref([])
const groupedProperties: Ref<Record<string, AASProperty[]>> = ref({})
const searchInput: Ref<HTMLInputElement | null> = ref(null)
const dropdownPosition: Ref<DropdownPosition> = ref({ top: 0, left: 0, width: 0 })

// Load available AAS properties
const loadAasProperties = async (): Promise<void> => {
  const aasIdShort = props.data.targetAAS
  if (!aasIdShort) {
    console.warn('No AAS ID Short (targetAAS) provided, cannot load properties')
    return
  }

  try {
    const result = await aasService.getAASProperties(aasIdShort)
    availableProperties.value = result.properties || []
    groupedProperties.value = result.grouped || {}
    console.log(`Loaded ${result.count} AAS properties for ${aasIdShort}`)
  } catch (error) {
    console.error('Error loading AAS properties:', error)
    if (error instanceof ApiError) {
      console.error(`API Error: ${error.status} - ${error.message}`)
    }
  }
}

// Watch for targetAAS changes and load properties when matching results are available
watch(() => [props.data.targetAAS, props.data.matchedProperties.length], ([newAAS, newLength]) => {
  if (newAAS && newLength > 0) {
    loadAasProperties()
  }
}, { immediate: true })

// Filter properties based on search
const filteredGroupedProperties: ComputedRef<Record<string, AASProperty[]>> = computed(() => {
  if (!searchQuery.value.trim()) {
    return groupedProperties.value
  }

  const query = searchQuery.value.toLowerCase()
  const filtered: Record<string, AASProperty[]> = {}

  for (const [group, propsList] of Object.entries(groupedProperties.value)) {
    const matchedProps = propsList.filter(p =>
      p.idShort.toLowerCase().includes(query) ||
      (p.description && p.description.toLowerCase().includes(query)) ||
      (p.semanticPath && p.semanticPath.toLowerCase().includes(query))
    )
    if (matchedProps.length > 0) {
      filtered[group] = matchedProps
    }
  }

  return filtered
})

// Get selected properties array from match object
const getSelectedProperties = (match: ExtendedMatchedProperty): AASProperty[] => {
  if (match.aasProperties && Array.isArray(match.aasProperties)) {
    // Normalize if array contains strings (legacy)
    return match.aasProperties.map(p => {
        if (typeof p === 'string') {
            return { idShort: p, type: match.aasType || 'Property' }
        }
        return p
    })
  }
  if (match.aasProperty) {
    return [{ idShort: match.aasProperty, type: match.aasType || '' }]
  }
  return []
}

// Check if a property is selected
const isPropertySelected = (match: ExtendedMatchedProperty, idShort: string): boolean => {
  const selected = getSelectedProperties(match)
  return selected.some(p => (p.idShort || p) === idShort)
}

// Get match type based on selected properties
const getMatchType = (match: ExtendedMatchedProperty): string => {
  const selected = getSelectedProperties(match)
  if (selected.length === 0) return 'none'

  const hasCollection = selected.some(p => {
    // `targetType` is what is *at* the path; `aasType` is what would be put
    // there, and for a hosted create that is a Property while the target is a
    // collection. Reading `aasType` here drew a matcher-proposed create as a
    // direct PROPERTY match, which is the opposite of what it is.
    const type = p.type || match.targetType || match.aasType
    return type === 'SubmodelElementCollection' || type === 'SubmodelElementList'
  })

  return hasCollection ? 'collection' : 'property'
}

const matchedCount: ComputedRef<number> = computed(() => {
  return props.data.matchedProperties.filter(m => getMatchType(m as ExtendedMatchedProperty) !== 'none').length
})

/**
 * Properties, not rows.
 *
 * Every extracted property is matched against every submodel of the shell, so
 * one property produces one row per submodel. Counting rows and calling them
 * properties reported "16 of 87" for a run that placed 15 of 29 -- an 18%
 * success rate presented for work that was 52%, with the difference being
 * rows the user is meant to ignore.
 */
const entityCount: ComputedRef<number> = computed(() => {
  return new Set(props.data.matchedProperties.map(m => m.extractedEntity)).size
})

const placedEntityCount: ComputedRef<number> = computed(() => {
  const placed = new Set<string>()
  for (const m of props.data.matchedProperties) {
    if (getMatchType(m as ExtendedMatchedProperty) !== 'none') placed.add(m.extractedEntity)
  }
  return placed.size
})

const unmatchedCount: ComputedRef<number> = computed(() => {
  return props.data.matchedProperties.filter(m => getMatchType(m as ExtendedMatchedProperty) === 'none').length
})

const selectedCount: ComputedRef<number> = computed(() => {
  return props.data.matchedProperties.filter(m =>
    (m as ExtendedMatchedProperty).selected !== false &&
    getMatchType(m as ExtendedMatchedProperty) !== 'none'
  ).length
})

const allMatchedSelected: ComputedRef<boolean> = computed(() => {
  const matchedItems = props.data.matchedProperties.filter(m => getMatchType(m as ExtendedMatchedProperty) !== 'none')
  return matchedItems.length > 0 && matchedItems.every(m => (m as ExtendedMatchedProperty).selected !== false)
})

const truncateValue = (value: any): string => {
  if (!value) return ''
  const str = String(value)
  return str.length > 20 ? str.substring(0, 20) + '...' : str
}

const getConfidenceBadgeClass = (confidence: number): string => {
  if (confidence >= 0.9) return 'confidence-badge confidence-high'
  if (confidence >= 0.7) return 'confidence-badge confidence-medium'
  return 'confidence-badge confidence-low'
}

const getMatchTypeBadgeClass = (matchType: string): string => {
  if (matchType === 'property') return 'match-type-badge match-property'
  if (matchType === 'collection') return 'match-type-badge match-collection'
  return 'match-type-badge match-none'
}

const formatMatchType = (matchType: string): string => {
  if (matchType === 'property') return 'Property'
  if (matchType === 'collection') return 'Collection'
  return 'No Match'
}

const getTypeClass = (type: string): string => {
  if (type === 'SubmodelElementCollection' || type === 'SubmodelElementList') {
    return 'type-collection'
  }
  return 'type-property'
}

const formatType = (type: string): string => {
  if (type === 'SubmodelElementCollection') return 'SMC'
  if (type === 'SubmodelElementList') return 'SML'
  if (type === 'MultiLanguageProperty') return 'MLP'
  if (type === 'ReferenceElement') return 'Ref'
  return type.substring(0, 4)
}

const openDropdown = async (index: number, event: MouseEvent): Promise<void> => {
  if (activeDropdown.value === index) {
    activeDropdown.value = null
  } else {
    const cell = event.currentTarget as HTMLElement
    const rect = cell.getBoundingClientRect()

    dropdownPosition.value = {
      top: rect.bottom + window.scrollY,
      left: rect.left + window.scrollX,
      width: Math.max(rect.width, 320)
    }

    activeDropdown.value = index
    searchQuery.value = ''

    await nextTick()
    if (searchInput.value) {
      searchInput.value.focus()
    }
  }
}

const closeDropdown = (): void => {
  activeDropdown.value = null
}

const closeDropdownOnOutsideClick = (event: MouseEvent): void => {
  if (activeDropdown.value !== null && !(event.target as HTMLElement).closest('.property-selector')) {
    activeDropdown.value = null
  }
}

const togglePropertySelection = (
  index: number,
  idShort: string,
  type: string,
  confidence: number,
  propData: AASProperty | null = null
): void => {
  const matches = [...props.data.matchedProperties] as ExtendedMatchedProperty[]
  const currentMatch = matches[index]

  if (!currentMatch.originalMatch && currentMatch.aasProperty) {
    currentMatch.originalMatch = currentMatch.aasProperty
    currentMatch.originalType = currentMatch.aasType
    currentMatch.originalConfidence = currentMatch.confidence
  }

  let currentSelections: AASProperty[] = []
  if (currentMatch.aasProperties && Array.isArray(currentMatch.aasProperties)) {
    currentSelections = [...currentMatch.aasProperties]
  } else if (currentMatch.aasProperty) {
    currentSelections = [{
      idShort: currentMatch.aasProperty,
      type: currentMatch.aasType || '',
      semanticPath: currentMatch.semanticPath || '',
      apiPath: currentMatch.apiPath || '',
      semanticId: currentMatch.semanticId || ''
    }]
  }

  const existingIndex = currentSelections.findIndex(p => p.idShort === idShort)

  if (existingIndex >= 0) {
    currentSelections.splice(existingIndex, 1)
  } else {
    currentSelections.push({
      idShort: idShort,
      type: type,
      semanticPath: propData?.semanticPath || '',
      apiPath: propData?.apiPath || '',
      semanticId: propData?.semanticId || '',
      description: propData?.description || ''
    })
  }

  matches[index] = {
    ...currentMatch,
    aasProperties: currentSelections,
    aasProperty: currentSelections.length > 0 ? currentSelections[0].idShort : '',
    aasType: currentSelections.length > 0 ? currentSelections[0].type : '',
    matchType: currentSelections.length > 0 ?
      (currentSelections.some(p => p.type === 'SubmodelElementCollection' || p.type === 'SubmodelElementList') ? 'collection' : 'property')
      : 'none',
    confidence: currentSelections.length > 0 ? confidence : 0,
    selected: currentSelections.length > 0,
    reason: currentSelections.length > 0 ?
      (currentSelections.length > 1 ? `Manually selected ${currentSelections.length} properties` : 'Manually selected')
      : 'No match selected'
  }

  emit('update', 'matchedProperties', matches)
}

const clearAllSelections = (index: number): void => {
  const matches = [...props.data.matchedProperties] as ExtendedMatchedProperty[]
  const currentMatch = matches[index]

  matches[index] = {
    ...currentMatch,
    aasProperties: [],
    aasProperty: '',
    aasType: '',
    matchType: 'none',
    confidence: 0,
    selected: false,
    reason: 'No match selected'
  }

  emit('update', 'matchedProperties', matches)
  activeDropdown.value = null
}

const toggleMatch = (index: number): void => {
  const match = props.data.matchedProperties[index] as ExtendedMatchedProperty
  if (getMatchType(match) === 'none') return

  const matches = [...props.data.matchedProperties] as ExtendedMatchedProperty[]
  matches[index] = {
    ...matches[index],
    selected: matches[index].selected === false ? true : false
  }
  emit('update', 'matchedProperties', matches)
}

const toggleAllMatched = (): void => {
  const newValue = !allMatchedSelected.value
  const matches = props.data.matchedProperties.map(m => ({
    ...m,
    selected: getMatchType(m as ExtendedMatchedProperty) !== 'none' ? newValue : false
  }))
  emit('update', 'matchedProperties', matches)
}

const selectAllMatched = (): void => {
  const matches = props.data.matchedProperties.map(m => ({
    ...m,
    selected: getMatchType(m as ExtendedMatchedProperty) !== 'none'
  }))
  emit('update', 'matchedProperties', matches)
}

const deselectAll = (): void => {
  const matches = props.data.matchedProperties.map(m => ({
    ...m,
    selected: false
  }))
  emit('update', 'matchedProperties', matches)
}

const filterProperties = (): void => {
  // Filtering is handled by computed property
}

const activeMatch = computed(() => {
  if (activeDropdown.value === null) return null
  return props.data.matchedProperties[activeDropdown.value] as ExtendedMatchedProperty
})

const onPropertySelect = (property: AASProperty) => {
    if (activeDropdown.value === null) return
    
    // Use the existing logic but adapted
    togglePropertySelection(
        activeDropdown.value, 
        property.idShort, 
        property.type, 
        1.0, 
        property
    )
    // Don't close immediately? Or close? User might want to select multiple?
    // User requested "click a AAS property, then pop up a new UI". Use "Done" to close.
    // If single select, maybe close. But we support multi-select.
}

const getPropTypeIcon = (type: string): string => {
  if (type === 'SubmodelElementCollection') return '📂'
  if (type === 'SubmodelElementList') return '🗂️'
  if (type === 'Property') return '📄'
  if (type === 'MultiLanguageProperty') return '🌐'
  if (type === 'File') return '📎'
  if (type === 'ReferenceElement') return '🔗'
  if (type === 'Range') return '↔️'
  return '🔹'
}

// Close dropdown on outside click - updated for modal
// Actually, the modal overlay handles self-click to close.
// We can keep the document listener as backup or remove it if it interferes.
// Removing it might be safer for modal.
// onMounted(() => {
//   document.addEventListener('click', closeDropdownOnOutsideClick as EventListener)
// })
// onUnmounted(() => {
//   document.removeEventListener('click', closeDropdownOnOutsideClick as EventListener)
// })
</script>

<style scoped>
/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 99999;
}

.modal-content {
  background: white;
  width: 80%;
  max-width: 900px;
  height: 80vh;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.modal-header {
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #999;
}

.close-btn:hover {
  color: #333;
}

.modal-body {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 0; /* Padding handled inside components or tree */
}

.current-selection-info {
  padding: 12px 20px;
  background: #f9f9f9;
  border-bottom: 1px solid #eee;
  font-size: 13px;
  color: #555;
}

.current-selection-info p {
  margin: 4px 0;
}

.modal-footer {
  padding: 16px 20px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 12px;
}

.selection-summary {
    margin-right: auto;
    font-size: 13px;
    color: #666;
    font-weight: 500;
}

/* Utils */
.tag-icon {
    margin-right: 4px;
    font-size: 11px;
}


.row-unmatched td {
  color: #999;
}

.row-unmatched .property-selector {
  opacity: 1;
}

.row-unmatched .property-dropdown {
  opacity: 1;
  background: white;
}

.row-deselected {
  background: #fafafa;
}

.row-deselected td {
  text-decoration: line-through;
  color: #aaa;
}

.row-deselected .property-selector {
  opacity: 1;
}

.row-deselected .property-dropdown {
  opacity: 1;
  background: white;
}

.match-type-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.match-property {
  background: #d4edda;
  color: #155724;
}

.match-collection {
  background: #cce5ff;
  color: #004085;
}

.match-none {
  background: #f5f5f5;
  color: #6c757d;
}

.multi-badge {
  display: inline-block;
  margin-left: 4px;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  background: #667eea;
  color: white;
}

.confidence-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}

.confidence-high {
  background: #d4edda;
  color: #155724;
}

.confidence-medium {
  background: #fff3cd;
  color: #856404;
}

.confidence-low {
  background: #f8d7da;
  color: #721c24;
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

/* Table needs higher stacking context for dropdowns */
.entity-table {
  position: relative;
  z-index: 100;
}

/* Property Selector Dropdown */
.aas-property-cell {
  position: relative;
  padding: 0 !important;
  overflow: visible;
}

.property-selector {
  position: relative;
  cursor: pointer;
  padding: 8px 12px;
}

.property-selector:hover {
  background: #f0f9f7;
}

.property-selector.is-open {
  z-index: 10000;
}

.selected-property {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.selected-props-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  flex: 1;
}

.selected-prop-tag {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 600;
  background: #e8f5f1;
  color: #179C7D;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: help;
}

.dropdown-arrow {
  font-size: 10px;
  color: #999;
  transition: transform 0.2s;
  flex-shrink: 0;
}

.property-selector.is-open .dropdown-arrow {
  transform: rotate(180deg);
}

.property-dropdown {
  min-width: 320px;
  max-height: 450px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
  z-index: 99999;
  overflow: hidden;
}

.dropdown-search {
  padding: 8px;
  border-bottom: 1px solid #e0e0e0;
}

.dropdown-search input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
  box-sizing: border-box;
}

.dropdown-search input:focus {
  outline: none;
  border-color: #179C7D;
}

.dropdown-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: #f8f9fa;
  border-bottom: 1px solid #e0e0e0;
}

.selected-count {
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.btn-done {
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 600;
  background: #179C7D;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-done:hover {
  background: #148a6e;
}

.dropdown-section {
  border-bottom: 1px solid #f0f0f0;
}

.dropdown-section:last-child {
  border-bottom: none;
}

.section-header {
  padding: 8px 12px;
  font-size: 11px;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  background: #f8f9fa;
}

.dropdown-items {
  max-height: 250px;
  overflow-y: auto;
}

.group-header {
  padding: 6px 12px;
  font-size: 11px;
  font-weight: 600;
  color: #179C7D;
  background: #f0fdf7;
}

.dropdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  transition: background 0.15s;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.item-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.item-path {
  font-size: 11px;
  color: #888;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-left: 24px;
  font-family: monospace;
}


.dropdown-item:hover {
  background: #f0f9f7;
}

.dropdown-item.is-selected {
  background: #e8f5f1;
}

.dropdown-item.suggestion {
  background: #fffef0;
}

.dropdown-item.suggestion:hover {
  background: #fff9d6;
}

.dropdown-item.no-match-option {
  color: #dc3545;
}

.dropdown-item.no-match-option:hover {
  background: #fff5f5;
}

.dropdown-item input[type="checkbox"] {
  flex-shrink: 0;
}

.item-name {
  font-weight: 500;
  font-size: 14px;
  flex: 1;
}

.item-badge {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 3px;
  background: #179C7D;
  color: white;
}

.item-type {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 3px;
}

.type-property {
  background: #d4edda;
  color: #155724;
}

.type-collection {
  background: #cce5ff;
  color: #004085;
}

.item-desc {
  font-size: 11px;
  color: #999;
  margin-left: 24px;
  font-style: italic;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.no-results {
  padding: 12px;
  text-align: center;
  color: #999;
  font-style: italic;
}
</style>
