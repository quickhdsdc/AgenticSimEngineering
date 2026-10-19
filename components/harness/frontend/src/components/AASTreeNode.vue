<template>
  <div class="tree-node" v-show="node.visible !== false">
    <div 
      class="node-content" 
      :class="{ 'is-selected': isSelected, 'is-selectable': isSelectable }"
      @click="onSelect"
    >
      <!-- Toggle Icon -->
      <span 
        v-if="hasChildren" 
        class="toggle-icon" 
        @click.stop="toggle"
      >
        <span v-if="node.expanded">▼</span>
        <span v-else>▶</span>
      </span>
      <span v-else class="toggle-placeholder"></span>

      <!-- Type Icon -->
      <span class="type-icon">{{ typeIcon }}</span>

      <!-- Name -->
      <span class="node-name">{{ node.name }}</span>

      <!-- Type Badge -->
      <span v-if="node.data" class="type-badge" :class="getTypeClass(node.data.type)">
        {{ formatType(node.data.type) }}
      </span>
    </div>

    <!-- Children -->
    <div v-if="hasChildren && node.expanded" class="node-children">
      <AASTreeNode
        v-for="child in node.children"
        :key="child.path"
        :node="child"
        :selected-ids="selectedIds"
        @select="(prop) => $emit('select', prop)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// Types (should be imported, but defining here for now)
interface AASProperty {
  idShort: string
  type: string
  description?: string
  semanticPath?: string
  apiPath?: string
  semanticId?: string
}

interface TreeNode {
  name: string
  path: string
  type: string
  data?: AASProperty
  children: TreeNode[]
  expanded: boolean
  visible: boolean
}

const props = defineProps<{
  node: TreeNode
  selectedIds: string[]
}>()

const emit = defineEmits<{
  (e: 'select', property: AASProperty): void
}>()

const hasChildren = computed(() => props.node.children && props.node.children.length > 0)

const isSelectable = computed(() => {
    // Only properties with data are selectable
    // Unless we allow selecting collections? Usually we match to properties or collections.
    return !!props.node.data
})

const isSelected = computed(() => {
  return props.node.data && props.selectedIds.includes(props.node.data.idShort)
})

const typeIcon = computed(() => {
  const t = props.node.type
  if (t === 'AAS') return '🏭' // Factory/AAS
  if (t === 'Submodel') return '📦' // Submodel Package
  if (t === 'SubmodelElementCollection' || t === 'SubmodelElementList' || t === 'Folder') return '📂' // Folder
  if (t === 'Property') return '📄' // Property
  if (t === 'MultiLanguageProperty') return '🌐'
  if (t === 'File' || t === 'Blob') return '📎'
  if (t === 'ReferenceElement') return '🔗'
  if (t === 'Range') return '↔️'
  return '🔹'
})

const getTypeClass = (type: string) => {
  if (type === 'SubmodelElementCollection' || type === 'SubmodelElementList') return 'badge-collection'
  return 'badge-property'
}

const formatType = (type: string) => {
  if (type === 'SubmodelElementCollection') return 'SMC'
  if (type === 'SubmodelElementList') return 'SML'
  if (type === 'MultiLanguageProperty') return 'MLP'
  if (type === 'ReferenceElement') return 'Ref'
  return type.substring(0, 4)
}

const toggle = () => {
  props.node.expanded = !props.node.expanded
}

const onSelect = () => {
  if (props.node.data) {
    emit('select', props.node.data)
  } else {
      toggle()
  }
}
</script>

<style scoped>
.tree-node {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-size: 13px;
  user-select: none;
}

.node-content {
  display: flex;
  align-items: center;
  padding: 4px 8px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.1s;
}

.node-content:hover {
  background-color: #f0f0f0;
}

.node-content.is-selected {
  background-color: #e6f7ff; /* Light blue highlight */
  border: 1px solid #1890ff;
}

.is-selectable {
    cursor: pointer;
}

.toggle-icon {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 4px;
  color: #888;
  font-size: 10px;
  cursor: pointer;
}

.toggle-icon:hover {
    color: #333;
}

.toggle-placeholder {
  width: 20px;
}

.type-icon {
  margin-right: 6px;
  font-size: 14px;
}

.node-name {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #333;
}

.type-badge {
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 3px;
  margin-left: 8px;
  color: #666;
  background-color: #f5f5f5;
  border: 1px solid #e0e0e0;
}

.badge-collection {
  background-color: #e6f7ff;
  color: #096dd9;
  border-color: #91d5ff;
}

.badge-property {
  background-color: #f6ffed;
  color: #389e0d;
  border-color: #b7eb8f;
}

.node-children {
  margin-left: 22px; /* Deeper indent */
  padding-left: 10px;
  border-left: 1px solid #e0e0e0;
}
</style>
