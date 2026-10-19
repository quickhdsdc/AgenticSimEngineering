<template>
  <div class="aas-tree-container">
    <div class="tree-header">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input
          type="text"
          v-model="searchQuery"
          placeholder="Search properties..."
          class="search-input"
          @input="onSearch"
        />
        <button v-if="searchQuery" class="clear-search" @click="clearSearch">✕</button>
      </div>
    </div>

    <div class="tree-body">
      <div v-if="treeData.length === 0" class="empty-state">
        <span v-if="loading">Loading properties...</span>
        <span v-else>No properties found</span>
      </div>
      
      <AASTreeNode
        v-for="node in treeData"
        :key="node.path"
        :node="node"
        :selected-ids="selectedIds"
        @select="onSelect"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import AASTreeNode from './AASTreeNode.vue'

// Types
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
  properties: AASProperty[]
  selectedProperties: AASProperty[]
  loading?: boolean
}>()

const emit = defineEmits<{
  (e: 'select', property: AASProperty): void
}>()

const searchQuery = ref('')
const treeData = ref<TreeNode[]>([])

const selectedIds = computed(() => {
  return props.selectedProperties.map(p => p.idShort)
})

// Build tree logic
const buildTree = (flatProps: AASProperty[]): TreeNode[] => {
  const rootNodes: TreeNode[] = []
  
  // Use a map to track created nodes by path
  const nodeMap = new Map<string, TreeNode>()

  // Helper to get or create a node
  const getOrCreateNode = (path: string, name: string, type: string, parentPath: string | null): TreeNode => {
      if (nodeMap.has(path)) return nodeMap.get(path)!

      const node: TreeNode = {
          name,
          path,
          type,
          children: [],
          expanded: true, // Default expanded
          visible: true
      }
      nodeMap.set(path, node)

      if (parentPath) {
          const parent = nodeMap.get(parentPath)
          if (parent) {
              parent.children.push(node)
          } else {
              // Parent doesn't exist? Should usually exist if we process top-down.
              // But if not, we might need to create virtual parents?
              // For simplicity, if parent not found, add to root (fallback)
              // Or better, handle hierarchy construction strictly.
              rootNodes.push(node) 
          }
      } else {
          rootNodes.push(node)
      }
      return node
  }

  // We need to sort properties by semanticPath to ensure parents are processed first?
  // Or just handle path splitting.
  // Semantic path example: "Battery/TechnicalData/Capacity"
  
  // Let's iterate and build.
  flatProps.forEach(prop => {
      // Normalize separators: replace dots with slashes
      const pathStr = (prop.semanticPath || prop.idShort).replace(/\./g, '/')
      const parts = pathStr.split('/').filter(p => p.trim() !== '') // Split and remove empty parts
      
      let currentPath = ''
      let parentPath: string | null = null

      parts.forEach((part, index) => {
          const isLast = index === parts.length - 1
          currentPath = currentPath ? `${currentPath}/${part}` : part
          
          let type = 'SubmodelElementCollection' // Default for deeper levels
          if (index === 0) type = 'AAS' 
          else if (index === 1) type = 'Submodel'
          
          if (isLast) type = prop.type // use actual type if leaf/defined

          // Check if node exists
          if (!nodeMap.has(currentPath)) {
             // Create it
             const node = getOrCreateNode(currentPath, part, type, parentPath)
             if (isLast) {
                 node.data = prop
                 node.type = prop.type // Ensure type matches prop
             }
          } else {
              // If it exists (maybe created as a parent for someone else previously), update data if this is the leaf for it
              if (isLast) {
                  const node = nodeMap.get(currentPath)!
                  node.data = prop
                  node.type = prop.type
              }
          }
          
          parentPath = currentPath
      })
  })

  // Determine root: "AAS" is usually the first part.
  // If api returns list of properties where semanticPath starts with AAS ID, then rootNodes will contain the AAS node.
  // If semanticPath omits AAS ID, then rootNodes will contain Submodels.
  
  return rootNodes
}

watch(() => props.properties, (newProps) => {
    if (newProps && newProps.length > 0) {
        treeData.value = buildTree(newProps)
    } else {
        treeData.value = []
    }
}, { immediate: true })

// Search logic
const onSearch = () => {
    if (!searchQuery.value.trim()) {
        resetVisibility(treeData.value)
        return
    }
    
    const query = searchQuery.value.toLowerCase()
    filterNodes(treeData.value, query)
}

const resetVisibility = (nodes: TreeNode[]) => {
    nodes.forEach(node => {
        node.visible = true
        resetVisibility(node.children)
    })
}

const filterNodes = (nodes: TreeNode[], query: string): boolean => {
    let hasVisibleNode = false
    
    nodes.forEach(node => {
        const nameMatch = node.name.toLowerCase().includes(query)
        const childMatch = filterNodes(node.children, query)
        
        if (nameMatch || childMatch) {
            node.visible = true
            hasVisibleNode = true
            // If match found deep, expand parents?
            // node.expanded = true // Optional: expand on search
        } else {
            node.visible = false
        }
    })
    
    return hasVisibleNode
}

const clearSearch = () => {
    searchQuery.value = ''
    onSearch()
}

const onSelect = (prop: AASProperty) => {
    emit('select', prop)
}

</script>

<style scoped>
.aas-tree-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-radius: 8px;
  overflow: hidden;
}

.tree-header {
  padding: 12px;
  border-bottom: 1px solid #eee;
  background: #fafafa;
}

.search-box {
  position: relative;
  display: flex;
  align-items: center;
}

.search-input {
  width: 100%;
  padding: 8px 32px 8px 32px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.1);
}

.search-icon {
  position: absolute;
  left: 10px;
  color: #999;
  font-size: 14px;
}

.clear-search {
  position: absolute;
  right: 10px;
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  font-size: 14px;
}

.clear-search:hover {
  color: #666;
}

.tree-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100px;
  color: #999;
  font-style: italic;
  font-size: 13px;
}
</style>
