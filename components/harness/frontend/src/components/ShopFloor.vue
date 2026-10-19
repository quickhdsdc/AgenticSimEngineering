<template>
  <div class="shop-floor-layout">
    <!-- Asset Library Sidebar -->
    <div class="asset-library">
      <h3>Asset Library</h3>
      <HelpTip text="Drag a blueprint onto the floor to create an asset." />
      
      <div 
        v-for="template in assetTemplates" 
        :key="template.type"
        class="asset-template"
        draggable="true"
        @dragstart="onDragStart($event, template)"
      >
        <img :src="template.icon" class="template-icon-img" alt="icon" />
        <div class="template-label">{{ template.label }}</div>
      </div>
    </div>

    <!-- Main Shop Floor Area -->
    <div class="shop-floor-container">
      <div class="shop-floor-header">
        <h2>Factory Floor Overview</h2>
        
        <button class="btn btn-info" @click="triggerManifestRefresh" style="margin-right: 10px;">
          Refresh Manifests
        </button>
        <button class="btn btn-danger" @click="resetShopFloor">
          Reset Shop Floor
        </button>
      </div>

      <!-- Shop Floor Visualization -->
      <div 
        class="floor-visual"
        @dragover.prevent
        @drop="onDrop"
        ref="floorRef"
      >
        <!-- Grid Background -->
        <div class="grid-bg"></div>

        <!-- Connections SVG Layer -->
        <svg class="connections-layer">
            <template v-for="conn in renderedConnections" :key="conn.id">
                <!-- Hit Area (Wider, Transparent) -->
                <path 
                  :d="conn.d" 
                  stroke="transparent" 
                  stroke-width="15" 
                  fill="none" 
                  class="connection-hit"
                  @click.stop="selectConnection(conn.id)"
                />
                <!-- Visible Line -->
                <path 
                  :d="conn.d" 
                  :stroke="conn.isSelected ? '#dc3545' : '#555'" 
                  :stroke-width="conn.isSelected ? 3 : 2"
                  fill="none" 
                  class="connection-line"
                  style="pointer-events: none;"
                />
            </template>
            
            <path v-if="dragConnection.active" 
                  :d="previewPath" 
                  stroke="#007bff" 
                  stroke-width="2" 
                  stroke-dasharray="5,5" 
                  fill="none" 
            />
        </svg>

        <!-- Dynamic Assets -->
        <div 
          v-for="asset in assets" 
          :key="asset.id"
          :class="['asset', asset.type]"
          :style="{ top: asset.y + '%', left: asset.x + '%' }"
          @click.stop="openAssetMenu(asset)"
          draggable="true"
          @dragstart.stop="onDragStartAsset($event, asset)"
        >
          <img :src="getIcon(asset.type)" class="asset-icon-img" alt="asset" />
          <div class="asset-label">{{ asset.label }}</div>
          <div :class="['status-indicator', asset.hasAas ? 'has-aas' : 'no-aas']"></div>
          
          <!-- Ports -->
          <div class="port port-top" @mousedown.stop="startConnection($event, asset.id, 'top')" @mouseup.stop="endConnection($event, asset.id, 'top')"></div>
          <div class="port port-right" @mousedown.stop="startConnection($event, asset.id, 'right')" @mouseup.stop="endConnection($event, asset.id, 'right')"></div>
          <div class="port port-bottom" @mousedown.stop="startConnection($event, asset.id, 'bottom')" @mouseup.stop="endConnection($event, asset.id, 'bottom')"></div>
          <div class="port port-left" @mousedown.stop="startConnection($event, asset.id, 'left')" @mouseup.stop="endConnection($event, asset.id, 'left')"></div>

        </div>

      </div>
    </div>

    <!-- The machine console, over everything else: while it is open the
         operator is looking at one machine, not at the floor. -->
    <MachineConsole
      v-if="consoleAsset"
      :asset="consoleAsset"
      @close="consoleAsset = null"
    />

    <!-- Context Menu / Modal -->
    <div v-if="selectedAsset" class="asset-modal-overlay" @click.self="selectedAsset = null">
      <div class="asset-modal">
        <h3>{{ selectedAsset.label }}</h3>
        <HelpTip :text="`Asset ID: ${selectedAsset.id}`" />
        
        <div class="modal-actions">
          <!-- First, because on a cell that has one it is the reason you
               clicked. Viewing the AAS is inspection; this is the machine. -->
          <button
            v-if="hasConsole(selectedAsset)"
            class="btn btn-primary"
            @click="openConsole"
          >
            Open machine console
          </button>

          <button class="btn btn-info" @click="handleAction('view')">
            View in BaSyx UI
          </button>
          
          <button class="btn btn-secondary" @click="handleAction('edit')">
            Edit Existing AAS
          </button>

          <button class="btn btn-danger" @click="deleteAsset" style="margin-top: 10px;">
            Remove Asset
          </button>
        </div>
        
        <button class="close-btn" @click="selectedAsset = null">×</button>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useWizardStore } from '../stores/wizardStore'
import { useEnvStore } from '../stores/environmentStore'
import type { AssetInstance } from '../types' // Import from shared types
import MachineConsole from './MachineConsole.vue'

const store = useWizardStore()
const envStore = useEnvStore()
const floorRef = ref<HTMLElement | null>(null)

// --- Types ---

interface AssetTemplate {
  type: string
  label: string
  icon: string
}

// AssetInstance is now imported from types

// --- State ---

// Icons as PNG paths
const assetTemplates: AssetTemplate[] = [
  { type: 'machine', label: 'CNC Machine', icon: '/assets/machine.png' },
  { type: 'metal_printer', label: 'Metal 3D Printer', icon: '/assets/printer.png' },
  { type: 'injection_molding', label: 'Injection Molding Machine', icon: '/assets/injection molding.png' },
  { type: 'disassembly_robot', label: 'Robotic Disassembly Station', icon: '/assets/robot.png' },
  { type: 'agv', label: 'AGV', icon: '/assets/agv.png' },
  { type: 'order', label: 'Production Order', icon: '/assets/order.png' },
  { type: 'product', label: 'Digital Battery Passport', icon: '/assets/product.png' },
  { type: 'process', label: 'Process', icon: '/assets/process.png' },
  { type: 'simulation', label: 'Simulation', icon: '/assets/simulation.png' },
  { type: 'aps', label: 'APS', icon: '/assets/aps.png' },
  { type: 'assemblyshop', label: 'Assembly Shop', icon: '/assets/assembly.png' },
  { type: 'testing_repair', label: 'Testing & Repair Station', icon: '/assets/repair.png' },
  { type: 'festo_cp_lab', label: 'FestoWorkCenterStamping', icon: '/assets/festo cp lab.png' }
]

// Assets that answer over OPC UA, and so have a console worth opening. Read
// off the type rather than probed: the shop floor draws before anything is
// instantiated, and a menu entry that appears a second late is worse than one
// that is simply right.
//
// The type key stays `festo_cp_lab` while the name is FestoWorkCenterStamping.
// It is an internal discriminator, not a label, and the store is persisted:
// renaming it would orphan every cell already on somebody's floor, leaving it
// with no icon and no console.
const CONSOLE_TYPES = new Set(['festo_cp_lab'])

// Use assets from store for persistence
const assets = computed({
  get: () => store.wizardData.assets,
  set: (newAssets) => store.updateData('assets', newAssets)
})

const selectedAsset = ref<AssetInstance | null>(null)


// --- Drag and Drop Logic ---

const onDragStart = (event: DragEvent, template: AssetTemplate) => {
  if (event.dataTransfer) {
    event.dataTransfer.setData('application/json', JSON.stringify({ type: 'template', data: template }))
    event.dataTransfer.effectAllowed = 'copy'
  }
}

const onDragStartAsset = (event: DragEvent, asset: AssetInstance) => {
  if (event.dataTransfer) {
    // Store cursor offset relative to asset center or top-left?
    // For simplicity, we drop centering on mouse for now
    event.dataTransfer.setData('application/json', JSON.stringify({ type: 'asset', id: asset.id }))
    event.dataTransfer.effectAllowed = 'move'
  }
}

const onDrop = (event: DragEvent) => {
  const rawData = event.dataTransfer?.getData('application/json')
  if (!rawData || !floorRef.value) return

  const payload = JSON.parse(rawData)
  
  // Calculate Drop Position relative to floor container
  const rect = floorRef.value.getBoundingClientRect()
  const clientX = event.clientX
  const clientY = event.clientY

  const relativeX = clientX - rect.left
  const relativeY = clientY - rect.top

  // Convert to percentage
  const percentX = (relativeX / rect.width) * 100
  const percentY = (relativeY / rect.height) * 100

  // Constrain check
  const finalX = Math.max(0, Math.min(100, percentX))
  const finalY = Math.max(0, Math.min(100, percentY))

  if (payload.type === 'template') {
    createNewAsset(payload.data, finalX, finalY)
  } else if (payload.type === 'asset') {
    updateAssetPosition(payload.id, finalX, finalY)
  }
}

const updateAssetPosition = (id: string, x: number, y: number) => {
    const asset = assets.value.find(a => a.id === id)
    if (asset) {
        asset.x = x
        asset.y = y
    }
}


// Map Asset Types to Blueprint IDs
const BLUEPRINT_IDS: Record<string, string> = {
  'machine': 'https://FraunhoferIPA/ids/aas/MachiningCenter',
  'metal_printer': 'https://fraunhoferIPA/ids/aas/Metal3DPrinter',
  'injection_molding': 'https://fraunhoferIPA/ids/aas/injectionMoldingMachine',
  'disassembly_robot': 'https://fraunhoferIPA.de/ids/aas/RoboticDisassemblyStation',
  'agv': 'https://admin-shell.io/idta/aas/AGV',
  'product': 'https://fraunhoferIPA/ids/aas/DigitalBatteryPassport',
  'process': 'https://h2giga.frhy/ids/aas/Process',
  'simulation': 'https://admin-shell.io/idta/aas/SimulationModels/1/0',
  'order': 'https://frauhoferIPA/ids/aas/ProductionOrder',
  'aps': 'https://fraunhoferIPA/ids/aas/aps',
  'assemblyshop': 'https://fraunhoferIPA/ids/aas/assemblyshop',
  'testing_repair': 'https://fraunhoferIPA/ids/aas/TestingRepair',
  'festo_cp_lab': 'https://fraunhoferIPA/ids/aas/FestoWorkCenterStamping'
}

import { instantiateAAS, deleteAAS, createConnection, deleteConnection, refreshManifests } from '../services/aasService'

const createNewAsset = async (template: AssetTemplate, x: number, y: number) => {
  const count = assets.value.filter(a => a.type === template.type).length + 1
  
  // Simple ID generation
  let prefix = ''
  switch (template.type) {
      case 'machine': prefix = 'MachiningCenter'; break;
      case 'metal_printer': prefix = 'Metal3DPrinter'; break;
      case 'injection_molding': prefix = 'InjectionMoldingMachine'; break;
      case 'disassembly_robot': prefix = 'RoboticDisassemblyStation'; break;
      case 'agv': prefix = 'AGV'; break;
      case 'order': prefix = 'ProductionOrder'; break;
      case 'product': prefix = 'DigitalBatteryPassport'; break;
      case 'process': prefix = 'Process'; break;
      case 'simulation': prefix = 'SimulationModel'; break;
      case 'aps': prefix = 'APS'; break;
      case 'assemblyshop': prefix = 'AssemblyShop'; break;
      case 'testing_repair': prefix = 'TestingRepairStation'; break;
      case 'festo_cp_lab': prefix = 'FestoWorkCenterStamping'; break;
      default: prefix = 'Asset';
  }
  
  // Random suffix for uniqueness
  const suffix = Math.floor(Math.random() * 1000).toString().padStart(3, '0')
  const newId = `${prefix}_${suffix}`
  
  const newAsset: AssetInstance = {
    id: newId,
    type: template.type,
    label: newId,
    x: Math.max(0, Math.min(100, x)), // Clamp to 0-100
    y: Math.max(0, Math.min(100, y)),
    hasAas: false
  }
  
  assets.value.push(newAsset)

  // Trigger AAS Instantiation
  const blueprintId = BLUEPRINT_IDS[template.type]
  
  if (blueprintId) {
    try {
      console.log(`Instantiating AAS for ${newId} from blueprint ${blueprintId}...`)
      // Set a visual indicator state if needed (e.g. flashing yellow)
      
      const response = await instantiateAAS(blueprintId, newId, newId) 
      
      // Update status on success
      const assetIndex = assets.value.findIndex(a => a.id === newId)
      if (assetIndex !== -1) {
        assets.value[assetIndex].hasAas = true
        assets.value[assetIndex].aasId = response.aas_id
        console.log(`AAS instantiated for ${newId}. Server ID: ${response.aas_id}`)
      }
    } catch (error) {
      console.error(`Failed to instantiate AAS for ${newId}:`, error)
    }
  }
}

// --- Interaction ---

const openAssetMenu = (asset: AssetInstance) => {
  selectedAsset.value = asset
}

const hasConsole = (asset: AssetInstance | null): boolean => {
  return !!asset && CONSOLE_TYPES.has(asset.type)
}

/**
 * Open the console on the *instance*, not on the blueprint.
 *
 * `aasId` is what instantiation returned, and it is what the console has to
 * read: a blueprint has no live machine behind it, and opening one on the
 * template would show an operator readings that belong to nothing.
 */
const openConsole = () => {
  if (!selectedAsset.value) return
  consoleAsset.value = selectedAsset.value
  selectedAsset.value = null
}

const consoleAsset = ref<AssetInstance | null>(null)

const handleAction = (mode: 'create' | 'edit' | 'view') => {
  if (!selectedAsset.value) return
  
  if (mode === 'view') {
      // Construct BaSyx UI URL
      // http://localhost:3000/?aas={server_url}/shells/{b64_id}
      const aasId = selectedAsset.value.aasId || "" // fallback?
      if (aasId) {
          const b64Id = btoa(aasId)
          // Hardcoded/Configured server URL (should ideally come from config)
          const serverUrl = envStore.aasServerUrl 
          const targetUrl = `${envStore.aasUiUrl}/?aas=${serverUrl}/shells/${b64Id}`
          window.open(targetUrl, '_blank')
      } else {
          console.warn("No AAS ID to view")
          alert("No AAS ID associated with this asset.")
      }
      return
  }
  
  store.enterWizard(mode, selectedAsset.value.id)
  selectedAsset.value = null
}

const resetShopFloor = async () => {
    if (confirm("Are you sure you want to reset the shop floor? This will DELETE ALL ASSETS and their corresponding AAS from the server.")) {
        // 1. Delete all assets from backend
        // We can do this in parallel
        const deletionPromises = assets.value.map(async (asset) => {
            if (asset.aasId) {
                try {
                    console.log(`Deleting AAS for ${asset.label} (${asset.aasId})...`)
                    await deleteAAS(asset.aasId)
                    console.log(`Deleted AAS: ${asset.aasId}`)
                } catch (e) {
                    console.error(`Failed to delete AAS ${asset.aasId}:`, e)
                }
            }
        })
        
        await Promise.all(deletionPromises)
        
        // 2. Clear frontend state
        assets.value = []
        connections.value = []
        selectedAsset.value = null
        selectedConnectionId.value = null
        
        
        console.log("Shop floor reset complete.")
    }
}

const triggerManifestRefresh = async () => {
    try {
        console.log("Refreshing AAS manifests...")
        const result = await refreshManifests()
        console.log("Manifests refreshed:", result.message)
        alert("Manifests successfully refreshed!")
    } catch (error) {
        console.error("Failed to refresh manifests:", error)
        alert("Failed to refresh manifests. Check console for details.")
    }
}





const getIcon = (type: string) => {
    const t = assetTemplates.find(t => t.type === type)
    return t ? t.icon : '📦'
}

// --- Connection Logic ---


import { onMounted, onUnmounted, reactive } from 'vue'

const floorContainer = ref<HTMLElement | null>(null)
const containerSize = reactive({ width: 0, height: 0 })

// Resize Observer to track container size
let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  
  if (floorRef.value) {
    resizeObserver = new ResizeObserver(entries => {
      for (let entry of entries) {
        containerSize.width = entry.contentRect.width
        containerSize.height = entry.contentRect.height
      }
    })
    resizeObserver.observe(floorRef.value)
  }
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
})

// Store connections
import type { Connection } from '../types'
const connections = computed({
  get: () => store.wizardData.connections,
  set: (newConns) => store.updateData('connections', newConns)
})

// Drag State
const dragConnection = reactive({
  active: false,
  sourceId: '',
  sourceType: '', // 'top', 'right', 'bottom', 'left'
  startX: 0,
  startY: 0,
  currentX: 0,
  currentY: 0
})

// Connection Selection
const selectedConnectionId = ref<string | null>(null)

const selectConnection = (id: string) => {
    selectedConnectionId.value = id
    selectedAsset.value = null // Deselect asset if any
}

// Global Key Listener for Deletion
const onKeyDown = (e: KeyboardEvent) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedConnectionId.value) {
        deleteSelectedConnection()
    }
}

const deleteSelectedConnection = async () => {
    if (selectedConnectionId.value) {
        // Find connection to get details before deletion
        const connToDelete = connections.value.find(c => c.id === selectedConnectionId.value)
        
        connections.value = connections.value.filter(c => c.id !== selectedConnectionId.value)
        selectedConnectionId.value = null
        
        // Backend Deletion
        if (connToDelete) {
             const sourceAsset = assets.value.find(a => a.id === connToDelete.sourceId)
             const targetAsset = assets.value.find(a => a.id === connToDelete.targetId)
             const relationshipId = (connToDelete as any).relationshipId
             
             if (sourceAsset?.aasId && targetAsset?.aasId && relationshipId) {
                 console.log(`Deleting relationship ${relationshipId} between ${sourceAsset.label} and ${targetAsset.label}...`)
                 try {
                     await deleteConnection(sourceAsset.aasId, targetAsset.aasId, relationshipId)
                     console.log("Relationship deleted successfully from backend.")
                 } catch (e) {
                     console.error("Failed to delete relationship from backend:", e)
                 }
             } else {
                 console.warn("Skipping backend relationship deletion: Missing AAS IDs or Relationship ID.")
             }
        }
    }
}

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  // ... existing resize observer ...
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  if (resizeObserver) resizeObserver.disconnect()
})


// Helper: Get Port Coordinates in Pixels
const getPortCoordinates = (assetId: string, port: string): { x: number, y: number } | null => {
  const asset = assets.value.find(a => a.id === assetId)
  if (!asset) return null

  // Asset center in pixels
  const cx = (asset.x / 100) * containerSize.width
  const cy = (asset.y / 100) * containerSize.height

  // Asset dimensions (fixed 100x100 for now)
  const size = 100
  const half = size / 2

  switch (port) {
    case 'top': return { x: cx, y: cy - half }
    case 'right': return { x: cx + half, y: cy }
    case 'bottom': return { x: cx, y: cy + half }
    case 'left': return { x: cx - half, y: cy }
    default: return { x: cx, y: cy }
  }
}

// Start Connection Drag
// Exposed to template
const startConnection = (event: MouseEvent, assetId: string, port: string) => {
  event.stopPropagation() // Prevent asset drag
  event.preventDefault()

  const coords = getPortCoordinates(assetId, port)
  if (!coords) return

  dragConnection.active = true
  dragConnection.sourceId = assetId
  dragConnection.sourceType = port
  dragConnection.startX = coords.x
  dragConnection.startY = coords.y
  
  // Initial current position is same as start
  dragConnection.currentX = coords.x
  dragConnection.currentY = coords.y

  // Add global listeners
  window.addEventListener('mousemove', onConnectionDrag)
  window.addEventListener('mouseup', onConnectionDropOutside)
}

// Drag Move
const onConnectionDrag = (event: MouseEvent) => {
  if (!dragConnection.active || !floorRef.value) return

  const rect = floorRef.value.getBoundingClientRect()
  dragConnection.currentX = event.clientX - rect.left
  dragConnection.currentY = event.clientY - rect.top
}

// Drop on Target Port (Success)
// Exposed to template
const endConnection = (event: MouseEvent, targetId: string, targetPort: string) => {
  event.stopPropagation()
  
  if (!dragConnection.active) return

  // Constraints
  if (dragConnection.sourceId === targetId) {
    console.warn("Cannot connect node to itself")
    cancelConnection()
    return
  }

  // Check duplicate
  const exists = connections.value.some(c => 
    (c.sourceId === dragConnection.sourceId && c.targetId === targetId) ||
    (c.sourceId === targetId && c.targetId === dragConnection.sourceId) // Undirected or bi-directional check? Requirement says "Duplicate connections... not permitted". Assuming directed for now but unique pair.
  )
  
  if (exists) {
     console.warn("Connection already exists")
     cancelConnection()
     return
  }

  // Create Connection
  const newConn: Connection = {
    id: `c_${Date.now()}`,
    sourceId: dragConnection.sourceId,
    targetId: targetId,
    sourcePort: dragConnection.sourceType as any,
    targetPort: targetPort as any
  }
  
  connections.value.push(newConn)
  cancelConnection() // Cleanup listeners

  // Trigger Backend Relationship Creation
  const sourceAsset = assets.value.find(a => a.id === newConn.sourceId)
  const targetAsset = assets.value.find(a => a.id === newConn.targetId)

  if (sourceAsset?.aasId && targetAsset?.aasId) {
      console.log(`Creating AAS Relationship between ${sourceAsset.label} (${sourceAsset.aasId}) and ${targetAsset.label} (${targetAsset.aasId})...`)
      createConnection(sourceAsset.aasId, targetAsset.aasId)
        .then(response => {
           console.log("Relationship created successfully:", response)
           // Update connection with Relationship ID
           const connIndex = connections.value.findIndex(c => c.id === newConn.id)
           if (connIndex !== -1) {
               // We need to extend the type? 
               // For now, let's just add it dynamically or we should update types.ts
               // Assuming we can add property or it's flexible.
               // Ideally update types.ts
               (connections.value[connIndex] as any).relationshipId = response.relationship_sm_id
           }
        })
        .catch(error => {
           console.error("Failed to create relationship in backend:", error)
        })
  } else {
      console.warn("Visual connection created, but AAS Relationship skipped because one or both assets rely on missing AAS IDs.")
  }
}

// Drop Outside (Cancel)
const onConnectionDropOutside = () => {
  cancelConnection()
}

const cancelConnection = () => {
  dragConnection.active = false
  window.removeEventListener('mousemove', onConnectionDrag)
  window.removeEventListener('mouseup', onConnectionDropOutside)
}


// Computed Paths for Rendering
const renderedConnections = computed(() => {
  return connections.value.map(conn => {
    const start = getPortCoordinates(conn.sourceId, conn.sourcePort)
    const end = getPortCoordinates(conn.targetId, conn.targetPort)
    
    if (!start || !end) return null

    // Bezier Curve Logic
    // Control points depend on port direction
    const dx = Math.abs(end.x - start.x)
    const dy = Math.abs(end.y - start.y)
    const curvature = Math.min(100, Math.max(dx, dy) / 2) // smooth curvature

    // Simple heuristic for control points: extend out from port
    const getControl = (x: number, y: number, port: string, dist: number) => {
       switch(port) {
         case 'top': return { x, y: y - dist }
         case 'bottom': return { x, y: y + dist }
         case 'left': return { x: x - dist, y }
         case 'right': return { x: x + dist, y }
         default: return { x, y }
       }
    }

    const cp1 = getControl(start.x, start.y, conn.sourcePort, curvature)
    const cp2 = getControl(end.x, end.y, conn.targetPort, curvature)

    return {
      id: conn.id,
      d: `M ${start.x} ${start.y} C ${cp1.x} ${cp1.y}, ${cp2.x} ${cp2.y}, ${end.x} ${end.y}`,
      color: '#555',
      isSelected: conn.id === selectedConnectionId.value
    }
  }).filter(c => c !== null)
})

const previewPath = computed(() => {
  if (!dragConnection.active) return ''
  
  const startX = dragConnection.startX
  const startY = dragConnection.startY
  const endX = dragConnection.currentX
  const endY = dragConnection.currentY
  
  // Use a simple curve to mouse
  return `M ${startX} ${startY} L ${endX} ${endY}`
})

// Update deleteAsset to remove connections
const deleteAsset = async () => {
    if (selectedAsset.value) {
        const idToDelete = selectedAsset.value.id
        const aasIdToDelete = selectedAsset.value.aasId
        
        // Remove associated connections
        connections.value = connections.value.filter(c => c.sourceId !== idToDelete && c.targetId !== idToDelete)

        // Optimistic UI update: Remove from UI immediately
        assets.value = assets.value.filter(a => a.id !== idToDelete)
        selectedAsset.value = null
        
        // Trigger AAS Deletion
        if (aasIdToDelete) {
            try {
                console.log(`Deleting AAS for ${idToDelete} (AAS ID: ${aasIdToDelete})...`)
                await deleteAAS(aasIdToDelete)
                console.log(`AAS deleted for ${idToDelete}`)
            } catch (error) {
                console.error(`Failed to delete AAS for ${idToDelete}:`, error)
            }
        } else {
             console.warn(`No AAS ID found for ${idToDelete}, skipping server deletion.`)
        }
    }
}

</script>

<style scoped>
.shop-floor-layout {
    display: flex;
    height: 100%;
    width: 100%;
    background-color: #f8f9fa;
}

.asset-library {
    width: 250px;
    background: white;
    border-right: 1px solid #ddd;
    padding: 20px;
    display: flex;
    flex-direction: column;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    z-index: 10;
    overflow-y: auto;
}

.asset-library h3 {
    margin: 0 0 5px 0;
    font-size: 1.2em;
    color: #333;
}

.asset-library .subtitle {
    margin: 0 0 20px 0;
    font-size: 0.9em;
    color: #888;
}

.asset-template {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    margin-bottom: 10px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    cursor: grab;
    transition: all 0.2s;
}

.asset-template:hover {
    border-color: #007bff;
    background: #e7f1ff;
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.asset-template:active {
    cursor: grabbing;
}

.template-icon-img {
    width: 32px;
    height: 32px;
    object-fit: contain;
}

.template-label {
    font-weight: 500;
    font-size: 0.95em;
    color: #333;
}


.shop-floor-container {
  display: flex;
  flex-direction: column;
  flex: 1; /* Take remaining width */
  padding: 20px;
  box-sizing: border-box;
}

.shop-floor-header {
  margin-bottom: 20px;
  text-align: center;
}

.floor-visual {
  flex: 1;
  position: relative;
  background-color: #fff;
  border: 2px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: inset 0 0 20px rgba(0,0,0,0.05);
}

.grid-bg {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image: linear-gradient(#f0f0f0 1px, transparent 1px),
                    linear-gradient(90deg, #f0f0f0 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

.asset {
  position: absolute;
  width: 100px;
  height: 100px;
  background-color: white;
  border: 2px solid #333;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  user-select: none;
  transform: translate(-50%, -50%); /* Center on coordinates */
}

/* Asset Types Styling Variations - Removed as requested for uniform look */
/* All assets now share the base .asset style */
.asset {
  /* ... base styles ... */
  border-color: #333; /* Default uniform color */
}

.asset:hover {
  transform: translate(-50%, -50%) scale(1.05);
  box-shadow: 0 8px 12px rgba(0,0,0,0.15);
  border-color: #007bff;
  z-index: 100;
}

.asset-icon-img {
    width: 48px;
    height: 48px;
    object-fit: contain;
    margin-bottom: 5px;
}

.asset-label {
  font-size: 0.8em;
  font-weight: bold;
  text-align: center;
  max-width: 90%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  position: absolute;
  top: 5px;
  right: 5px;
  border: 1px solid rgba(0,0,0,0.1);
}

.has-aas { background-color: #28a745; } /* Green for AAS */
.no-aas { background-color: #ffc107; }   /* Yellow for No AAS */



.connections-layer {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Let clicks pass through */
  z-index: 1;
}

.connections-layer path {
  pointer-events: stroke; 
}
.connection-hit {
    cursor: pointer;
}


.asset {
  /* ... previous styles ... */
  z-index: 2; /* Assets above lines */
}

/* Ports */
.port {
  position: absolute;
  width: 12px;
  height: 12px;
  background-color: #666;
  border-radius: 50%;
  border: 2px solid white;
  opacity: 0;
  transition: opacity 0.2s, background-color 0.2s;
  cursor: crosshair;
  z-index: 3;
}

.asset:hover .port {
  opacity: 1;
}

/* Position Ports */
.port-top { top: -6px; left: 50%; transform: translateX(-50%); }
.port-right { right: -6px; top: 50%; transform: translateY(-50%); }
.port-bottom { bottom: -6px; left: 50%; transform: translateX(-50%); }
.port-left { left: -6px; top: 50%; transform: translateY(-50%); }

.port:hover {
  background-color: #007bff;
  transform: scale(1.2); /* Basic scaling, verify transform combos */
}

/* Fix transform conflicts */
.port-top:hover { transform: translateX(-50%) scale(1.2); }
.port-right:hover { transform: translateY(-50%) scale(1.2); }
.port-bottom:hover { transform: translateX(-50%) scale(1.2); }
.port-left:hover { transform: translateY(-50%) scale(1.2); }


.asset-modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(0,0,0,0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.asset-modal {
  background-color: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.2);
  width: 300px;
  text-align: center;
  position: relative;
}

.modal-actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 20px;
}

.btn {
  padding: 10px 15px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.2s;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}
.btn-primary:hover {
  background-color: #0056b3;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}
.btn-secondary:hover {
  background-color: #5a6268;
}

.btn-danger {
  background-color: #dc3545;
  color: white;
}
.btn-danger:hover {
  background-color: #bd2130;
}

.btn-info {
  background-color: #17a2b8;
  color: white;
}
.btn-info:hover {
  background-color: #138496;
}


.close-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  background: none;
  border: none;
  font-size: 1.5em;
  cursor: pointer;
  color: #999;
}
.close-btn:hover {
  color: #333;
}
</style>
