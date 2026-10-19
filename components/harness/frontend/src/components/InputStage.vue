<template>
  <div class="input-stage">
    <!-- Block 1: Endpoint Configuration -->
    <h2 class="section-title">Endpoint Configuration</h2>
    
    <div class="form-group">
      <label for="aasServerEndpoint">AAS Server Endpoint *</label>
      <div class="input-with-button">
        <input
          id="aasServerEndpoint"
          type="text"
          :value="data.aasServerEndpoint"
          @input="emit('update', 'aasServerEndpoint', $event.target.value)"
          :placeholder="environmentStore.aasServerUrl"
        />
        <button 
          class="btn btn-secondary" 
          @click="refreshData"
          :disabled="templatesLoading || aasListLoading"
          title="Refresh Templates and AAS List"
        >
          Refresh
        </button>
      </div>
      <HelpTip text="Enter the AAS server API endpoint. Click Refresh to reload templates and existing AASs." />
    </div>

    <div class="form-group">
      <label for="aasUIEndpoint">BaSyx UI Endpoint</label>
      <input
        id="aasUIEndpoint"
        type="text"
        :value="data.aasUIEndpoint"
        @input="emit('update', 'aasUIEndpoint', $event.target.value)"
        :placeholder="environmentStore.aasUiUrl"
      />
      <HelpTip text="Enter the BaSyx web UI endpoint for viewing the AAS" />
    </div>

    <!-- Block 2: AAS Configuration -->
    <h2 class="section-title" style="margin-top: 40px">AAS Configuration</h2>

    <!-- Mode Selection -->
    <div class="form-group">
      <label>Mode</label>
      <div class="tabs" style="margin-bottom: 20px">
        <button
          :class="['tab', { active: data.mode === 'create' }]"
          @click="setMode('create')"
        >
          Create New AAS
        </button>
        <button
          :class="['tab', { active: data.mode === 'edit' }]"
          @click="setMode('edit')"
        >
          Edit Existing AAS
        </button>
      </div>
    </div>

    <!-- Create Mode Content -->
    <div v-if="data.mode === 'create'" class="mode-content">
      <!-- Target AAS ID (Moved to top as requested) -->
      <div class="form-group">
        <label for="targetAAS">Target AAS ID Short *</label>
        <input
          id="targetAAS"
          type="text"
          :value="data.targetAAS"
          @input="emit('update', 'targetAAS', $event.target.value)"
          placeholder="NewProductAAS"
        />
        <HelpTip text="Enter the ID Short for the new AAS (e.g. NewProductAAS)" />
      </div>

      <!-- Submodel Templates -->
      <div class="form-group">
        <label>Submodel Templates * (select one or more)</label>
        <div v-if="templatesLoading" class="loading-state">Loading templates...</div>
        <div v-else-if="templatesError" class="error-state">{{ templatesError }}</div>
        <div v-else class="template-list">
          <label
            v-for="template in availableTemplates"
            :key="template.id"
            class="template-item"
            :class="{ selected: isTemplateSelected(template.id) }"
          >
            <input
              type="checkbox"
              :value="template.id"
              :checked="isTemplateSelected(template.id)"
              @change="toggleTemplate(template.id)"
            />
            <div class="template-info">
              <strong>{{ template.name }}</strong>
              <HelpTip v-if="template.description" :text="template.description" />
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- Edit Mode Content -->
    <div v-if="data.mode === 'edit'" class="mode-content">
      <!-- Select Target AAS -->
      <div class="form-group">
        <label>Select Target AAS *</label>
        
        <div v-if="aasListLoading" class="loading-state">
          Loading existing AASs...
        </div>
        <div v-else-if="aasListError" class="error-state">
          {{ aasListError }}
        </div>
        <div v-else class="template-list">
          <label
            v-for="aas in aasList"
            :key="aas.id"
            class="template-item"
            :class="{ selected: data.targetAAS === aas.idShort }"
          >
            <input
              type="radio"
              name="targetAAS"
              :value="aas.idShort"
              :checked="data.targetAAS === aas.idShort"
              @change="onAASSelect"
            />
            <div class="template-info">
              <strong>{{ aas.idShort }}</strong>
              <span class="template-description">{{ aas.id }}</span>
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- Block 3: Data Source (Header already exists in next lines) -->

    <h2 class="section-title" style="margin-top: 40px">Data Source</h2>

    <div class="tabs">
      <button
        :class="['tab', { active: data.dataSourceType === 'url' }]"
        @click="setDataSourceType('url')"
      >
        Website URL
      </button>
      <button
        :class="['tab', { active: data.dataSourceType === 'pdf' }]"
        @click="setDataSourceType('pdf')"
      >
        PDF Document
      </button>
      <button
        :class="['tab', { active: data.dataSourceType === 'structured' }]"
        @click="setDataSourceType('structured')"
      >
        Structured File
      </button>
      <button
        :class="['tab', { active: data.dataSourceType === 'text' }]"
        @click="setDataSourceType('text')"
      >
        Text Input
      </button>
    </div>

    <div v-if="data.dataSourceType === 'text'" class="tab-content">
      <div class="alert alert-info">
        Paste the text content directly to extract technical information.
      </div>
      <textarea
        :value="data.dataSourceValue"
        @input="emit('update', 'dataSourceValue', ($event.target as HTMLTextAreaElement).value)"
        placeholder="Paste specification text here..."
        style="width: 100%; min-height: 200px; padding: 16px; font-size: 14px; border-radius: 8px; border: 2px solid #ddd; resize: vertical;"
      ></textarea>
    </div>

    <div v-if="data.dataSourceType === 'url'" class="tab-content">
      <div class="alert alert-info">
        Enter the URL of a product page or datasheet to extract technical information.
      </div>
      <div class="input-with-button">
        <input
          type="url"
          :value="data.dataSourceValue"
          @input="emit('update', 'dataSourceValue', $event.target.value)"
          placeholder="https://example.com/products/motor-xm2000"
          style="flex: 1; padding: 16px; font-size: 18px; height: 60px; border-radius: 8px; border: 2px solid #ddd;"
        />
      </div>
    </div>

    <div v-if="data.dataSourceType === 'pdf' || data.dataSourceType === 'structured'" class="tab-content">
      <div class="alert alert-info">
        {{ data.dataSourceType === 'structured' ? 
           'Upload a CSV or JSON file with "entity", "value", and "description" columns.' : 
           'Upload a PDF datasheet to extract product specifications and technical data.' 
        }}
      </div>

      <div
        v-if="!data.uploadedFiles || data.uploadedFiles.length === 0"
        class="upload-area"
        :class="{ 'drag-over': isDragging }"
        @click="triggerFileInput"
        @dragover.prevent="isDragging = true"
        @dragenter.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="handleDrop"
      >
        <div class="upload-icon">📄</div>
        <h3>Click to upload {{ data.dataSourceType === 'structured' ? 'File(s)' : 'PDF(s)' }}</h3>
        <p>or drag and drop multiple files</p>
        <p style="color: #999; font-size: 12px; margin-top: 8px">
            {{ data.dataSourceType === 'structured' ? 'Supported: .csv, .json' : 'Maximum file size: 10MB per file' }}
        </p>
      </div>

      <div v-else class="uploaded-files-container">
        <div v-for="(file, index) in data.uploadedFiles" :key="index" class="uploaded-file">
          <div class="file-icon">📄</div>
          <div style="flex: 1">
            <strong>{{ file.name }}</strong>
            <p style="color: #666; font-size: 12px; margin-top: 4px">
              {{ formatFileSize(file.size) }}
            </p>
          </div>
          <button class="btn btn-secondary" @click="removeFile(index)">Remove</button>
        </div>
        <button class="btn btn-primary" @click="triggerFileInput" style="margin-top: 12px">
          Add More Files
        </button>
      </div>

      <input
        ref="fileInput"
        type="file"
        :accept="data.dataSourceType === 'structured' ? '.csv, .json, application/json, text/csv' : '.pdf, application/pdf'"
        multiple
        class="file-input"
        @change="handleFileUpload"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, type Ref, watch } from 'vue'
import { useEnvStore } from '@/stores/environmentStore'
import { fileService } from '@/services/fileService'
import { aasService } from '@/services/aasService'
import { ApiError } from '@/services/api'
import type { WizardData, SubmodelTemplate } from '@/types'

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
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)

// AAS List Logic
const aasList = ref<any[]>([])
const aasListLoading = ref(false)
const aasListError = ref<string | null>(null)

const fetchAASList = async () => {
    aasListLoading.value = true
    aasListError.value = null
    try {
        const endpoint = props.data.aasServerEndpoint || environmentStore.aasServerUrl
        const list = await aasService.getAASList(endpoint)
        
        // Filter: only show Instance AAS (not Type/Template blueprints or SMT collections)
        aasList.value = list.filter((aas: any) => {
            const idShort = aas.idShort || ''
            const assetKind = aas.assetKind || 'Unknown'
            
            const isCollection = idShort.toLowerCase().includes('collectionofsmt')
            const isInstance = assetKind === 'Instance'
            
            return !isCollection && isInstance
        })
    } catch (e: any) {
        console.error("Failed to fetch AAS list", e)
        aasListError.value = e.message || "Failed to load AAS list"
    } finally {
        aasListLoading.value = false
    }
}

const refreshData = async () => {
    await fetchTemplates()
    if (props.data.mode === 'edit') {
        await fetchAASList()
    }
}

const setMode = (mode: 'create' | 'edit') => {
    emit('update', 'mode', mode)
    if (mode === 'edit') {
        fetchAASList()
    }
}

const onAASSelect = (event: Event) => {
    const target = event.target as HTMLSelectElement
    const idShort = target.value
    emit('update', 'targetAAS', idShort)
    
    // Find selected AAS to stick endpoint if needed
    const selectedAAS = aasList.value.find(a => a.idShort === idShort)
    if (selectedAAS && selectedAAS.endpoint) {
         emit('update', 'aasServerEndpoint', selectedAAS.endpoint)
    }
}

// Watch endpoint changes to refresh list in edit mode
watch(() => props.data.aasServerEndpoint, (newVal) => {
    if (props.data.mode === 'edit' && newVal) {
        // Debounce could be added here, but for now relying on manual refresh button 
        // or just letting it fire (though it might be spammy)
        // actually, user asked for button, so maybe we SHOULD remove auto-watch?
        // But keeping it for now as "live" feel is often nice, but if user explicitly asked for button...
        // "should add a button allowing user to change the default endpoint to re-fetch"
        // This implies the fetch shouldn't happen automatically on type, potentially.
        // But let's keep it for now, it's safer functionality-wise.
        // fetchAASList() 
    }
})

// Data Source Logic
const setDataSourceType = (type: 'url' | 'pdf' | 'text' | 'structured') => {
    // Only update if changing
    if (props.data.dataSourceType === type) return

    emit('update', 'dataSourceType', type)
    
    // Clear data from other mode
    emit('update', 'dataSourceValue', '')
    emit('update', 'uploadedFiles', [])
    
    // Reset file input if exists
    if (fileInput.value) {
        fileInput.value.value = ''
    }
}

// Available submodel templates - fetched from backend
const availableTemplates: Ref<SubmodelTemplate[]> = ref([])
const templatesLoading = ref(true)
const templatesError: Ref<string | null> = ref(null)

// Fetch templates from backend on mount
const fetchTemplates = async (): Promise<void> => {
  try {
    templatesLoading.value = true
    templatesError.value = null

    const endpoint = props.data.aasServerEndpoint || environmentStore.aasServerUrl
    const templates = await fileService.getTemplates(endpoint)
    availableTemplates.value = templates
  } catch (error) {
    console.error('Error fetching templates:', error)
    templatesError.value = error instanceof ApiError ? error.message : 'Failed to load templates'

    // Fallback to mock data if API fails
    availableTemplates.value = [
      {
        id: 'https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2',
        name: 'TechnicalData',
        idShort: 'TechnicalData',
        version: '1.2',
        description: 'IDTA 02003 - Technical specifications and parameters'
      },
      {
        id: 'https://admin-shell.io/zvei/nameplate/2/0/Nameplate',
        name: 'Nameplate',
        idShort: 'Nameplate',
        version: '2.0',
        description: 'IDTA 02006 - Product identification and nameplate data'
      },
      {
        id: 'https://admin-shell.io/idta/ContactInformation/1/0',
        name: 'ContactInformation',
        idShort: 'ContactInformation',
        version: '1.0',
        description: 'IDTA 02002 - Contact details for manufacturers and suppliers'
      },
      {
        id: 'https://admin-shell.io/idta/Documentation/1/0',
        name: 'Documentation',
        idShort: 'Documentation',
        version: '1.0',
        description: 'IDTA 02004 - Technical documentation and manuals'
      },
      {
        id: 'https://admin-shell.io/idta/CarbonFootprint/0/9',
        name: 'CarbonFootprint',
        idShort: 'CarbonFootprint',
        version: '0.9',
        description: 'IDTA 02023 - Environmental impact and carbon footprint data'
      },
      {
        id: 'https://admin-shell.io/idta/TimeSeries/1/1',
        name: 'TimeSeries',
        idShort: 'TimeSeries',
        version: '1.1',
        description: 'IDTA 02008 - Time series measurements and sensor data'
      }
    ]
  } finally {
    templatesLoading.value = false
  }
}

// Fetch templates on component mount
fetchTemplates()

// Check if a template is selected
const isTemplateSelected = (templateId: string): boolean => {
  if (Array.isArray(props.data.submodelTemplate)) {
    return props.data.submodelTemplate.includes(templateId)
  }
  return props.data.submodelTemplate === templateId
}

// Toggle template selection
const toggleTemplate = (templateId: string): void => {
  let selectedTemplates: string[] = Array.isArray(props.data.submodelTemplate)
    ? [...props.data.submodelTemplate]
    : props.data.submodelTemplate
    ? [props.data.submodelTemplate]
    : []

  const index = selectedTemplates.indexOf(templateId)
  if (index > -1) {
    // Remove if already selected
    selectedTemplates.splice(index, 1)
  } else {
    // Add if not selected
    selectedTemplates.push(templateId)
  }

  emit('update', 'submodelTemplate', selectedTemplates)
}

const triggerFileInput = (): void => {
  fileInput.value?.click()
}

const handleDrop = async (event: DragEvent): Promise<void> => {
  isDragging.value = false
  if (!event.dataTransfer) return

  const files = Array.from(event.dataTransfer.files)
  
  // Define allowed types based on mode
  const isStructured = props.data.dataSourceType === 'structured'
  
  // Filter relevant files
  const validFiles = files.filter(f => {
      if (isStructured) {
          const name = f.name.toLowerCase()
          return name.endsWith('.csv') || name.endsWith('.json')
      }
      return f.type === 'application/pdf'
  })

  const typeLabel = isStructured ? 'CSV or JSON' : 'PDF'

  if (validFiles.length === 0) {
    alert(`Please drop ${typeLabel} files only`)
    return
  }

  if (validFiles.length !== files.length) {
    alert(`Some files were skipped (only ${typeLabel} files are accepted)`)
  }

  await processFiles(validFiles)
}

const processFiles = async (files: File[]): Promise<void> => {
  // Create a copy of existing files array to avoid mutation
  const currentFiles = [...(props.data.uploadedFiles || [])]
  const filePaths: string[] = []
  
  const isStructured = props.data.dataSourceType === 'structured'
  const allowedTypes = isStructured 
      ? ['.csv', '.json', 'text/csv', 'application/json', 'application/vnd.ms-excel'] 
      : ['application/pdf']

  for (const file of files) {
    // Validate file before upload
    const validation = fileService.validateFile(file, {
      maxSize: 10 * 1024 * 1024, // 10MB
      allowedTypes: allowedTypes
    })

    if (!validation.valid) {
      alert(`${file.name}: ${validation.error}`)
      continue
    }

    // Upload file to server
    try {
      const result = await fileService.uploadFile(file)
      console.log('File uploaded successfully:', result)

      // Add to uploaded files list (store serializable metadata)
      currentFiles.push({
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified
      })
      filePaths.push(result.path)
    } catch (error) {
      console.error('File upload failed:', error)
      const errorMsg = error instanceof ApiError
        ? `${file.name}: Upload failed: ${error.message}`
        : `${file.name}: Failed to upload file. Please try again.`
      alert(errorMsg)
    }
  }

  // Update state with all uploaded files
  emit('update', 'uploadedFiles', currentFiles)

  // Update dataSourceValue with comma-separated file paths for the agent
  // Append new paths to existing ones
  const existingPaths = props.data.dataSourceValue ? props.data.dataSourceValue.split(',') : []
  const allPaths = [...existingPaths, ...filePaths]
  emit('update', 'dataSourceValue', allPaths.join(','))
}

const handleFileUpload = async (event: Event): Promise<void> => {
  const target = event.target as HTMLInputElement
  if (!target.files) return

  const files = Array.from(target.files)
  const isStructured = props.data.dataSourceType === 'structured'
  
  const validFiles = files.filter(f => {
      if (isStructured) {
          const name = f.name.toLowerCase()
          return name.endsWith('.csv') || name.endsWith('.json')
      }
      return f.type === 'application/pdf'
  })

  if (validFiles.length > 0) {
    await processFiles(validFiles)
  }

  // Reset file input so the same file can be selected again if needed
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const removeFile = (index: number): void => {
  const currentFiles = [...(props.data.uploadedFiles || [])]
  currentFiles.splice(index, 1)

  emit('update', 'uploadedFiles', currentFiles)

  // If no files left, clear dataSourceValue
  if (currentFiles.length === 0) {
    emit('update', 'dataSourceValue', '')
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  }
}



const formatFileSize = fileService.formatFileSize
</script>

<style scoped>
.loading-state,
.error-state {
  padding: 20px;
  text-align: center;
  background: #f5f5f5;
  border-radius: 8px;
  color: #666;
}

.error-state {
  color: #d32f2f;
  background: #ffebee;
}

.small-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
  margin-top: 10px;
}

.template-item.compact {
  padding: 10px;
  min-height: auto;
}

.template-item.compact .template-info strong {
  font-size: 13px;
}

/* Fix for radio buttons inheriting 100% width from form-group */
.template-item input[type="radio"] {
  width: 18px;
  height: 18px;
  margin-top: 2px;
  cursor: pointer;
  flex-shrink: 0;
}
</style>
