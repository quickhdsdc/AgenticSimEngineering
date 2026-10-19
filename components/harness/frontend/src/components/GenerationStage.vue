<template>
  <div class="generation-stage">
    <h2 class="section-title">AAS Population</h2>

    <div v-if="data.isGenerating" class="progress-indicator">
      <div class="spinner"></div>
      <h3>Populating AAS...</h3>
    </div>

    <div v-else-if="data.generationComplete" class="success-message">
      <div class="success-icon">✓</div>
      <h2 style="color: #28a745; margin-bottom: 10px">AAS Populated Successfully!</h2>
      

      <div style="margin-top: 40px; padding: 30px; background: #f8f9fa; border-radius: 8px; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto">
        <h3 style="margin-bottom: 20px; color: #333">Population Details</h3>

        <div style="margin-bottom: 15px">
          <strong style="color: #666; font-size: 13px">Template:</strong>
          <div style="margin-top: 4px; color: #333">{{ data.submodelTemplate }}</div>
        </div>

        <div style="margin-bottom: 15px">
          <strong style="color: #666; font-size: 13px">Target AAS:</strong>
          <div style="margin-top: 4px; color: #333; word-break: break-all">{{ data.targetAAS }}</div>
        </div>

        <div style="margin-bottom: 15px">
          <strong style="color: #666; font-size: 13px">Submodel ID:</strong>
          <div style="margin-top: 4px; color: #333; word-break: break-all">{{ data.submodelId }}</div>
        </div>

        <div style="margin-bottom: 15px">
          <strong style="color: #666; font-size: 13px">ID Short:</strong>
          <div style="margin-top: 4px; color: #333">{{ data.submodelIdShort }}</div>
        </div>

        <div>
          <strong style="color: #666; font-size: 13px">Properties Populated:</strong>
          <div style="margin-top: 4px; color: #333">{{ data.matchedProperties.length }} properties</div>
        </div>
      </div>

      <div class="download-links">
        <button class="btn btn-primary" @click="downloadAASX">
          📦 Download AASX
        </button>
        <button class="btn btn-primary" @click="downloadJSON">
          📄 Download JSON
        </button>
        <button class="btn btn-secondary" @click="viewOnServer">
          🔗 View on Server
        </button>
      </div>

      <div style="margin-top: 40px">
        <button class="btn btn-success" @click="emit('reset')">
          ✨ Populate Another AAS
        </button>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { useEnvStore } from '@/stores/environmentStore'
import { aasService } from '@/services/aasService'
import { ApiError } from '@/services/api'
import type { WizardData } from '@/types'

// Props
interface Props {
  data: WizardData
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  reset: []
}>()

// Environment store
const environmentStore = useEnvStore()

const downloadAASX = async (): Promise<void> => {
  console.log('Downloading AASX package...')
  const aasIdShort = props.data.targetAAS
  const endpoint = props.data.aasServerEndpoint || environmentStore.aasServerUrl

  try {
    const blob = await aasService.downloadAASX(aasIdShort, endpoint)
    aasService.triggerDownload(blob, `${aasIdShort}.aasx`)
  } catch (error) {
    console.error('Failed to download AASX:', error)
    const errorMsg = error instanceof ApiError
      ? `Download failed: ${error.message}`
      : 'Failed to download AASX file. Please try again.'
    alert(errorMsg)
  }
}

const downloadJSON = async (): Promise<void> => {
  console.log('Downloading JSON...')
  const aasIdShort = props.data.targetAAS
  const endpoint = props.data.aasServerEndpoint || environmentStore.aasServerUrl

  try {
    const blob = await aasService.downloadJSON(aasIdShort, endpoint)
    aasService.triggerDownload(blob, `${aasIdShort}.json`)
  } catch (error) {
    console.error('Failed to download JSON:', error)
    const errorMsg = error instanceof ApiError
      ? `Download failed: ${error.message}`
      : 'Failed to download JSON file. Please try again.'
    alert(errorMsg)
  }
}

const viewOnServer = async (): Promise<void> => {
  console.log('Opening BaSyx UI...')
  const uiEndpoint = props.data.aasUIEndpoint || environmentStore.aasUiUrl
  const serverUrl = props.data.aasServerEndpoint || environmentStore.aasServerUrl
  const aasIdShort = props.data.targetAAS

  // Try to deep-link straight to the populated shell, matching the ShopFloor
  // "view" pattern: {uiUrl}/?aas={serverUrl}/shells/{base64(aasId)}.
  try {
    if (aasIdShort) {
      const res = await aasService.checkAAS(aasIdShort, serverUrl) as { exists?: boolean; aas_id?: string; id?: string }
      const aasId = res?.aas_id || res?.id
      if (aasId) {
        const b64Id = btoa(aasId)
        window.open(`${uiEndpoint}/?aas=${serverUrl}/shells/${b64Id}`, '_blank')
        return
      }
    }
  } catch (error) {
    console.warn('Could not resolve AAS id for deep-link, opening BaSyx UI root instead:', error)
  }

  // Fallback: open the BaSyx UI root.
  window.open(uiEndpoint, '_blank')
}
</script>
