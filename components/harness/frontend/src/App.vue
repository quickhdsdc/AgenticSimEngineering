<template>
  <div :class="{'industrial-mode': activeWorkspace === 'recovery'}">
    <nav class="workspace-navigation">
      <strong>A4S <small>CONSOLE</small></strong>
      <button :class="{ selected: activeWorkspace === 'recovery' }" @click="activeWorkspace = 'recovery'">Operations</button>
      <button :class="{ selected: activeWorkspace === 'floor' }" @click="activeWorkspace = 'floor'">Shop floor & AAS</button>
      <a v-if="environmentStore.mesUiUrl" :href="environmentStore.mesUiUrl" target="_blank" rel="noopener noreferrer" style="margin-left:auto;color:inherit">OpenMES ↗</a>
    </nav>
    <div class="main-container" ref="splitContainer">
      <RecoveryWorkspace ref="recoveryWorkspaceRef" v-if="activeWorkspace === 'recovery'" :agent-locked="operationsAgentLocked" :style="{ width: leftPanelWidth + 'px', flex: 'none' }" />
      <!-- Shop Floor View -->
      <ShopFloor 
        v-else-if="wizardData.viewMode === 'shop-floor'"
        :style="{ width: leftPanelWidth + 'px', flex: 'none' }"
      />

      <!-- Wizard View -->
      <div v-else class="wizard-container" :style="{ width: leftPanelWidth + 'px' }">
        <div class="wizard-header" style="padding: 10px 30px; border-bottom: none; display: flex; justify-content: space-between; align-items: center;">
          <!-- Header text removed as requested -->
          <button class="btn btn-sm btn-outline-secondary" @click="store.exitWizard">
            ← Back to Floor
          </button>
        </div>

        <div class="wizard-progress">
          <div
            v-for="(step, index) in steps"
            :key="index"
            :class="['progress-step', { active: currentStep === index, completed: currentStep > index }]"
            @click="store.setStep(index)"
          >
            <div class="step-circle">{{ index + 1 }}</div>
            <div class="step-label">{{ step }}</div>
          </div>
        </div>

        <div class="wizard-content">
          <InputStage v-if="currentStep === 0" :data="wizardData" @update="store.updateData" />
          <ExtractionStage v-if="currentStep === 1" :data="wizardData" @update="store.updateData" />
          <MatchingStage v-if="currentStep === 2" :data="wizardData" @update="store.updateData" />
          <GenerationStage v-if="currentStep === 3" :data="wizardData" @reset="store.resetPopulation" />
        </div>

        <div class="wizard-actions">
          <button
            v-if="currentStep > 0 && currentStep < 3"
            class="btn btn-secondary"
            @click="store.previousStep"
          >
            ← Previous
          </button>
          <div v-else></div>

          <button
            v-if="currentStep < 3"
            class="btn btn-primary"
            @click="nextStep"
            :disabled="!canProceed || isProcessing"
          >
            {{ isProcessing ? 'Processing...' : (currentStep === 2 ? 'Populate AAS' : 'Next →') }}
          </button>
        </div>
      </div>

      <!-- Resizable divider -->
      <div
        class="resizer"
        @pointerdown="startResize"
        role="separator"
        aria-label="Resize operations and chat panels"
        aria-orientation="vertical"
        title="Drag to resize"
      ></div>

      <ChatWindow ref="chatWindowRef" @agentComplete="onAgentComplete" @interaction-state="operationsAgentLocked = $event" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import ShopFloor from './components/ShopFloor.vue'
import RecoveryWorkspace from './components/RecoveryWorkspace.vue'
import InputStage from './components/InputStage.vue'
import ExtractionStage from './components/ExtractionStage.vue'
import MatchingStage from './components/MatchingStage.vue'
import GenerationStage from './components/GenerationStage.vue'
import ChatWindow from './components/ChatWindow.vue'
import { useEnvStore } from './stores/environmentStore'
import { useWizardStore } from './stores/wizardStore'
import { aasService } from './services/aasService'
import { api, ApiError } from './services/api'
import type { WizardData } from './types'

// ChatWindow component instance type
interface ChatWindowInstance {
  sendAgentMessage: (message: string) => Promise<void>
}

const environmentStore = useEnvStore()
onMounted(async () => {
  try {
    const {data} = await api.get<{mes_url:string|null}>('/api/service-links')
    if (data.mes_url) environmentStore.mesUiUrl = data.mes_url
  } catch { /* Keep the configured URL if the backend is offline. */ }
})
const store = useWizardStore()
const { currentStep, wizardData, isProcessing, canProceed } = storeToRefs(store)

const steps: string[] = ['Input', 'Extraction', 'Matching', 'Generation']
const activeWorkspace = ref<'recovery' | 'floor'>('recovery')
const recoveryWorkspaceRef = ref<any>(null)
const chatWindowRef = ref<ChatWindowInstance | null>(null)
const operationsAgentLocked = ref(false)

    // Helper function for extraction filenames (handles all types)
    const getExtractionFilename = (): string | null => {
      const type = wizardData.value.dataSourceType
      const files = wizardData.value.uploadedFiles || []
      const value = wizardData.value.dataSourceValue

      if (type === 'pdf') {
          if (!files || files.length === 0) return null
          
          if (files.length === 1) {
            return files[0].name
          } else {
            // For multiple PDFs, use same hash-based logic as backend
            const firstBase = files[0].name.replace('.pdf', '')
            const sortedNames = files.map(f => f.name).sort()
            const hashInput = sortedNames.join(',')
        
            // Use simple consistent hash
            let hash = 0
            for (let i = 0; i < hashInput.length; i++) {
              hash = ((hash << 5) - hash) + hashInput.charCodeAt(i)
              hash |= 0 
            }
            const fileHash = (Math.abs(hash) >>> 0).toString(16).padStart(8, '0').substring(0, 8)
            return `${firstBase}_combined_${fileHash}.pdf`
          }
      } else if (type === 'structured') {
          if (!files || files.length === 0) return null
          return files[0].name.replace(/\.(csv|json)$/i, '')
      } else if (type === 'text') {
           if (!value) return null
           // For text, create a hash of the content
           let hash = 0
           for (let i = 0; i < value.length; i++) {
               hash = ((hash << 5) - hash) + value.charCodeAt(i)
               hash |= 0
           }
           const textHash = (Math.abs(hash) >>> 0).toString(16).padStart(8, '0')
           // Text extractor saves as .json, but filename logic in nextStep uses .md as base sometimes?
           // In nextStep: `extractionFilename = ... .md`
           // But prompt says `replace('.md', '.json')`
           // Let's stick to the base filename expected by checkExtraction. 
           // `checkExtraction` looks for the file. 
           // The tool saves as .json. `getExtractedEntities` reads .json.
           // `checkExtraction` likely checks .json if validation passes?
           // Let's use .json directly because text_extractor saves as .json
           return `text_input_${textHash}.json`
      } else if (type === 'url') {
           if (!value) return null
           let hash = 0
           for (let i = 0; i < value.length; i++) {
               hash = ((hash << 5) - hash) + value.charCodeAt(i)
               hash |= 0
           }
           const urlHash = (Math.abs(hash) >>> 0).toString(16).padStart(8, '0')
           const cleanUrl = value.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30)
           return `url_${cleanUrl}_${urlHash}.json` // Web extractor also produces JSON
      }
      
      return null
    }

const nextStep = async (): Promise<void> => {
  if (currentStep.value < steps.length - 1) {
    // Special handling for step 0 -> 1 (Input -> Extraction)
    if (currentStep.value === 0) {
      store.setProcessing(true)

      try {
        const uploadedFiles = wizardData.value.uploadedFiles || []
        const targetAAS = wizardData.value.targetAAS || ''
        const endpoint = wizardData.value.aasServerEndpoint || environmentStore.aasServerUrl
        const dataSourceType = wizardData.value.dataSourceType
        const dataSourceValue = wizardData.value.dataSourceValue

        // Validation based on type
        if ((dataSourceType === 'pdf' || dataSourceType === 'structured') && uploadedFiles.length === 0) {
           alert('Please upload at least one file first')
          store.setProcessing(false)
          return
        }

        if ((dataSourceType === 'url' || dataSourceType === 'text') && !dataSourceValue) {
           const msg = dataSourceType === 'url' ? 'Please enter a valid URL' : 'Please enter text content'
           alert(msg)
           store.setProcessing(false)
           return
        }

        // Get extraction filename/identifier
        let extractionFilename = getExtractionFilename()

        if (!extractionFilename) {
          alert('Unable to determine extraction identifier')
          store.setProcessing(false)
          return
        }

        // Check if extraction JSON already exists
        const extractionResult = await aasService.checkExtraction(extractionFilename)

        // Check if AAS already exists
        const aasResult = await aasService.checkAAS(targetAAS, endpoint)

        console.log('Extraction check:', extractionResult)
        console.log('AAS check:', aasResult)

        // Fetch preview for structured files
        let structuredPreview: any = null
        if (dataSourceType === 'structured') {
            try {
                structuredPreview = await aasService.previewStructuredFile(uploadedFiles[0]?.name || '')
            } catch (e) {
                console.error("Preview failed", e)
            }
        }

        // Mode Check
        const mode = wizardData.value.mode // 'create' or 'edit'
        
        // If in 'edit' mode, trigger AAS parsing to ensure CSV exists
        if (mode === 'edit' && aasResult.exists) {
            try {
                console.log('Triggering AAS parsing for', targetAAS)
                await aasService.parseAAS(targetAAS, endpoint)
            } catch (e) {
                console.error("Failed to parse AAS:", e)
                // Warn but continue, maybe it's already there?
                console.warn("Proceeding despite parse failure.")
            }
        }

        // Logic branching:
        // Case 1: Extraction & AAS both exist -> Go straight to Matching
        if (extractionResult.exists && extractionResult.filename && aasResult.exists) {
            // Load extracted entities from JSON (don't gate on `success` flag)
            const entitiesData = await aasService.getExtractedEntities(extractionResult.filename)
            if (entitiesData.entities && entitiesData.entities.length > 0) {
                store.updateData('extractedEntities', entitiesData.entities)
                store.updateData('extractionConfigurations', entitiesData.configurations || [])
                store.updateData('extractionWarnings', entitiesData.warnings || [])
                store.updateData('isExtracting', false)
                store.nextStep()
                store.setProcessing(false)
                return
            }
        }

        // Case 2: Action needed (Extraction or AAS creation)
        store.updateData('isExtracting', true)
        store.nextStep()

        // Build the prompt for the agent
        let agentPrompt = ''
        
        // Define tasks based on mode and missing data
        const needExtraction = !extractionResult.exists
        const needAAS = !aasResult.exists && mode === 'create' 
        
        // If in edit mode and AAS missing, we technically can't "edit", but maybe user entered ID wrong?
        // But for now, let's assume if it's missing in edit mode, we just don't request creation.
        
        // Where the extraction has to land. `check-extraction` owns that
        // naming convention and returns it, rather than the rule being
        // reimplemented here and drifting from the server's copy.
        const entitiesOut = extractionResult.expectedPath || ''

        let instructionPart1 = ''

        if (dataSourceType === 'pdf') {
             const fileList = uploadedFiles.map(f => f.name).join(', ')
             const pdfArgs = (dataSourceValue || '').split(',')
                 .filter(Boolean).map(p => `"${p.trim()}"`).join(' ')
             instructionPart1 = `extract the technical data from the uploaded datasheet(s) ${fileList}:

    a4s extract --pdf ${pdfArgs} --out "${entitiesOut}"

Use exactly that --out path; the wizard reads the entities back from it.`
        } else if (dataSourceType === 'structured') {
             const fileName = uploadedFiles[0]?.name || ''
             const tablePath = (dataSourceValue || '').split(',')[0].trim()
             let context = ''
             if (structuredPreview) {
                 context = `\n\nHeaders: ${structuredPreview.headers.join(', ')}\nFirst rows:\n${JSON.stringify(structuredPreview.rows, null, 2)}\n\nIf the columns are unambiguous, add --map entity=<col>,value=<col>,unit=<col>; otherwise let the model read the headers itself.`
             }
             instructionPart1 = `extract the technical data from the uploaded table ${fileName}:

    a4s extract --table "${tablePath}" --out "${entitiesOut}"

Use exactly that --out path; the wizard reads the entities back from it.${context}`
        } else if (dataSourceType === 'text') {
             instructionPart1 = `extract the technical data from the text below.

a4s extract reads text from a file, not from an argument, so write the text out with write_file first, then:

    a4s extract --text <that file> --out "${entitiesOut}"

Use exactly that --out path; the wizard reads the entities back from it.

The text:
"""${dataSourceValue}"""`
        } else { // url
             const url = dataSourceValue
             instructionPart1 = `extract the technical data from ${url}:

    a4s extract --url "${url}" --out "${entitiesOut}"

Use exactly that --out path; the wizard reads the entities back from it.`
        }

        const instructionPart2 = `create an Asset Administration Shell with the idShort "${targetAAS}", instantiating the submodel templates named in the wizard context below and attaching each one to it:

    aas shell create <shell-id> --id-short ${targetAAS} --global-asset-id <asset-id>
    aas smt instantiate <template-id> --as <submodel-id> --id-short <name> --attach-to <shell-id>

Instantiate rather than creating submodels by hand: the template carries the semanticIds that make each slot mean something, and reproducing them by hand is where conformance is lost.`

        if (needExtraction && needAAS) {
            agentPrompt = `Please do the following two tasks only:
1. ${instructionPart1}
2. ${instructionPart2}

Do NOT perform matching or populate steps. Stop after completing these two tasks.`
        } else if (needExtraction) {
             agentPrompt = `Please ${instructionPart1}.`
        } else if (needAAS) {
             agentPrompt = `Please ${instructionPart2}.`
        } else {
             // Corner case: In edit mode, AAS exists. Extraction exists. But maybe entities failed to load?
             // Or explicit re-run requested?
             // If we are here, it means we didn't return in Case 1.
             // So just trigger extraction again to be safe/fresh?
             agentPrompt = `Please ${instructionPart1}.`
        }

        // Send message to agent via ChatWindow
        if (chatWindowRef.value && chatWindowRef.value.sendAgentMessage) {
          chatWindowRef.value.sendAgentMessage(agentPrompt)
        }

      } catch (error) {
        console.error('Error checking existing data:', error)
        // Fallback: proceed with agent extraction
        store.updateData('isExtracting', true)
        store.nextStep()
        // ... fallback prompt logic if needed (simplified here) ...
      } finally {
        store.setProcessing(false)
      }

      return
    }

    // Special handling for step 1 -> 2 (Extraction -> Matching)
    // Save edited entities and trigger matching
    if (currentStep.value === 1) {
      const uploadedFiles = wizardData.value.uploadedFiles || []
      const targetAAS = wizardData.value.targetAAS || ''
      const endpoint = wizardData.value.aasServerEndpoint || environmentStore.aasServerUrl

      // Get extraction filename using helper function
      const extractionFilename = getExtractionFilename()

      if (extractionFilename && wizardData.value.extractedEntities.length > 0) {
        try {
          // Save edited entities first
          const result = await aasService.saveEditedEntities({
            original_filename: extractionFilename,
            entities: wizardData.value.extractedEntities,
            // Carried so the saved file keeps saying what the document was,
            // and records which configuration this asset was declared to be.
            configuration: wizardData.value.chosenConfiguration || undefined,
            configurations: wizardData.value.extractionConfigurations,
            warnings: wizardData.value.extractionWarnings
          })

          if (result.success) {
            store.updateData('editedEntitiesFile', result.filename)
            console.log('Saved edited entities to:', result.filename)

            // Check if matching results already exist
            const matchingResult = await aasService.checkMatching(extractionFilename, targetAAS)

            if (matchingResult.exists && matchingResult.filename) {
              // Load existing matching results
              const resultsData = await aasService.getMatchingResults(matchingResult.filename, targetAAS)

              if (resultsData.success && resultsData.results.length > 0) {
                store.updateData('matchedProperties', resultsData.results)
                store.updateData('isMatching', false)
                store.nextStep()
                return
              }
            }

            // Run the matching. A stage button is a deterministic
            // commitment, so it calls a function rather than asking the agent
            // to run one. The previous version sent a prompt naming the exact
            // commands and then polled for a file at an exact path -- a
            // deterministic script executed by a non-deterministic runner --
            // and it failed the way that fails: the agent matched all four
            // submodels correctly and never wrote the manifest being polled
            // for, so a finished job showed "no matches found yet" forever.
            //
            // The agent stays on the socket for the open-ended half: why a row
            // did not match, or moving one elsewhere. It answers from the plan
            // file, which carries every score, band and rejected candidate.
            store.updateData('isMatching', true)
            store.nextStep()

            try {
              const matched = await aasService.runMatch(
                extractionFilename, targetAAS, endpoint
              )
              store.updateData('matchedProperties', matched.results || [])
            } catch (error) {
              // A timeout is not a failure. Aborting the fetch stops this
              // browser waiting; the server never hears about it and goes on
              // matching, then writes the plan. Reporting "Matching failed"
              // for a run that is working normally is the wrong answer, and
              // the plan file is how to get the right one.
              const timedOut = error instanceof ApiError && error.status === 408
              if (timedOut) {
                if (chatWindowRef.value && chatWindowRef.value.showInfo) {
                  chatWindowRef.value.showInfo(
                    'Matching is taking longer than expected. It is still running on ' +
                    'the server — waiting for the plan.'
                  )
                }
                const recovered = await aasService.waitForMatch(extractionFilename, targetAAS)
                if (recovered) {
                  store.updateData('matchedProperties', recovered)
                  return  // the finally below clears isMatching
                }
              }
              console.error('Matching failed:', error)
              store.updateData('matchedProperties', [])
              // Surfaced in the pane rather than left as a spinner: an empty
              // table with no explanation is what the polling version showed,
              // and it is indistinguishable from still working.
              if (chatWindowRef.value && chatWindowRef.value.showError) {
                chatWindowRef.value.showError(
                  timedOut
                    ? 'Matching did not finish within the time the wizard waits. ' +
                      'It may still be running; reopen this step to pick up the plan.'
                    : `Matching failed: ${error instanceof Error ? error.message : error}`
                )
              }
            } finally {
              store.updateData('isMatching', false)
            }
            return
          } else {
            console.error('Failed to save edited entities:', result)
          }
        } catch (error) {
          console.error('Error saving edited entities:', error)
        }
      }
    }

    // Special handling for step 2 -> 3 (Matching -> Generation)
    // Save matching results and trigger population
    if (currentStep.value === 2) {
      store.setProcessing(true)
      const uploadedFiles = wizardData.value.uploadedFiles || []
      const targetAAS = wizardData.value.targetAAS || ''
      const endpoint = wizardData.value.aasServerEndpoint || environmentStore.aasServerUrl

      // Get extraction filename using helper function
      const extractionFilename = getExtractionFilename()

      if (extractionFilename && wizardData.value.matchedProperties.length > 0) {
        try {
          // Construct proper matching filename with AAS ID to avoid overwriting
          // Format: {pdf_name}_{aas_id}_matching_result.json
          let matchBase = extractionFilename
          if (matchBase.toLowerCase().endsWith('.pdf')) {
            matchBase = matchBase.slice(0, -4)
          }
          const finalFilename = targetAAS ? `${matchBase}_${targetAAS}` : matchBase

          // Filter out invalid matches (e.g. Submodels or cleared selections) before saving
          // matchedProperties contains all original extractions. We want those with a valid assignment.
          // User confirmed Collections ARE valid selections. Only Submodels/AAS roots are not.
          const validResults = wizardData.value.matchedProperties.filter((p: any) => 
              p.selected !== false && 
              p.aasType && 
              !['AssetAdministrationShell', 'Submodel'].includes(p.aasType)
          )

          if (validResults.length === 0) {
              alert('No valid properties matched. Please ensure you have selected actual Properties (not Collections) for at least one item.')
              store.setProcessing(false)
              return
          }

          // Save matching results first
          const saveResult = await aasService.saveMatchingResults({
            filename: finalFilename,
            results: validResults
          })

          if (saveResult.success) {
            console.log('Saved matching results to:', saveResult.path)

            // The save endpoint returns the full path to the _edit file
            const matchResultPath = saveResult.path

            // Set generating state and move to step 3
            store.updateData('isGenerating', true)
            store.updateData('generationComplete', false)
            store.nextStep()

            // Apply the reviewed plan
            if (chatWindowRef.value && chatWindowRef.value.sendAgentMessage) {
              const agentPrompt = `apply the reviewed match plan to the AAS "${targetAAS}" on ${endpoint}.

The plan the user confirmed is at:
    ${matchResultPath}

That file is a manifest listing one plan per submodel. Read it, then for each entry run:

    aas populate --plan <that entry's path> --skip-warnings

Report how many values were written and anything that failed. Do not re-match: the targets in the plan are the ones the user chose, including any element that has to be created.`
              chatWindowRef.value.sendAgentMessage(agentPrompt)
            }

            store.setProcessing(false)
            return
          } else {
            console.error('Failed to save matching results:', saveResult)
            alert('Failed to save matching results. Please try again.')
          }
        } catch (error) {
          console.error('Error saving matching results:', error)
          const errorMessage = error instanceof Error ? error.message : 'Unknown error'
          alert('Error saving matching results: ' + errorMessage)
        }
      } else {
        alert('No matching results to populate. Please ensure entities are matched.')
      }

      store.setProcessing(false)
      return
    }

    // Normal step progression for other steps
    store.nextStep()

    // Trigger auto-actions when entering certain stages
    if (currentStep.value === 1 && wizardData.value.extractedEntities.length === 0) {
      simulateExtraction()
    }
    if (currentStep.value === 3) {
      simulateGeneration()
    }
  }
}

// Method to load extracted entities after agent completes.
// `forceStop` is true only on a real run completion; on a mere agent question
// (awaiting_input) we keep the spinner running if nothing has been produced yet.
const loadExtractedEntities = async (forceStop: boolean): Promise<void> => {
  // Get extraction filename using helper function (handles all types)
  const extractionFilename = getExtractionFilename()

  if (!extractionFilename) {
    console.error('Unable to determine extraction filename')
    if (forceStop) store.updateData('isExtracting', false)
    return
  }

  let loaded = false
  try {
    const extractionResult = await aasService.checkExtraction(extractionFilename)

    if (extractionResult.exists && extractionResult.filename) {
      const entitiesData = await aasService.getExtractedEntities(extractionResult.filename)

      // Accept the data whenever entities are present. Don't gate on `success`:
      // depending on which backend route answers, that flag may be absent.
      if (entitiesData.entities && entitiesData.entities.length > 0) {
        store.updateData('extractedEntities', entitiesData.entities)
        store.updateData('extractionConfigurations', entitiesData.configurations || [])
        store.updateData('extractionWarnings', entitiesData.warnings || [])
        loaded = true
      }
    }
  } catch (error) {
    console.error('Error loading extracted entities:', error)
    if (error instanceof ApiError) {
      console.error(`API Error: ${error.status} - ${error.message}`)
    }
  } finally {
    // Stop spinning only when the run truly finished, or when entities actually
    // arrived. A premature ask_user with no results must not blank the table.
    if (forceStop || loaded) store.updateData('isExtracting', false)
  }
}

// Method to load matching results after agent completes
const loadMatchingResults = async (forceStop: boolean): Promise<void> => {
  // Get extraction filename using helper function
  const extractionFilename = getExtractionFilename()

  if (!extractionFilename) {
    console.error('Unable to determine extraction filename')
    if (forceStop) store.updateData('isMatching', false)
    return
  }

  const targetAAS = wizardData.value.targetAAS || ''

  let loaded = false
  try {
    const matchingResult = await aasService.checkMatching(extractionFilename, targetAAS)

    if (matchingResult.exists && matchingResult.filename) {
      const resultsData = await aasService.getMatchingResults(matchingResult.filename, targetAAS)

      if (resultsData.success && resultsData.results.length > 0) {
        store.updateData('matchedProperties', resultsData.results)
        console.log('Loaded matching results:', resultsData.results.length, 'matches')
        loaded = true
      }
    }
  } catch (error) {
    console.error('Error loading matching results:', error)
    if (error instanceof ApiError) {
      console.error(`API Error: ${error.status} - ${error.message}`)
    }
  } finally {
    if (forceStop || loaded) store.updateData('isMatching', false)
  }
}

// Handler for when agent completes - load extracted entities or matching results.
// `payload.awaitingInput` true => agent only paused for a question, not done.
const onAgentComplete = async (payload?: { awaitingInput: boolean }): Promise<void> => {
  if (recoveryWorkspaceRef.value) void recoveryWorkspaceRef.value.refresh()
  const forceStop = !payload?.awaitingInput

  // Load extracted entities if on extraction stage
  if (currentStep.value === 1 && wizardData.value.isExtracting) {
    console.log('Loading extracted entities...')
    await loadExtractedEntities(forceStop)
  }

  // Load matching results if on matching stage
  if (currentStep.value === 2 && wizardData.value.isMatching) {
    console.log('Loading matching results...')
    await loadMatchingResults(forceStop)
  }

  // Handle completion of population on generation stage
  if (currentStep.value === 3 && wizardData.value.isGenerating) {
    console.log('Population completed')
    store.updateData('isGenerating', false)
    store.updateData('generationComplete', true)
  }
}

// Mock extraction simulation
const simulateExtraction = (): void => {
  store.updateData('isExtracting', true)

  setTimeout(() => {
    store.updateData('extractedEntities', [
      { entity: 'Product Name', value: 'Industrial Motor XM-2000', description: 'Main product identifier' },
      { entity: 'Manufacturer', value: 'TechMotors GmbH', description: 'Product manufacturer' },
      { entity: 'Rated Power', value: '2.5 kW', description: 'Maximum continuous power output' },
      { entity: 'Voltage', value: '380-400V', description: 'Operating voltage range' },
      { entity: 'Frequency', value: '50/60 Hz', description: 'Operating frequency' },
      { entity: 'Speed', value: '1450 rpm', description: 'Rotational speed at rated load' },
      { entity: 'Efficiency Class', value: 'IE3', description: 'Energy efficiency classification' },
      { entity: 'Protection Class', value: 'IP55', description: 'Ingress protection rating' },
      { entity: 'Weight', value: '25 kg', description: 'Total weight of the motor' },
      { entity: 'Ambient Temperature', value: '-20°C to +40°C', description: 'Operating temperature range' }
    ])
    store.updateData('isExtracting', false)
  }, 2000)
}

// Mock matching simulation - Keeping for reference but unused
const simulateMatching = (): void => {
    // ... code omitted for brevity ...
}

// Generation is now handled by the step 2 -> 3 transition with real agent call
const simulateGeneration = (): void => {
  // This is now a no-op since population is triggered in nextStep for step 2 -> 3
  // The isGenerating state is managed there and completed via onAgentComplete
  console.log('simulateGeneration called - population handled by agent')
}

// Resizable panel logic
const leftPanelWidth = ref(window.innerWidth * 0.74)
const splitContainer = ref<HTMLElement | null>(null)
let resizeObserver: ResizeObserver | undefined
const constrainWidth = (width: number) => {
  const available = splitContainer.value?.clientWidth || window.innerWidth
  const maximum = Math.max(0, available - 246)
  return Math.min(maximum, Math.max(Math.min(400, maximum), width))
}
onMounted(() => {
  resizeObserver = new ResizeObserver(() => { leftPanelWidth.value = constrainWidth(leftPanelWidth.value) })
  if (splitContainer.value) resizeObserver.observe(splitContainer.value)
})
onUnmounted(() => resizeObserver?.disconnect())

const startResize = (e: PointerEvent): void => {
  e.preventDefault()
  const handle = e.currentTarget as HTMLElement
  handle.setPointerCapture(e.pointerId)
  const move = (event: PointerEvent) => {
    const left = splitContainer.value?.getBoundingClientRect().left || 0
    leftPanelWidth.value = constrainWidth(event.clientX - left)
  }
  const end = () => {
    handle.removeEventListener('pointermove', move)
    handle.removeEventListener('lostpointercapture', end)
    handle.removeEventListener('pointerup', end)
    handle.removeEventListener('pointercancel', end)
    if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId)
  }
  handle.addEventListener('pointermove', move)
  handle.addEventListener('pointerup', end)
  handle.addEventListener('pointercancel', end)
  handle.addEventListener('lostpointercapture', end)
}
</script>

<style scoped>
.industrial-mode { width:100%; min-width:0; overflow:hidden; }
.workspace-navigation { display: flex; gap: 12px; align-items: center; padding: 10px 20px; border-bottom: 1px solid #dce4ed; background: white; min-height: 48px; box-sizing: border-box; }
.workspace-navigation strong { letter-spacing: .12em; color: #174f79; }
.workspace-navigation button { border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; background: #f0f4f8; color: #34516d; }
.workspace-navigation button.selected { background: #174f79; color: white; }
.workspace-navigation span { font-size: 12px; color: #6c7e91; margin-left: auto; }
.main-container { height: calc(100vh - 68px); }
.industrial-mode .workspace-navigation { height:44px; min-height:44px; padding:6px 14px; background:#0f131a; border-color:#283240; gap:14px; }
.industrial-mode .workspace-navigation strong { color:#e4eaf2; font-size:13px; }
.industrial-mode .workspace-navigation strong small { font-size:9px; color:#8794a7; font-weight:400; }
.industrial-mode .workspace-navigation button { background:transparent; border-radius:2px; color:#a2afbf; font-size:11px; padding:6px 10px; }
.industrial-mode .workspace-navigation button.selected { color:#38bdf8; background:#182733; }
.industrial-mode .workspace-navigation span { color:#8794a7; font:10px Consolas,monospace; }
.industrial-mode .main-container { height:calc(100vh - 44px); background:#0a0c11; flex-direction:row; }
.industrial-mode :deep(.recovery-workspace) { min-width:0; overflow-x:auto; }
.industrial-mode :deep(.workspace-body) { min-width:490px; }
.resizer { touch-action:none; user-select:none; }
.industrial-mode :deep(.resizer) { background:#283240; }
.industrial-mode :deep(.chat-container) { background:#0f131a; border-color:#283240; border-radius:0; box-shadow:none; min-width:240px; width:0; flex:1 1 0; height:100%; }
.industrial-mode :deep(.chat-messages) { background:#0f131a; padding:12px; }
.industrial-mode :deep(.chat-messages)::before { content:'AGENT / OPERATIONS'; display:block; font:10px Consolas,monospace; letter-spacing:1px; color:#8898ac; border-bottom:1px solid #283240; padding:0 0 12px; margin-bottom:16px; }
.industrial-mode :deep(.assistant-content), .industrial-mode :deep(.thinking-content), .industrial-mode :deep(.ask-user-card) { background:#151f2b; border-color:#2c3f51; color:#d1dce9; border-radius:4px; font-size:12px; }
.industrial-mode :deep(.user-content) { background:#15405b; color:#deedf5; border-radius:4px; }
.industrial-mode :deep(.message-avatar) { background:#23303f; font-size:16px; width:26px; height:26px; }
.industrial-mode :deep(.message-time) { color:#8794a7; }
.industrial-mode :deep(.chat-input-area) { background:#0f131a; border-color:#283240; }
.industrial-mode :deep(.chat-input-wrapper), .industrial-mode :deep(.chat-input-wrapper:focus-within) { background:#17202c; border-color:#34445a; border-radius:4px; }
.industrial-mode :deep(.chat-input) { color:#e4eaf2; }
.industrial-mode :deep(.reset-chat-btn-input) { background:#17202c; color:#aab8c8; border-color:#34445a; }
.industrial-mode :deep(.error-content), .industrial-mode :deep(.status-message.error) { background:#351e25; border-color:#713745; color:#ffb2b2; border-radius:3px; }
.industrial-mode :deep(.tool-call-card), .industrial-mode :deep(.tool-code) { background:#202532; border-color:#444363; color:#cfd5eb; }
.industrial-mode :deep(.ask-user-btn) { background:#182b3b; border-color:#355a74; color:#e1eaf4; }
.industrial-mode :deep(.answered-state) { background:#15382e; color:#8ee1bd; }
@media (max-width: 900px) { .workspace-navigation span { display: none; } }
.industrial-mode .workspace-navigation button {font-size:15px}
.industrial-mode .workspace-navigation strong {font-size:16px}
.industrial-mode :deep(.assistant-content), .industrial-mode :deep(.thinking-content), .industrial-mode :deep(.ask-user-card), .industrial-mode :deep(.user-content), .industrial-mode :deep(.chat-input) {font-size:16px;line-height:1.6}
</style>
