<template>
  <div class="chat-container">
    <!-- Header removed as requested -->

    <div class="chat-messages" ref="messagesContainer">
      <div v-if="messages.length === 0" class="chat-empty-state">
        <div class="icon">💬</div>
        <h4>Start a Conversation</h4>
        
      </div>

      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['chat-message', message.type]"
      >
        <!-- User Message -->
        <template v-if="message.type === 'user'">
          <div class="message-avatar user">{{ message.automated ? '⚡' : '👤' }}</div>
          <div style="flex: 1">
            <div class="message-content user-content">
              {{ message.content }}
              <details v-if="message.details"><summary>Request details</summary><pre class="request-details">{{ message.details }}</pre></details>
            </div>
            <div class="message-time">{{ message.time }}</div>
          </div>
        </template>

        <!-- Agent Thinking -->
        <template v-if="message.type === 'thinking'">
          <div class="message-avatar assistant">🤖</div>
          <div style="flex: 1">
            <div class="message-content thinking-content">
              <div style="display: flex; align-items: center; gap: 8px">
                <div class="spinner-small"></div>
                <span>{{ message.content }}</span>
              </div>
            </div>
          </div>
        </template>

        <!-- Assistant Message -->
        <template v-if="message.type === 'assistant'">
          <div class="message-avatar assistant">🤖</div>
          <div style="flex: 1">
            <div class="message-content assistant-content" v-html="formatMessage(message.content)"></div>
            <div class="message-time">{{ message.time }}</div>
          </div>
        </template>

        <!-- Tool Call / Ask User -->
        <template v-if="message.type === 'tool_call'">
          <!-- Interactive AskUser Widget -->
          <template v-if="message.tool_name === 'ask_user'">
            <div class="message-avatar assistant ask-user">🧑‍💼</div>
            <div style="flex: 1">
              <div class="message-content assistant-content ask-user-card">
                <div class="ask-user-header">
                  <strong>Action Required: User Feedback</strong>
                </div>
                
                <div class="ask-user-prompt" v-if="message.tool_args && message.tool_args.question" style="margin: 12px 0; color: #374151; font-size: 14px; white-space: pre-wrap;" v-html="formatMessage(message.tool_args.question)">
                </div>

                <!-- Display standard prompt from the tool args if any -->
                <div class="ask-user-options" v-if="message.tool_args && message.tool_args.options && message.tool_args.options.length">
                  <template v-if="message.answeredWith">
                    <div class="answered-state">
                      <span class="check-icon">✓</span> Answered: <strong>{{ message.answeredWith }}</strong>
                    </div>
                  </template>
                  <template v-else>
                    <button 
                      v-for="(opt, idx) in message.tool_args.options" 
                      :key="idx"
                      @click="handleAskUserOption(opt, index)"
                      class="ask-user-btn"
                    >
                      {{ opt }}
                    </button>
                  </template>
                </div>
                
                <div v-if="message.tool_args && message.tool_args.allow_open_text && !message.answeredWith" class="ask-user-hint">
                  <small>Or type your custom response in the chat input below.</small>
                </div>
              </div>
              <div class="message-time">{{ message.time }}</div>
            </div>
          </template>

          <!-- Standard Tool Call (Collapsible) -->
          <template v-else>
            <div class="message-avatar assistant">🔧</div>
            <div style="flex: 1">
              <div
                class="tool-call-card"
                :class="{ expanded: message.expanded }"
                @click="toggleExpand(index)"
              >
                <div class="tool-call-header">
                  <span class="tool-icon">🔧</span>
                  <strong>{{ message.tool_name || 'Processing Step' }}</strong>
                  <span class="tool-status" v-if="message.tool_status === 'completed'">✓ Completed</span>
                  <span class="tool-status running" v-else>⏳ Running</span>
                  <span class="expand-icon">{{ message.expanded ? '▼' : '▶' }}</span>
                </div>

                <div v-if="message.expanded" class="tool-call-details">
                  <div class="tool-section">
                    <div class="tool-section-title">Arguments:</div>
                    <pre class="tool-code">{{ formatJSON(message.tool_args) }}</pre>
                  </div>

                  <div v-if="message.tool_result" class="tool-section">
                    <div class="tool-section-title">Result:</div>
                    <pre class="tool-code">{{ message.tool_result }}</pre>
                  </div>
                </div>
              </div>
              <div class="message-time">{{ message.time }}</div>
            </div>
          </template>
        </template>

        <!-- Error Message -->
        <template v-if="message.type === 'error'">
          <div class="message-avatar error">⚠️</div>
          <div style="flex: 1">
            <div class="message-content error-content">
              <strong>Error:</strong> {{ message.content }}
            </div>
            <div class="message-time">{{ message.time }}</div>
          </div>
        </template>
      </div>

      <!-- Connection Status -->
      <div v-if="connectionStatus !== 'connected'" class="connection-status">
        <div v-if="connectionStatus === 'connecting'" class="status-message">
          <div class="spinner-small"></div>
          <span>Connecting to agent...</span>
        </div>
        <div v-if="connectionStatus === 'disconnected'" class="status-message error">
          <span>⚠️ Disconnected from agent</span>
          <button @click="reconnect" class="btn-reconnect">Reconnect</button>
        </div>
      </div>
    </div>

    <div class="chat-input-area">
      <div class="chat-input-wrapper">
        <button class="reset-chat-btn-input" @click="resetChat" title="Reset Chat Context">
          ↺
        </button>
        <textarea
          v-model="inputMessage"
          @keydown.enter.prevent="handleSend"
          class="chat-input"
          placeholder="Type your message..."
          rows="1"
          :disabled="!isConnected || isSending"
        ></textarea>
        <button
          type="button"
          @click="isSending ? handleStop() : handleSend()"
          class="chat-send-btn"
          :class="{ 'stop-mode': isSending }"
          :aria-label="isSending ? (isStopping ? 'Stopping agent' : 'Stop agent') : 'Send message'"
          :title="isSending ? (isStopping ? 'Stopping agent…' : 'Stop agent') : 'Send message'"
          :disabled="!isConnected || (isSending ? isStopping : !inputMessage.trim())"
        >
          <span v-if="isSending" class="stop-square" aria-hidden="true"></span>
          <span v-else aria-hidden="true">➤</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, onMounted, onUnmounted, type Ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useEnvStore } from '@/stores/environmentStore'
import { useWizardStore } from '@/stores/wizardStore'
import { useSharedWebSocket, buildChatMessage } from '@/services/websocketService'
import type { WizardData } from '@/types'
import { automatedRequest, boundaryFeedback } from '../services/agentPresentation.mjs'

// Message type definitions
interface UserMessage {
  type: 'user'
  content: string
  time: string
  details?: string
  automated?: boolean
}

interface ThinkingMessage {
  type: 'thinking'
  content: string
  time: string
}

interface AssistantMessage {
  type: 'assistant'
  content: string
  time: string
}

interface ToolCallMessage {
  type: 'tool_call'
  tool_call_id?: string
  tool_name: string
  tool_args: any
  tool_result: string | null
  tool_status: string
  expanded: boolean
  time: string
  answeredWith?: string
}

interface ErrorMessage {
  type: 'error'
  content: string
  time: string
}

type ChatMessage = UserMessage | ThinkingMessage | AssistantMessage | ToolCallMessage | ErrorMessage

// WebSocket message types
interface FileContext {
  name: string
  path: string
}

interface WizardContext {
  selectedTemplates: string[]
  aasServerEndpoint: string
  targetAAS: string | null
  dataSourceType: string | null
  editedEntitiesFile: string | null
}

interface ShopFloorContext {
  assets: any[]
  connections: any[]
}

interface WebSocketMessage {
  type: string
  content?: string
  tool_call_id?: string
  tool_name?: string
  tool_args?: any
  tool_result?: string
  tool_status?: string
  result?: string
}

// Emits
const emit = defineEmits<{
  // `awaitingInput` distinguishes a real run completion from the agent merely
  // pausing for a question (ask_user). The latter must not tear down an
  // in-flight extraction/matching wait.
  agentComplete: [payload?: { awaitingInput: boolean }]
  interactionState: [locked: boolean]
}>()

// State
const messages: Ref<ChatMessage[]> = ref([])
const inputMessage = ref('')
const isSending = ref(false)
const isStopping = ref(false)
const awaitingAnswer = ref(false)
watch(() => isSending.value || awaitingAnswer.value, locked => emit('interactionState', locked), {immediate:true})
const messagesContainer = ref<HTMLDivElement | null>(null)
const environmentStore = useEnvStore()
const wizardStore = useWizardStore()
const { wizardData, currentStep } = storeToRefs(wizardStore)

// Use shared WebSocket service
const { connectionStatus, isConnected, connect, disconnect, send, onMessage } = useSharedWebSocket()

// Helper functions
const formatTime = (): string => {
  const now = new Date()
  return now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

const formatJSON = (obj: any): string => {
  return JSON.stringify(obj, null, 2)
}

const formatMessage = (content: string): string => {
  if (!content) return ''
  // Basic markdown formatting and handle <thinking> tags to make them visible
  let formatted = content
    .replace(/<thinking>([\s\S]*?)<\/thinking>/g, (match, p1) => {
      return `<div style="background: #f3f4f6; color: #6b7280; font-style: italic; padding: 8px; border-radius: 6px; margin-bottom: 8px; font-size: 13px; border-left: 3px solid #179C7D;">🤔 <strong>Agent Thought Process:</strong><br/>${p1.replace(/\n/g, '<br>')}</div>`
    })
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
    .replace(/\n/g, '<br>')  // Line breaks
  return formatted
}

const toggleExpand = (index: number): void => {
  const message = messages.value[index]
  if (message.type === 'tool_call') {
    message.expanded = !message.expanded
  }
}

const scrollToBottom = async (): Promise<void> => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// Message handler for WebSocket messages
const handleWebSocketMessage = async (data: WebSocketMessage): Promise<void> => {
  console.log('Received:', data)

  const feedback = boundaryFeedback(data, messages.value)
  if (feedback) {
    clearThinking()
    messages.value.push({ type: 'assistant', content: feedback, time: formatTime() })
  }

  switch (data.type) {
    case 'user_message':
      // User message already added, skip
      break

    case 'thinking':
      // Only show thinking if there's meaningful content
      if (data.content && data.content.trim()) {
        // Clear first: one spinner at the tail, never a trail of them.
        clearThinking()
        messages.value.push({
          type: 'thinking',
          content: data.content,
          time: formatTime()
        })
        await scrollToBottom()
      }
      break

    case 'assistant_message':
      clearThinking()

      // Only add non-empty messages
      if (data.content && data.content.trim()) {
        messages.value.push({
          type: 'assistant',
          content: data.content,
          time: formatTime()
        })
        await scrollToBottom()
      }
      break

    case 'tool_call':
      // Check if this tool call already exists (by ID)
      const existingToolIndex = messages.value.findIndex(m => 
        m.type === 'tool_call' && 
        data.tool_call_id && 
        m.tool_call_id === data.tool_call_id
      )

      if (existingToolIndex !== -1) {
        // Update existing message
        const existingMsg = messages.value[existingToolIndex] as ToolCallMessage
        
        // If we are receiving a "completed" status, update the result and status
        if (data.tool_status === 'completed') {
            existingMsg.tool_result = data.tool_result || null
            existingMsg.tool_status = 'completed'
            // Auto-expand if completed to show result? Maybe keep collapsed/expanded as is.
            // existingMsg.expanded = true 
        } else {
            // Otherwise just update details (idempotent)
            existingMsg.tool_name = data.tool_name || existingMsg.tool_name
            existingMsg.tool_args = data.tool_args || existingMsg.tool_args
        }
      } else {
        // A tool card is a real message, so the spinner that preceded it is
        // over. Without this the spinner is buried under the card and the
        // trailing-only removal below can never reach it again.
        clearThinking()
        // New tool call - push to array
        messages.value.push({
            type: 'tool_call',
            tool_call_id: data.tool_call_id,
            tool_name: data.tool_name || '',
            tool_args: data.tool_args || {},
            tool_result: data.tool_result || null,
            tool_status: data.tool_status || 'called',
            expanded: false,
            time: formatTime()
        })
      }
      await scrollToBottom()
      break

    case 'tool_result':
      // Legacy support - find matching tool call and update result
      for (let i = messages.value.length - 1; i >= 0; i--) {
        const message = messages.value[i]
        if (message.type === 'tool_call' &&
            message.tool_name === data.tool_name &&
            !message.tool_result) {
          message.tool_result = data.result || null
          message.tool_status = 'completed'
          break
        }
      }
      await scrollToBottom()
      break

    case 'awaiting_input':
      // Agent asked a question: the run is paused, so nothing is working.
      clearThinking()
      awaitingAnswer.value = true
      isSending.value = false
      await scrollToBottom()
      // Agent paused for a question — not a real completion. Let the parent
      // opportunistically load any results, but it must keep waiting if none
      // exist yet (extraction may still be pending after the user answers).
      emit('agentComplete', { awaitingInput: true })
      break

    case 'complete':
      clearThinking()
      awaitingAnswer.value = false
      isSending.value = false
      isStopping.value = false
      await scrollToBottom()
      // Real run completion.
      emit('agentComplete', { awaitingInput: false })
      break

    case 'stopping':
      isStopping.value = true
      break

    case 'stopped':
      clearThinking()
      isSending.value = false
      isStopping.value = false
      awaitingAnswer.value = false
      messages.value.push({ type: 'assistant', content: data.content || 'Stopped. You can send a new instruction.', time: formatTime() })
      await scrollToBottom()
      emit('agentComplete', { awaitingInput: false })
      break

    case 'error':
      clearThinking()
      awaitingAnswer.value = false
      messages.value.push({
        type: 'error',
        content: data.content || 'Unknown error',
        time: formatTime()
      })
      isSending.value = false
      isStopping.value = false
      await scrollToBottom()
      break
  }
}

/**
 * Remove every "Working..." indicator, wherever it sits.
 *
 * It used to be removed only when it happened to be the *last* message:
 * `if (last.type === 'thinking') messages.pop()`. A step pushes the spinner and
 * then pushes a tool card, so from the first tool call onward the spinner is
 * buried in the middle and that check never matches it again. The result is a
 * transcript with a spinner frozen between two completed cards, still claiming
 * to be working after the run has finished or stopped to ask a question --
 * which is exactly the state the *status* is not in.
 *
 * A spinner is a transient indicator, so there is never a reason to keep more
 * than the one at the tail, and no reason to keep any once a real message has
 * arrived.
 */
const clearThinking = (): void => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    if (messages.value[i].type === 'thinking') messages.value.splice(i, 1)
  }
}

/**
 * Send whatever the machine console handed over, once there is a socket.
 *
 * Watched rather than called directly because the console lives under
 * ShopFloor and this pane is a sibling: routing through the store keeps the
 * two from needing a reference to each other, and keeps every agent turn in
 * this one session.
 */
watch(
  () => [wizardStore.wizardData.pendingAgentQuestion, isConnected.value, isSending.value, awaitingAnswer.value] as const,
  async ([question, connected, sending, awaiting]) => {
    if (!question || !connected || sending || awaiting) return
    wizardStore.updateData('pendingAgentQuestion', '')
    await sendAgentMessage(question)
  },
  { immediate: true }
)

const reconnect = (): void => {
  disconnect()
  setTimeout(() => connect(), 100)
}

const handleStop = (): void => {
  if (!isSending.value || !isConnected.value || isStopping.value) return
  isStopping.value = true
  send({ type: 'stop' })
}

const handleSend = async (): Promise<void> => {
  if (!inputMessage.value.trim() || !isConnected.value || isSending.value) return

  const userMsg = inputMessage.value.trim()

  // Add user message to UI
  messages.value.push({
    type: 'user',
    content: userMsg,
    time: formatTime()
  })

  inputMessage.value = ''
  isSending.value = true
  await scrollToBottom()

  // Prepare file context if files are uploaded
  let fileContext: FileContext | null = null
  if (wizardData.value.uploadedFiles && wizardData.value.uploadedFiles.length > 0) {
    const fileNames = wizardData.value.uploadedFiles.map(f => f.name).join(', ')
    fileContext = {
      name: fileNames,
      path: wizardData.value.dataSourceValue || wizardData.value.uploadedFiles.map(f => `./Attachments/${f.name}`).join(',')
    }
  }

  // Prepare wizard context from input stage configuration
  let wizardContext: WizardContext | null = null
  if (wizardData.value.viewMode !== 'shop-floor' && (wizardData.value.submodelTemplate || wizardData.value.targetAAS || wizardData.value.aasServerEndpoint)) {
    wizardContext = {
      selectedTemplates: Array.isArray(wizardData.value.submodelTemplate)
        ? wizardData.value.submodelTemplate
        : wizardData.value.submodelTemplate
          ? [wizardData.value.submodelTemplate]
          : [],
      aasServerEndpoint: wizardData.value.aasServerEndpoint || environmentStore.aasServerUrl,
      targetAAS: wizardData.value.targetAAS || null,
      dataSourceType: wizardData.value.dataSourceType || null,
      editedEntitiesFile: wizardData.value.editedEntitiesFile || null
    }
  }

  // Prepare Shop Floor context
  let shopFloorContext: ShopFloorContext | null = null
  if (wizardData.value.viewMode === 'shop-floor') {
      shopFloorContext = {
          assets: wizardData.value.assets || [],
          connections: wizardData.value.connections || []
      }
  }

  // Build and send message using service
  const chatMessage = buildChatMessage(userMsg, fileContext, wizardContext, shopFloorContext)
  send(chatMessage)
}

const resetChat = async (): Promise<void> => {
  if (!confirm('Are you sure you want to clear the chat context? This cannot be undone.')) return
  
  try {
    // Call backend to reset agent memory
    await fetch(new URL('/api/chat/reset', environmentStore.backendUrl).toString(), {
        method: 'POST'
    })
    
    // Clear local messages
    messages.value = []
    
    // Add system message
    messages.value.push({
        type: 'assistant',
        content: 'Conversation history has been reset.',
        time: formatTime()
    })
    
  } catch (e) {
      console.error("Failed to reset chat:", e)
      messages.value.push({
          type: 'error',
          content: 'Failed to reset chat context.',
          time: formatTime()
      })
  }
}

// Method to send agent message programmatically (called from parent)
const sendAgentMessage = async (message: string): Promise<void> => {
  if (!message || !isConnected.value) {
    console.warn('Cannot send agent message: not connected or empty message')
    return
  }

  // Add user message to UI (marked as system-initiated)
  messages.value.push({
    type: 'user',
    ...automatedRequest(message),
    time: formatTime()
  })

  isSending.value = true
  await scrollToBottom()

  // Prepare file context if files are uploaded
  let fileContext: FileContext | null = null
  if (wizardData.value.uploadedFiles && wizardData.value.uploadedFiles.length > 0) {
    const fileNames = wizardData.value.uploadedFiles.map(f => f.name).join(', ')
    fileContext = {
      name: fileNames,
      path: wizardData.value.dataSourceValue || wizardData.value.uploadedFiles.map(f => `./Attachments/${f.name}`).join(',')
    }
  }

  // Prepare wizard context from input stage configuration
  let wizardContext: WizardContext | null = null
  if (wizardData.value.viewMode !== 'shop-floor' && (wizardData.value.submodelTemplate || wizardData.value.targetAAS || wizardData.value.aasServerEndpoint)) {
    wizardContext = {
      selectedTemplates: Array.isArray(wizardData.value.submodelTemplate)
        ? wizardData.value.submodelTemplate
        : wizardData.value.submodelTemplate
          ? [wizardData.value.submodelTemplate]
          : [],
      aasServerEndpoint: wizardData.value.aasServerEndpoint || environmentStore.aasServerUrl,
      targetAAS: wizardData.value.targetAAS || null,
      dataSourceType: wizardData.value.dataSourceType || null,
      editedEntitiesFile: wizardData.value.editedEntitiesFile || null
    }
  }

  // Prepare Shop Floor context
  let shopFloorContext: ShopFloorContext | null = null
  if (wizardData.value.viewMode === 'shop-floor') {
      shopFloorContext = {
          assets: wizardData.value.assets || [],
          connections: wizardData.value.connections || []
      }
  }

  // Build and send message using service
  const chatMessage = buildChatMessage(message, fileContext, wizardContext, shopFloorContext)
  send(chatMessage)
}

const handleAskUserOption = (option: string, index: number): void => {
  const msg = messages.value[index]
  if (msg.type === 'tool_call' && msg.tool_name === 'ask_user') {
    (msg as ToolCallMessage).answeredWith = option
  }
  sendAgentMessage(option)
}

// Watch for file uploads
watch(() => wizardData.value.uploadedFiles, (newFiles, oldFiles) => {
  if (newFiles && newFiles.length > 0 && isConnected.value) {
    const oldCount = oldFiles ? oldFiles.length : 0
    const newCount = newFiles.length

    // Only show message if files were added
    if (newCount > oldCount) {
      const fileNames = newFiles.map(f => f.name).join(', ')
      const fileWord = newFiles.length === 1 ? 'file' : 'files'
      messages.value.push({
        type: 'assistant',
        content: `📎 ${newFiles.length} ${fileWord} uploaded: **${fileNames}**\n\nI can now help you extract data from ${newFiles.length === 1 ? 'this file' : 'these files'}. Just ask me to "extract data from the uploaded PDFs" or "read the PDFs and extract technical specifications".`,
        time: formatTime()
      })
      scrollToBottom()
    }
  }
}, { deep: true })

// Watch for step changes and provide contextual hints
watch(currentStep, (newStep: number) => {
  if (!isConnected.value) return

  setTimeout(() => {
    let hint = ''
    switch (newStep) {
      case 0:
        hint = "You're at the input stage. I can help you select the right submodel template and guide you through the data source options."
        break
      case 1:
        hint = "Data extraction is in progress. I can extract technical specifications from your PDF or web source."
        break
      case 2:
        hint = "Now we'll match your extracted properties to standardized AAS properties. I can explain the confidence scores or help adjust mappings."
        break
      case 3:
        hint = "Your submodel has been generated! I can explain the structure or help you with the download options."
        break
    }
    if (hint) {
      messages.value.push({
        type: 'assistant',
        content: hint,
        time: formatTime()
      })
      scrollToBottom()
    }
  }, 1000)
})

onMounted(() => {
  // Connect to WebSocket
  connect()

  // Add welcome message when connected
  watch(isConnected, (connected) => {
      if (connected && messages.value.length === 0) {
        messages.value.push({
          type: 'assistant',
          content: "Ready to monitor faults and help with production or maintenance.",
          time: formatTime()
        })
        scrollToBottom()
      }
  }, { immediate: true })

  // Subscribe to WebSocket messages
  const unsubscribe = onMessage(handleWebSocketMessage)

  // Cleanup on unmount
  onUnmounted(() => {
    unsubscribe()
    disconnect()
  })
})

// Expose sendAgentMessage for parent component to call
/**
 * Put a failure from a deterministic wizard call into the transcript.
 *
 * The stage buttons no longer go through the agent, so a failure in one has
 * nowhere to appear -- and an empty table with no message is what the old
 * polling version showed, which is indistinguishable from still working. The
 * pane is where the user is already looking.
 */
const showError = (text: string): void => {
  messages.value.push({
    type: 'error',
    content: text,
    // `time: formatTime()`, as every other push does. This said
    // `timestamp: new Date()` -- a key the template does not render and a type
    // `ChatMessage` does not declare -- so wizard failures were the only
    // messages in the transcript with no time on them.
    time: formatTime()
  })
  scrollToBottom()
}

/**
 * Say what a slow stage is doing, without calling it a failure.
 *
 * A matching run that outlives the client's timeout has not failed -- the
 * server never heard the abort and is still working. That needs saying in the
 * pane while the wizard waits for the plan, and saying it in red would be a
 * lie about what happened.
 */
const showInfo = (text: string): void => {
  messages.value.push({
    type: 'assistant',
    content: text,
    time: formatTime()
  })
  scrollToBottom()
}

defineExpose({
  sendAgentMessage,
  showError,
  showInfo
})
</script>

<style scoped>
.request-details{white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto;font-size:12px}.user-content details{margin-top:8px}.user-content summary{cursor:pointer;font-size:12px}
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.chat-header {
  display: none;
}

.reset-chat-btn-input {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid #d1d5db;
  background: white;
  color: #6b7280;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  font-size: 16px;
  margin-right: 4px;
}

.reset-chat-btn-input:hover {
  background: #f3f4f6;
  color: #374151;
  border-color: #9ca3af;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: #f9fafb;
}

.chat-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #6b7280;
  text-align: center;
  padding: 24px;
}

.chat-empty-state .icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.chat-empty-state h4 {
  margin: 0 0 8px;
  font-size: 18px;
  color: #374151;
}

.chat-empty-state p {
  margin: 0;
  font-size: 14px;
  line-height: 1.5;
}

.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  animation: fadeIn 0.3s ease-out;

  /* A flex item's default `min-width: auto` is its content's intrinsic width,
     so a long unbroken path makes the row wider than the pane no matter what
     `max-width` the bubble declares -- the bubble is measured against a track
     that has already grown. This is the rule that actually confines it. */
  min-width: 0;
}

.chat-message > * {
  min-width: 0;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-avatar {
  width: 32px;
  height: 32px;
  /* Exempt from the shrink rule above: an avatar squeezed to nothing is worse
     than a slightly narrower bubble. */
  flex: 0 0 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
  background: #e5e7eb;
}

.message-avatar.user {
  background: #179C7D;
  color: white;
}

.message-avatar.assistant {
  background: #dbeafe;
  color: #1e40af;
}

.message-avatar.error {
  background: #fee;
  color: #c00;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);

  /* A bubble had no width constraint and no wrapping rule, so one long token
     pushed it past the pane and the pane grew a horizontal scrollbar -- which
     is how a message ends up half off-screen with its text unreadable.
     The instructions this agent is sent are full of absolute Windows paths,
     and a path has no spaces to break at, so `overflow-wrap: normal` finds
     nowhere to wrap and gives up. `anywhere` is the rule that breaks inside an
     unbroken run; `break-word` alone still refuses in a flex child. */
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
  /* Newlines in an agent's report are meaningful -- it writes lists and
     indented commands -- and collapsing them makes a structured answer read as
     one paragraph. */
  white-space: pre-wrap;
}

.user-content {
  background: #179C7D;
  color: white;
  border-top-left-radius: 2px;
}

.assistant-content {
  background: #fff;
  color: #374151;
  border: 1px solid #e5e7eb;
  border-top-left-radius: 2px;
}

.thinking-content {
  background: #f3f4f6;
  color: #6b7280;
  font-style: italic;
  border: 1px solid #e5e7eb;
}

.error-content {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fecaca;
}

.message-time {
  font-size: 11px;
  color: #9ca3af;
  margin-top: 4px;
  text-align: right;
}

.chat-input-area {
  padding: 16px;
  background: white;
  border-top: 1px solid #e5e7eb;
}

.chat-input-wrapper {
  display: flex;
  gap: 8px;
  background: #f3f4f6;
  padding: 8px;
  border-radius: 24px;
  border: 1px solid #e5e7eb;
  transition: all 0.2s;
}

.chat-input-wrapper:focus-within {
  background: white;
  border-color: #179C7D;
  box-shadow: 0 0 0 3px rgba(23, 156, 125, 0.1);
}

.chat-input {
  flex: 1;
  border: none;
  background: transparent;
  padding: 8px;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  outline: none;
  max-height: 100px;
}

.chat-send-btn {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: #179C7D;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.chat-send-btn:hover:not(:disabled) {
  background: #138a68;
  transform: scale(1.05);
}

.chat-send-btn:disabled {
  background: #d1d5db;
  cursor: not-allowed;
  transform: none;
}

.chat-send-btn.stop-mode {
  background: #b91c1c;
}

.chat-send-btn.stop-mode:hover:not(:disabled) { background: #991b1b; }

.chat-send-btn.stop-mode:disabled { background: #b91c1c; opacity: .55; }

.stop-square { width: 12px; height: 12px; border-radius: 2px; background: currentColor; }

.spinner-small {
  border: 2px solid #f3f3f3;
  border-top: 2px solid #179C7D;
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 1s linear infinite;
  display: inline-block;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.tool-call-card {
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: 8px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.3s;
}

.tool-call-card:hover {
  background: #fef08a;
}

.tool-call-card.expanded {
  background: #fffbeb;
}

.tool-call-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.tool-icon {
  font-size: 16px;
}

.tool-status {
  margin-left: 4px;
  font-size: 14px;
  color: #15803d;
}

.tool-status.running {
  color: #c2410c;
  animation: pulse 1.5s ease-in-out infinite;
}

.expand-icon {
  margin-left: auto;
  color: #92400e;
  font-size: 12px;
}

.tool-call-details {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #fcd34d;
}

.tool-section {
  margin-bottom: 12px;
}

.tool-section:last-child {
  margin-bottom: 0;
}

.tool-section-title {
  font-size: 12px;
  font-weight: 600;
  color: #92400e;
  margin-bottom: 6px;
}

.tool-code {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 4px;
  padding: 8px;
  font-size: 11px;
  font-family: 'Courier New', monospace;
  margin: 0;
  color: #78350f;

  /* A command line is code: wrapping it mid-token would make it unreadable and
     un-copyable, so this one scrolls inside its own box rather than wrapping.
     `max-width` is what keeps that scroll local -- without it the block sets
     the width of everything above it and the whole pane scrolls instead. */
  max-width: 100%;
  overflow-x: auto;
  white-space: pre;
}

.connection-status {
  padding: 16px;
  text-align: center;
}

.status-message {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 20px;
  font-size: 13px;
  color: #075985;
}

.status-message.error {
  background: #fee;
  color: #c00;
  border-color: #fcc;
}

.btn-reconnect {
  margin-left: 8px;
  padding: 4px 12px;
  background: #179C7D;
  color: white;
  border: none;
  border-radius: 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-reconnect:hover {
  background: #138a68;
}

.message-avatar.ask-user {
  background: #e0f2fe;
  color: #0369a1;
}

.ask-user-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-top-left-radius: 2px !important;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  padding: 16px;
}

.ask-user-header {
  color: #1e293b;
  margin-bottom: 12px;
  font-size: 15px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.ask-user-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 8px;
}

.ask-user-btn {
  background: #ffffff;
  border: 1px solid #179C7D;
  color: #179C7D;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
}

.ask-user-btn:hover {
  background: #179C7D;
  color: #ffffff;
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(23, 156, 125, 0.2);
}

.ask-user-btn:active {
  transform: translateY(0);
}

.answered-state {
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  color: #065f46;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
}

.check-icon {
  font-weight: bold;
  margin-right: 6px;
}

.ask-user-hint {
  color: #64748b;
  margin-top: 8px;
}

</style>
