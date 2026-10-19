import { ref, computed, type Ref } from 'vue'
import { useEnvStore } from '../stores/environmentStore'

/**
 * Type definitions for WebSocket service
 */

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected'

export interface FileContext {
  name: string
  path: string
}

export interface WizardContext {
  selectedTemplates: string[]
  aasServerEndpoint: string
  targetAAS: string | null
  dataSourceType: string | null
  editedEntitiesFile: string | null
}

export interface ShopFloorContext {
  assets: any[]
  connections: any[]
}

export interface UserChatMessage {
  message: string
  file_context?: FileContext | null
  wizard_context?: WizardContext | null
  shop_floor_context?: ShopFloorContext | null
}

export type ChatMessage = UserChatMessage | { type: 'stop' }

export interface WebSocketMessage {
  type: 'user_message' | 'thinking' | 'assistant_message' | 'tool_call' | 'tool_result' | 'awaiting_input' | 'complete' | 'error' | 'state_update' | 'stopping' | 'stopped'
  content?: string
  tool_name?: string
  tool_args?: any
  tool_result?: any
  tool_status?: string
  result?: any
  key?: string
  value?: any
}

export type MessageHandler = (data: WebSocketMessage) => void

/**
 * WebSocket Service Composable
 * Provides reactive WebSocket connection management for the chat system
 */
export function useWebSocket() {
  const ws: Ref<WebSocket | null> = ref(null)
  const connectionStatus: Ref<ConnectionStatus> = ref('disconnected')
  const messageHandlers: Set<MessageHandler> = new Set()

  const isConnected = computed(() => connectionStatus.value === 'connected')

  /**
   * Get WebSocket URL from environment
   */
  const getWebSocketURL = (): string => {
    const envStore = useEnvStore()
    const backendUrl = envStore.backendUrl
    const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://')
    return `${wsUrl}/ws/chat`
  }

  /**
   * Connect to WebSocket server
   */
  const connect = (): void => {
    if (ws.value?.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected')
      return
    }

    connectionStatus.value = 'connecting'
    const url = getWebSocketURL()

    console.log('Connecting to WebSocket:', url)
    ws.value = new WebSocket(url)

    ws.value.onopen = () => {
      console.log('WebSocket connected')
      connectionStatus.value = 'connected'
    }

    ws.value.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data)
        console.log('WebSocket message received:', data)

        // Notify all registered handlers
        messageHandlers.forEach((handler) => handler(data))
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.value.onerror = (error: Event) => {
      console.error('WebSocket error:', error)
      connectionStatus.value = 'disconnected'
    }

    ws.value.onclose = () => {
      console.log('WebSocket disconnected')
      connectionStatus.value = 'disconnected'
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = (): void => {
    if (ws.value) {
      ws.value.close()
      ws.value = null
    }
    connectionStatus.value = 'disconnected'
  }

  /**
   * Send a message through WebSocket
   */
  const send = (data: ChatMessage): void => {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected')
      return
    }

    const payload = JSON.stringify(data)
    console.log('Sending WebSocket message:', payload)
    ws.value.send(payload)
  }

  /**
   * Send a state update to the backend
   */
  const sendStateUpdate = (key: string, value: any): void => {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      // Quietly fail if not connected, as this might happen during init
      return
    }

    const payload = JSON.stringify({
      type: 'state_update',
      key,
      value
    })
    ws.value.send(payload)
  }

  /**
   * Register a message handler
   */
  const onMessage = (handler: MessageHandler): (() => void) => {
    messageHandlers.add(handler)

    // Return unsubscribe function
    return () => {
      messageHandlers.delete(handler)
    }
  }

  /**
   * Reconnect to WebSocket
   */
  const reconnect = (): void => {
    disconnect()
    setTimeout(() => {
      connect()
    }, 100)
  }

  /**
   * Check if WebSocket is ready to send
   */
  const isReady = computed(() => {
    return ws.value?.readyState === WebSocket.OPEN
  })

  return {
    // State
    connectionStatus,
    isConnected,
    isReady,

    // Methods
    connect,
    disconnect,
    reconnect,
    send,
    sendStateUpdate,
    onMessage,
  }
}

/**
 * Singleton WebSocket instance for shared connection
 * Use this when you want a single connection across the app
 */
let sharedWebSocket: ReturnType<typeof useWebSocket> | null = null

export function useSharedWebSocket() {
  if (!sharedWebSocket) {
    sharedWebSocket = useWebSocket()
  }
  return sharedWebSocket
}

/**
 * Helper: Build chat message with contexts
 */
export function buildChatMessage(
  message: string,
  fileContext?: FileContext | null,
  wizardContext?: WizardContext | null,
  shopFloorContext?: ShopFloorContext | null
): ChatMessage {
  return {
    message,
    file_context: fileContext || null,
    wizard_context: wizardContext || null,
    shop_floor_context: shopFloorContext || null
  }
}
