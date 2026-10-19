/**
 * Shared TypeScript types for the AAS Submodel Creator frontend
 */

// Wizard data structure
export interface WizardData {
  // Input stage
  viewMode: 'shop-floor' | 'wizard'
  mode: 'create' | 'edit'
  submodelTemplate: string[]
  aasServerEndpoint: string
  aasUIEndpoint: string
  targetAAS: string
  submodelId?: string
  submodelIdShort?: string
  dataSourceType: 'url' | 'pdf' | 'text' | 'structured'
  dataSourceValue: string
  uploadedFiles: Array<File | { name: string; size: number; type: string; lastModified: number }>

  // Extraction stage
  extractedEntities: Entity[]
  isExtracting: boolean
  editedEntitiesFile: string | null
  /** Configurations the document covers; more than one needs a choice. */
  extractionConfigurations: string[]
  /** The configuration chosen for this asset, "" while none is chosen. */
  chosenConfiguration: string
  /** What the extraction reported it could not do cleanly. */
  extractionWarnings: string[]

  // Matching stage
  matchedProperties: MatchedProperty[]
  isMatching: boolean

  // Generation stage
  isGenerating: boolean
  generationComplete: boolean

  /**
   * A question the machine console handed to the chat pane.
   *
   * Routed through the store rather than opening a second agent: one session
   * per browser, one ledger thread, one place the approval prompt appears. A
   * console with its own runner would be a second unwatched agent against the
   * same machine.
   */
  pendingAgentQuestion: string

  // Shop Floor State
  assets: AssetInstance[]
  connections: Connection[]
}

// Asset Instance on Shop Floor
export interface AssetInstance {
  id: string
  type: string
  label: string
  x: number // percentage
  y: number // percentage
  hasAas: boolean
  aasId?: string // The actual AAS ID on the server
}

export interface Connection {
  id: string
  sourceId: string
  targetId: string
  sourcePort: 'top' | 'right' | 'bottom' | 'left'
  targetPort: 'top' | 'right' | 'bottom' | 'left'
  relationshipId?: string
}

// Entity extracted from PDF
export interface Entity {
  entity: string
  value: string
  description: string
  unit?: string
  /**
   * Which configuration of the product this value belongs to, "" when it holds
   * for all of them. A supplier datasheet describes a product family; an AAS
   * instance describes one asset, and this is what tells them apart.
   */
  variant?: string
}

// Matched property
// Matched property
export interface MatchedProperty {
  extractedEntity: string
  extractedValue: string
  aasProperty: string
  /** The element that would be written or created. */
  aasType?: string
  /** The element at `apiPath`: the slot for a write, the container for a create. */
  targetType?: string
  confidence: number
  semanticId?: string
  matchType?: string
  selected?: boolean
  reason?: string
  // Multi-select support
  aasProperties?: Array<{
    idShort: string
    type: string
    semanticPath?: string
    apiPath?: string
    semanticId?: string
    description?: string
  }>
  // For preserving AI suggestion
  originalMatch?: string
  originalType?: string
  originalConfidence?: number
}

// Chat message
export interface ChatMessage {
  type: 'user' | 'assistant' | 'system' | 'tool' | 'error'
  content: string
  time: string
  tool_name?: string
  tool_input?: any
  status?: 'running' | 'completed' | 'error'
}

// File context for chat
export interface FileContext {
  name: string
  path: string
}

// Wizard context for chat
export interface WizardContext {
  selectedTemplates: string[]
  aasServerEndpoint: string
  targetAAS: string | null
  dataSourceType: string | null
  editedEntitiesFile: string | null
}

// WebSocket message
export interface WebSocketMessage {
  type: string
  content?: string
  data?: any
  status?: string
  tool_name?: string
  tool_input?: any
}

// Submodel template
export interface SubmodelTemplate {
  id: string
  name: string
  version: string
  description: string
}

// API responses
export interface ApiResponse<T = any> {
  data: T
  status: number
  statusText: string
  headers: Headers
}

export interface ExtractionCheckResponse {
  exists: boolean
  path: string | null
  filename: string
  is_edited: boolean
}

export interface ExtractedEntitiesResponse {
  success: boolean
  entities: Entity[]
  filename: string
}

export interface MatchingCheckResponse {
  exists: boolean
  path: string | null
  filename: string
}

export interface MatchingResultsResponse {
  success: boolean
  results: MatchedProperty[]
  filename: string
}

export interface AASCheckResponse {
  exists: boolean
  id: string | null
  endpoint: string
}

export interface UploadResponse {
  path: string
  filename: string
  size: number
}
