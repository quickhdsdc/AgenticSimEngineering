import { api, MODEL_BOUND_TIMEOUT } from './api'
import { useEnvStore } from '../stores/environmentStore'


/**
 * Type definitions for AAS service
 */

export interface ExtractionCheckResponse {
  exists: boolean
  filename?: string
  /** True when the user's corrections are what came back, not the extractor's. */
  edited?: boolean
  /**
   * Where the extraction has to be written. The server owns this naming
   * convention and returns it so the wizard can put it in the agent's prompt;
   * rebuilding the rule here would let the two copies drift.
   */
  expectedPath?: string
}

export interface AASCheckResponse {
  exists: boolean
  id?: string
}

export interface ExtractedEntity {
  entity: string
  value: string
  description: string
  unit?: string
  /** Which configuration of the product this value belongs to; "" for all. */
  variant?: string
  selected?: boolean
}

export interface ExtractedEntitiesResponse {
  success: boolean
  entities: ExtractedEntity[]
  count: number
  /** Configurations the document distinguishes. Empty means one product. */
  configurations?: string[]
  /** What the extraction could not do cleanly, for a person to check. */
  warnings?: string[]
}

export interface SaveEditedEntitiesRequest {
  original_filename: string
  entities: ExtractedEntity[]
  /** The configuration the asset actually is, when the document lists several. */
  configuration?: string
  configurations?: string[]
  warnings?: string[]
}

export interface SaveEditedEntitiesResponse {
  success: boolean
  filename: string
  /** Absolute path. The matching prompt hands it to `aas match --entities`. */
  path: string
  count?: number
  dropped?: number
}

export interface MatchingCheckResponse {
  exists: boolean
  filename?: string
  edited?: boolean
  /** Where the match plan has to be written; see ExtractionCheckResponse. */
  expectedPath?: string
}

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

export interface MatchingResultsResponse {
  success: boolean
  results: MatchedProperty[]
  count: number
}

export interface SaveMatchingResultsRequest {
  filename: string
  results: MatchedProperty[]
  aas_idShort?: string
}

export interface SaveMatchingResultsResponse {
  success: boolean
  /** Absolute path to the manifest listing one plan per submodel. */
  path: string
  plans?: Array<{ submodel: string; path: string; counts?: Record<string, number> }>
  /** Rows whose target submodel could not be established. Never silently dropped. */
  unresolved?: Array<{ entity: string; reason: string }>
}

export interface AASProperty {
  idShort: string
  type: string
  semanticPath: string
  apiPath: string
  semanticId?: string
  description?: string
  valueType?: string
}

export interface AASPropertiesResponse {
  success: boolean
  properties: AASProperty[]
  grouped: Record<string, AASProperty[]>
  count: number
}

/**
 * AAS Service
 * Handles all AAS-related API operations
 */
export const aasService = {
  /**
   * Check if extraction results exist for a file
   */
  async checkExtraction(filename: string): Promise<ExtractionCheckResponse> {
    const { data } = await api.get<ExtractionCheckResponse>(
      `/api/check-extraction/${encodeURIComponent(filename)}`
    )
    return data
  },

  /**
   * Check if AAS exists on the server
   */
  async checkAAS(aasIdShort: string, endpoint?: string): Promise<AASCheckResponse> {
    const { data } = await api.get<AASCheckResponse>(
      `/api/check-aas/${encodeURIComponent(aasIdShort)}`,
      {
        params: endpoint ? { endpoint } : undefined,
      }
    )
    return data
  },

  /**
   * Get extracted entities from a file
   */
  async getExtractedEntities(filename: string): Promise<ExtractedEntitiesResponse> {
    const { data } = await api.get<ExtractedEntitiesResponse>(
      `/api/extracted-entities/${encodeURIComponent(filename)}`
    )
    return data
  },

  /**
   * Save edited entities
   */
  async saveEditedEntities(
    request: SaveEditedEntitiesRequest
  ): Promise<SaveEditedEntitiesResponse> {
    const { data } = await api.post<SaveEditedEntitiesResponse>(
      '/api/save-edited-entities',
      request
    )
    return data
  },

  /**
   * Run the matching for one document against one AAS.
   *
   * A stage button is a deterministic commitment, so it calls a function. The
   * wizard used to send the agent a prompt naming the exact commands and then
   * poll for a file: a deterministic script run by a non-deterministic runner,
   * which failed the way that fails -- the agent matched all four submodels
   * correctly and never wrote the manifest being polled for, so a finished job
   * showed as "no matches found yet" indefinitely.
   *
   * The agent is still on the socket for the open-ended half: why a row did
   * not match, or moving one somewhere else. It answers those by reading the
   * plan, which carries every score, band and rejected candidate.
   */
  async runMatch(
    filename: string,
    aasIdShort: string,
    endpoint?: string
  ): Promise<MatchingResultsResponse & { path: string; submodels: string[] }> {
    const { data } = await api.post<any>(
      '/api/match',
      {
        filename,
        aas_id_short: aasIdShort,
        endpoint,
      },
      // One model round trip per middle-band entity, times the number of
      // submodels. The default read timeout is not a budget for that.
      { timeout: MODEL_BOUND_TIMEOUT }
    )
    return data
  },

  /**
   * Wait for a match that is still running server-side.
   *
   * Aborting a fetch stops the browser waiting, not the server working. The
   * plan is written when the run finishes, so a timed-out matching run is
   * recovered by watching for the file rather than by reporting a failure that
   * did not happen.
   *
   * Safe against picking up a stale plan: the caller checks for an existing
   * one *before* starting the run and returns early if it finds it, so
   * anything appearing after that point belongs to this run.
   */
  async waitForMatch(
    filename: string,
    aasIdShort: string,
    { attempts = 60, everyMs = 5000 }: { attempts?: number; everyMs?: number } = {}
  ): Promise<MatchedProperty[] | null> {
    for (let i = 0; i < attempts; i++) {
      await new Promise(resolve => setTimeout(resolve, everyMs))
      try {
        const check = await aasService.checkMatching(filename, aasIdShort)
        if (check.exists && check.filename) {
          const results = await aasService.getMatchingResults(check.filename, aasIdShort)
          if (results.success && results.results.length > 0) return results.results
        }
      } catch {
        // A poll that fails is not an answer. Keep waiting; the deadline is
        // the attempt count, not the first hiccup.
      }
    }
    return null
  },

  /**
   * Check if matching results exist for a file
   */
  async checkMatching(filename: string, aasIdShort?: string): Promise<MatchingCheckResponse> {
    const params = aasIdShort ? { aas_id_short: aasIdShort } : undefined
    const { data } = await api.get<MatchingCheckResponse>(
      `/api/check-matching/${encodeURIComponent(filename)}`,
      { params }
    )
    return data
  },

  /**
   * Get matching results from a file
   */
  async getMatchingResults(filename: string, aasIdShort?: string): Promise<MatchingResultsResponse> {
    const params = aasIdShort ? { aas_id_short: aasIdShort } : undefined
    const { data } = await api.get<MatchingResultsResponse>(
      `/api/matching-results/${encodeURIComponent(filename)}`,
      { params }
    )
    return data
  },

  /**
   * Save matching results
   */
  async saveMatchingResults(
    request: SaveMatchingResultsRequest
  ): Promise<SaveMatchingResultsResponse> {
    const { data } = await api.post<SaveMatchingResultsResponse>(
      '/api/save-matching-results',
      request
    )
    return data
  },

  /**
   * Get available AAS properties for a given AAS
   */
  async getAASProperties(aasIdShort: string): Promise<AASPropertiesResponse> {
    const { data } = await api.get<AASPropertiesResponse>(
      `/api/aas-properties/${encodeURIComponent(aasIdShort)}`
    )
    return data
  },

  /**
   * Trigger AAS parsing to CSV
   */
  async parseAAS(aasIdShort: string, endpoint?: string): Promise<any> {
    const { data } = await api.post<any>(
      `/api/parse-aas/${encodeURIComponent(aasIdShort)}`,
      {},
      {
        params: endpoint ? { endpoint } : undefined,
      }
    )
    return data
  },

  /**
   * Download AAS as AASX package
   */
  async downloadAASX(aasIdShort: string, endpoint?: string): Promise<Blob> {
    const blob = await api.downloadFile(
      `/api/download/aasx/${encodeURIComponent(aasIdShort)}`,
      {
        params: endpoint ? { endpoint } : undefined,
      }
    )
    return blob
  },

  /**
   * Download AAS as JSON
   */
  async downloadJSON(aasIdShort: string, endpoint?: string): Promise<Blob> {
    const blob = await api.downloadFile(
      `/api/download/json/${encodeURIComponent(aasIdShort)}`,
      {
        params: endpoint ? { endpoint } : undefined,
      }
    )
    return blob
  },

  /**
   * Helper: Trigger download of blob with filename
   */
  triggerDownload(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  },

  /**
   * Preview a structured file
   */
  async previewStructuredFile(filename: string): Promise<any> {
    const { data } = await api.get<any>(`/api/preview-structured-file/${encodeURIComponent(filename)}`)
    return data
  },

  /**
   * Get list of available AASs from the server
   */
  async getAASList(endpoint?: string): Promise<any[]> {
    const { data } = await api.get<any[]>(
      '/api/aas-list',
      {
        params: endpoint ? { endpoint } : undefined,
      }
    )
    return data
  },

  /**
   * Instantiate AAS from Blueprint
   */
  async instantiateAAS(blueprintId: string, newAasId: string, newAasIdShort: string, endpoint?: string): Promise<any> {
    const { data } = await api.post<any>(
      '/api/instantiate-aas',
      {
        blueprint_id: blueprintId,
        new_aas_id: newAasId,
        new_aas_id_short: newAasIdShort,
        endpoint: endpoint || useEnvStore().aasServerUrl
      }
    )
    return data
  },

  /**
   * Delete AAS and its submodels
   */
  async deleteAAS(aasId: string, endpoint?: string): Promise<any> {
    const { data } = await api.delete<any>(
      `/api/delete-aas/${encodeURIComponent(aasId)}`,
      {
        params: endpoint ? { endpoint } : undefined
      }
    )
    return data
  },


  /**
   * Create a connection relationship between two AAS
   */
  async createConnection(sourceAasId: string, targetAasId: string, ruleId?: string, endpoint?: string): Promise<any> {
    const { data } = await api.post<any>(
      '/api/create-connection',
      {
        source_aas_id: sourceAasId,
        target_aas_id: targetAasId,
        rule_id: ruleId,
        endpoint: endpoint || useEnvStore().aasServerUrl
      }
    )
    return data
  },

  /**
   * Delete a connection relationship
   */
  async deleteConnection(sourceAasId: string, targetAasId: string, relationshipId: string, endpoint?: string): Promise<any> {
    const { data } = await api.post<any>(
      '/api/delete-connection',
      {
        source_aas_id: sourceAasId,
        target_aas_id: targetAasId,
        relationship_sm_id: relationshipId,
        endpoint: endpoint || useEnvStore().aasServerUrl
      }
    )
    return data
  },

  /**
   * Refreshes the AAS Service and Resource Manifests
   */
  async refreshManifests(): Promise<{status: string, message: string}> {
    const { data } = await api.post<{status: string, message: string}>('/api/manifests/refresh')
    return data
  }
}

// Export individual functions for convenience
export const {
  checkExtraction,
  checkAAS,
  getExtractedEntities,
  saveEditedEntities,
  checkMatching,
  runMatch,
  getMatchingResults,
  saveMatchingResults,
  getAASProperties,
  parseAAS,
  downloadAASX,
  downloadJSON,
  triggerDownload,
  getAASList,
  previewStructuredFile,
  instantiateAAS,
  deleteAAS,
  createConnection,
  deleteConnection,
  refreshManifests,
} = aasService
