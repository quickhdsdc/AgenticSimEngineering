import { useEnvStore } from '../stores/environmentStore'

/**
 * API Response wrapper
 */
export interface ApiResponse<T = any> {
  data: T
  status: number
  statusText: string
  headers: Headers
}

/**
 * API Error class for better error handling
 */
export class ApiError extends Error {
  status: number
  statusText: string
  data: any

  constructor(message: string, status: number, statusText: string, data?: any) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.statusText = statusText
    this.data = data
  }
}

/**
 * The server's own explanation, when it sent one.
 *
 * FastAPI puts it in `detail`, as a string for an `HTTPException` and as a list
 * of field errors for a validation failure. Both are worth showing; neither is
 * worth showing as `[object Object]`.
 */
function detailOf(data: any): string | null {
  const detail = data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((item: any) =>
        typeof item === 'string'
          ? item
          : `${(item?.loc || []).join('.')}: ${item?.msg || 'invalid'}`)
      .join('; ')
  }
  return null
}

/**
 * How long a request may take before the client gives up on it.
 *
 * The default suits a read: a listing, a tree, a file. It does not suit a
 * stage that calls a language model once per extracted entity -- a 317-entity
 * datasheet against four submodels is minutes of work, and aborting it at 30
 * seconds reported "Matching failed" for a job that was running normally and
 * went on to finish.
 *
 * Note what the abort does and does not do. `AbortController` stops the
 * *client* waiting; the server has no idea and keeps working. So a timeout
 * here is never evidence that the work failed, only that the browser stopped
 * watching -- which is why the matching stage falls back to polling for the
 * plan rather than reporting an error.
 */
export const DEFAULT_TIMEOUT = 30_000
export const MODEL_BOUND_TIMEOUT = 20 * 60_000

/**
 * Request configuration options
 */
export interface RequestConfig extends RequestInit {
  params?: Record<string, string | number | boolean>
  timeout?: number
}

/**
 * Base API client class
 */
class ApiClient {
  private baseURL: string = ''
  private defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  }

  /**
   * Get base URL from environment store (lazy initialization)
   */
  private getBaseURL(): string {
    if (!this.baseURL) {
      const envStore = useEnvStore()
      this.baseURL = envStore.backendUrl
    }
    return this.baseURL
  }

  /**
   * Build URL with query parameters
   */
  private buildURL(endpoint: string, params?: Record<string, string | number | boolean>): string {
    const baseURL = this.getBaseURL() // Get base URL lazily
    const url = new URL(endpoint, baseURL)

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value))
      })
    }

    return url.toString()
  }

  /**
   * Make HTTP request with error handling
   */
  private async request<T = any>(
    endpoint: string,
    config: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const { params, timeout = DEFAULT_TIMEOUT, headers, skipDefaultHeaders, ...fetchConfig } = config as any

    const url = this.buildURL(endpoint, params)

    // Merge headers - skip default headers for FormData uploads
    const mergedHeaders = skipDefaultHeaders
      ? { ...headers }  // Only use provided headers (browser will set Content-Type for FormData)
      : {
          ...this.defaultHeaders,
          ...headers,
        }

    // Create abort controller for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      const response = await fetch(url, {
        ...fetchConfig,
        headers: mergedHeaders,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      // Parse response
      let data: T
      const contentType = response.headers.get('content-type')

      if (contentType?.includes('application/json')) {
        data = await response.json()
      } else if (contentType?.includes('text/')) {
        data = (await response.text()) as any
      } else {
        // For binary data (files, etc.), return as blob
        data = (await response.blob()) as any
      }

      // Check if response is ok
      if (!response.ok) {
        // The backend answers a failure with a `detail` saying what went wrong
        // and, on a transport error, which endpoint it tried. Reporting only
        // "HTTP 502: Bad Gateway" throws that away and leaves the user with a
        // red box that names no cause -- the exact thing the backend's own
        // no-mock-fallback rule exists to avoid.
        throw new ApiError(
          detailOf(data) || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          response.statusText,
          data
        )
      }

      return {
        data,
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      }
    } catch (error) {
      clearTimeout(timeoutId)

      // Handle abort/timeout
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError('Request timeout', 408, 'Request Timeout')
      }

      // Re-throw ApiError
      if (error instanceof ApiError) {
        throw error
      }

      // Handle network errors
      throw new ApiError(
        error instanceof Error ? error.message : 'Network error',
        0,
        'Network Error'
      )
    }
  }

  /**
   * GET request
   */
  async get<T = any>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...config, method: 'GET' })
  }

  /**
   * POST request
   */
  async post<T = any>(
    endpoint: string,
    data?: any,
    config?: RequestConfig
  ): Promise<ApiResponse<T>> {
    // For FormData, we need to let the browser set Content-Type with boundary
    const isFormData = data instanceof FormData
    const headers = isFormData
      ? { ...config?.headers }  // Don't include defaultHeaders for FormData
      : { ...config?.headers }

    return this.request<T>(endpoint, {
      ...config,
      method: 'POST',
      body: isFormData ? data : JSON.stringify(data),
      headers,
      // Add a flag to skip default headers for FormData
      skipDefaultHeaders: isFormData,
    } as any)
  }

  /**
   * PUT request
   */
  async put<T = any>(
    endpoint: string,
    data?: any,
    config?: RequestConfig
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  /**
   * PATCH request
   */
  async patch<T = any>(
    endpoint: string,
    data?: any,
    config?: RequestConfig
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  /**
   * DELETE request
   */
  async delete<T = any>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...config, method: 'DELETE' })
  }

  /**
   * Download file (returns blob)
   */
  async downloadFile(endpoint: string, config?: RequestConfig): Promise<Blob> {
    const response = await this.request<Blob>(endpoint, {
      ...config,
      method: 'GET',
    })
    return response.data
  }

  /**
   * Set default header
   */
  setHeader(key: string, value: string) {
    this.defaultHeaders = {
      ...this.defaultHeaders,
      [key]: value,
    }
  }

  /**
   * Remove default header
   */
  removeHeader(key: string) {
    const headers = { ...this.defaultHeaders }
    delete headers[key]
    this.defaultHeaders = headers
  }
}

// Export singleton instance
export const api = new ApiClient()

// Export class for testing/customization
export default ApiClient
