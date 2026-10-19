import { api } from './api'

/**
 * Type definitions for File service
 */

export interface Template {
  id: string
  name: string
  idShort: string
  description: string
}

export interface UploadResponse {
  success: boolean
  filename: string
  path: string
  size: number
}

/**
 * File Service
 * Handles file upload/download and template operations
 */
export const fileService = {
  /**
   * Get available submodel templates from AAS server
   */
  async getTemplates(endpoint?: string): Promise<Template[]> {
    const { data } = await api.get<Template[]>('/api/templates', {
      params: endpoint ? { endpoint } : undefined,
    })
    return data
  },

  /**
   * Upload a file (PDF, AASX, etc.)
   */
  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const { data } = await api.post<UploadResponse>('/api/upload', formData)
    return data
  },

  /**
   * Validate file before upload
   */
  validateFile(file: File, options?: {
    maxSize?: number // in bytes
    allowedTypes?: string[]
  }): { valid: boolean; error?: string } {
    const { maxSize = 10 * 1024 * 1024, allowedTypes = ['application/pdf'] } = options || {}

    // Check file size
    if (file.size > maxSize) {
      return {
        valid: false,
        error: `File size exceeds ${Math.round(maxSize / 1024 / 1024)}MB limit`,
      }
    }

    // Check file type
    const fileType = file.type
    const fileName = file.name.toLowerCase()

    const isValid = allowedTypes.some(type => {
      if (type.startsWith('.')) {
        return fileName.endsWith(type.toLowerCase())
      }
      return fileType === type
    })

    if (!isValid) {
      return {
        valid: false,
        error: `File type ${file.type || 'unknown'} is not allowed. Allowed types: ${allowedTypes.join(', ')}`,
      }
    }

    return { valid: true }
  },

  /**
   * Format file size for display
   */
  formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  },
}

// Export individual functions for convenience
export const { getTemplates, uploadFile, validateFile, formatFileSize } = fileService
