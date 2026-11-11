import { api, request } from './client'
import type {
  DocumentStatus,
  ReviewDecision,
  ReviewResponse,
  UploadResponse,
} from './types'

/** POST /documents/upload（multipart；后端返回 202 或重复上传 200）。 */
export function uploadPaper(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return request<UploadResponse>('/documents/upload', { method: 'POST', formData })
}

/** GET /documents/{paper_id} */
export function getPaperStatus(paperId: string): Promise<DocumentStatus> {
  return api.get<DocumentStatus>(`/documents/${encodeURIComponent(paperId)}`)
}

/** POST /documents/{paper_id}/retry（仅 failed/method_not_found 可用） */
export function retryPaper(paperId: string): Promise<DocumentStatus> {
  return api.post<DocumentStatus>(`/documents/${encodeURIComponent(paperId)}/retry`)
}

/** POST /chat action=review */
export function reviewPaper(
  paperId: string,
  decisions: ReviewDecision[],
): Promise<ReviewResponse> {
  return api.post<ReviewResponse>('/chat', {
    paper_id: paperId,
    action: 'review',
    payload: { decisions },
  })
}
