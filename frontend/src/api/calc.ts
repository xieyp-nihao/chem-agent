import { api, requestBlob } from './client'
import type {
  CalcSlots,
  ConfirmRequest,
  ConfirmResponse,
  HubbardU,
  KPoints,
} from './types'

export interface ConfirmPayload {
  decision: 'approve' | 'abort' | 'patch'
  patch?: {
    encut_ev?: number
    kpoints?: KPoints
    hubbard_u?: HubbardU[]
  }
  comment?: string
}

/** POST /calc/draft（JSON 入口） */
export function draftCalc(slots: CalcSlots): Promise<ConfirmRequest> {
  return api.post<ConfirmRequest>('/calc/draft', { slots })
}

/** POST /chat action=confirm（approve/patch/abort） */
export function confirmCalc(
  sessionId: string,
  payload: ConfirmPayload,
): Promise<ConfirmResponse> {
  return api.post<ConfirmResponse>('/chat', {
    session_id: sessionId,
    action: 'confirm',
    payload,
  })
}

/** GET /files/{path} → Blob（页面负责触发下载） */
export function downloadFile(path: string): Promise<Blob> {
  return requestBlob(path)
}
