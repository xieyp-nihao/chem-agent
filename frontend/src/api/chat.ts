import { api } from './client'
import type { ChatMessage, ChatResponse } from './types'

/** POST /chat action=message */
export function sendMessage(params: ChatMessage): Promise<ChatResponse> {
  return api.post<ChatResponse>('/chat', {
    message: params.message,
    paper_id: params.paper_id || undefined,
    session_id: params.session_id || undefined,
    action: 'message',
  })
}
