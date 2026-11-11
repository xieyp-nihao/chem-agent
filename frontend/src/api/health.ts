import { api } from './client'
import type { HealthData } from './types'

/** GET /health；down 时后端返回 HTTP 503 + envelope code=0，这里放行以便页面展示 down。 */
export function getHealth(): Promise<HealthData> {
  return api.get<HealthData>('/health', { allowNonOkEnvelope: true })
}
