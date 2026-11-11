import type { ApiResponse } from './types'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

/** 统一 API 错误：code/message/data/requestId，供页面展示与排查。 */
export class ApiError extends Error {
  readonly code: number
  readonly data: unknown
  readonly requestId?: string

  constructor(code: number, message: string, data: unknown = null, requestId?: string) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.data = data
    this.requestId = requestId
  }
}

export interface RequestOptions {
  method?: 'GET' | 'POST'
  /** JSON body；与 formData 互斥。 */
  body?: unknown
  /** multipart body；不要手动设置 Content-Type。 */
  formData?: FormData
  headers?: Record<string, string>
  signal?: AbortSignal
  /**
   * /health 在 down 时返回 HTTP 503 + envelope code=0（后端 §7.8 实现）。
   * 开启后允许这种 "HTTP 非 2xx + code=0" 返回 data，以便页面展示红色 down；
   * 其他请求保持默认：HTTP 非 2xx 一律抛 ApiError。
   */
  allowNonOkEnvelope?: boolean
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = `${BASE_URL}${path.startsWith('/') ? path : `/${path}`}`
  const headers: Record<string, string> = { ...(options.headers ?? {}) }

  let body: BodyInit | undefined
  if (options.formData) {
    if (options.body !== undefined) {
      throw new ApiError(-3, 'body 与 formData 不能同时提供')
    }
    body = options.formData
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(options.body)
  }

  let response: Response
  try {
    response = await fetch(url, {
      method: options.method ?? 'GET',
      headers,
      body,
      signal: options.signal,
    })
  } catch (err) {
    throw new ApiError(-1, '网络错误', err)
  }

  const headerRequestId = response.headers.get('X-Request-ID') ?? undefined
  const text = await response.text()

  let envelope: ApiResponse<T> | null = null
  if (text) {
    try {
      envelope = JSON.parse(text) as ApiResponse<T>
    } catch {
      envelope = null
    }
  }

  if (envelope && typeof envelope.code === 'number') {
    const requestId = envelope.request_id ?? headerRequestId
    if (envelope.code === 0) {
      if (!response.ok && !options.allowNonOkEnvelope) {
        throw new ApiError(
          response.status,
          envelope.message || `HTTP ${response.status}`,
          envelope.data,
          requestId,
        )
      }
      if (requestId) {
        console.debug(`[api] ${options.method ?? 'GET'} ${url} request_id=${requestId}`)
      }
      return envelope.data
    }
    throw new ApiError(envelope.code, envelope.message || '请求失败', envelope.data, requestId)
  }

  if (!response.ok) {
    throw new ApiError(
      response.status,
      text || response.statusText || `HTTP ${response.status}`,
      null,
      headerRequestId,
    )
  }
  throw new ApiError(-2, '响应格式错误：缺少统一 envelope', text, headerRequestId)
}

export const api = {
  get<T>(path: string, options: Omit<RequestOptions, 'method' | 'body' | 'formData'> = {}) {
    return request<T>(path, { ...options, method: 'GET' })
  },
  post<T>(
    path: string,
    body?: unknown,
    options: Omit<RequestOptions, 'method' | 'body' | 'formData'> = {},
  ) {
    return request<T>(path, { ...options, method: 'POST', body })
  },
}

/** GET 二进制文件（/files/...）；错误时按 envelope 抛 ApiError。 */
export async function requestBlob(path: string): Promise<Blob> {
  const url = `${BASE_URL}${path.startsWith('/') ? path : `/${path}`}`
  let response: Response
  try {
    response = await fetch(url)
  } catch (err) {
    throw new ApiError(-1, '网络错误', err)
  }
  if (response.ok) {
    return response.blob()
  }
  const text = await response.text()
  let envelope: ApiResponse<unknown> | null = null
  if (text) {
    try {
      envelope = JSON.parse(text) as ApiResponse<unknown>
    } catch {
      envelope = null
    }
  }
  if (envelope && typeof envelope.code === 'number') {
    throw new ApiError(
      envelope.code,
      envelope.message || `HTTP ${response.status}`,
      envelope.data,
      envelope.request_id,
    )
  }
  throw new ApiError(
    response.status,
    text || response.statusText || `HTTP ${response.status}`,
  )
}
