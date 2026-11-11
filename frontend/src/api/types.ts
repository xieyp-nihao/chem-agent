/** 统一响应 envelope（§7.1）。 */
export interface ApiResponse<T> {
  code: number
  message: string
  data: T
  request_id: string
}

/** 错误响应里的 data（§7.1：null 或 {type:"error", details}）。 */
export interface ErrorData {
  type?: 'error'
  details?: Record<string, unknown>
}

/* ---------- /health（§7.8） ---------- */

export type HealthStatus = 'ok' | 'degraded' | 'down'

export type ComponentName =
  | 'sqlite'
  | 'checkpointer'
  | 'llm'
  | 'mp'
  | 'chroma'
  | 'embedding_model'

export type ComponentStatus = 'ok' | 'unavailable'

export interface HealthData {
  type: 'health'
  status: HealthStatus
  components: Record<ComponentName, ComponentStatus>
}

/* ---------- 文献领域（§3.3 子集，只定义前端需要的字段） ---------- */

export type PaperStatus =
  | 'queued'
  | 'parsing'
  | 'extracting'
  | 'validating'
  | 'needs_review'
  | 'saved'
  | 'method_not_found'
  | 'failed'
  | 'rejected'

export interface Evidence {
  source: 'text' | 'table' | 'caption'
  page?: number | null
  bbox?: [number, number, number, number] | null
  quote: string
  field?: string | null
  confidence?: number | null
}

export interface HubbardU {
  element: string
  orbital: 's' | 'p' | 'd' | 'f'
  u_ev: number
  j_ev?: number
}

export interface KPoints {
  scheme: 'gamma' | 'monkhorst-pack' | 'explicit'
  mesh?: [number, number, number] | null
  spacing?: number | null
  gamma_centered?: boolean | null
  explicit?: Array<[number, number, number]> | null
  raw?: string | null
}

export interface DFTParameters {
  calculation_type: 'relax' | 'static'
  functional: string
  functional_family: 'PBE' | 'PBE+U' | 'other'
  hubbard_u: HubbardU[]
  encut_ev?: number | null
  kpoints?: KPoints | null
  ediff?: number | null
  ediffg?: number | null
  spin_polarized?: boolean | null
  vdw?: string | null
  pseudopotential?: string | null
  extra_incar?: Record<string, string>
  notes?: string | null
  evidence?: Evidence[]
}

export interface SurfaceModel {
  miller: [number, number, number]
  layers: number
  vacuum_angstrom?: number | null
  termination?: string | null
  fixed_layers?: number | null
  dipole_correction?: boolean
  supercell?: [number, number, number]
  slab_thickness_angstrom?: number | null
  evidence?: Evidence[]
}

export interface RuleFinding {
  rule_id: string
  severity: 'error' | 'warning' | 'info'
  field?: string | null
  message: string
  suggestion?: string | null
  context?: Record<string, string>
}

export interface ReviewCandidate {
  index: number
  system_formula: string
  system_normalized: string
  parameters: DFTParameters
  surface: SurfaceModel | null
  findings: RuleFinding[]
  missing_fields: string[]
  evidence: Evidence[]
}

export interface ReviewRequest {
  type: 'review_request'
  paper_id: string
  thread_id: string
  round: number
  candidates: ReviewCandidate[]
}

export interface ReviewDecision {
  index: number
  decision: 'approve' | 'edit' | 'reject'
  parameters?: Partial<DFTParameters>
  surface?: Partial<SurfaceModel>
  comment?: string
}

/* ---------- 文档状态（§7.3/§7.4；含 F2 后端扩展字段） ---------- */

export interface Card {
  card_id: string
  paper_id: string
  system_formula?: string
  system_normalized: string
  calculation_type: string
  functional: string
  functional_family?: string
  missing_fields: string[]
  review_status: 'auto_saved' | 'reviewed'
  created_at?: string
  updated_at?: string
}

export interface DocumentStatus {
  type: 'document_status'
  paper_id: string
  status: PaperStatus
  method_found?: boolean | null
  page_count?: number | null
  error_msg?: string | null
  error_code?: number | null
  card_ids: string[]
  cards?: Card[]
  pending_action?: 'review' | 'confirm' | null
  thread_id?: string
  review_request?: ReviewRequest | null
  created_at: string
  updated_at: string
}

/** 上传响应是 document_status 的窄子集（无 card_ids/created_at）。 */
export interface UploadResponse {
  type: 'document_status'
  paper_id: string
  status: PaperStatus
  status_url?: string
}

/** /chat action=review 的 cards 响应。 */
export interface CardsResponse {
  type: 'cards'
  source: string
  cards: Card[]
}

export type ReviewResponse = CardsResponse | ReviewRequest | DocumentStatus

/* ---------- 计算侧（F3） ---------- */

export interface CalcSurface {
  miller: [number, number, number]
  layers: number
  vacuum_angstrom: number
}

export interface CalcSlots {
  calculation_type: 'relax' | 'static'
  functional: string
  functional_family: 'PBE' | 'PBE+U' | 'other'
  formula?: string
  mp_id?: string
  surface?: CalcSurface
  encut_ev?: number
  kpoints?: KPoints
  hubbard_u?: HubbardU[]
}

export interface DraftFiles {
  incar: string
  poscar: string
  kpoints: string
  potcar_order: string[]
  overrides_applied?: Record<string, string>
  warnings?: string[]
}

export interface ValidationSummary {
  findings: RuleFinding[]
  missing_fields: string[]
  rules_version: string
}

export interface ConfirmRequest {
  type: 'confirm_request'
  session_id: string
  thread_id: string
  round: number
  draft: DraftFiles
  validation: ValidationSummary
  applied_overrides?: Record<string, string>
}

export interface OutputFile {
  name: string
  path: string
  url: string
  size: number
}

export interface OutputFilesResponse {
  type: 'output_files'
  session_id: string
  files: OutputFile[]
}

export interface AckResponse {
  type: 'ack'
  message: string
  session_id?: string
}

export type ConfirmResponse = ConfirmRequest | OutputFilesResponse | AckResponse

/* ---------- 助手页 /chat action=message（F4） ---------- */

export interface ChatMessage {
  message: string
  paper_id?: string
  session_id?: string
}

/** /chat 的 cards 可能是 cache（完整 brief）或 semantic（仅 card_id/score），用宽松类型。 */
export interface ChatCard {
  card_id: string
  paper_id?: string
  system_normalized?: string
  calculation_type?: string
  functional?: string
  missing_fields?: string[]
  review_status?: string
  score?: number
}

export interface CardsData {
  type: 'cards'
  source: string
  cards: ChatCard[]
}

/** action=message 的 document_status 是子集（无 card_ids/created_at）。 */
export interface ChatDocumentStatus {
  type: 'document_status'
  paper_id: string
  status: PaperStatus
  pending_action?: 'review' | 'confirm' | null
  thread_id?: string
  card_ids?: string[]
}

export interface ClarifyData {
  type: 'clarify'
  message?: string
  missing_slots?: string[]
  suggestions?: string[]
}

export interface ChatError {
  type: 'error'
  code: number
  message: string
  data?: unknown
}

export type ChatSubResult =
  | CardsData
  | ConfirmRequest
  | ChatDocumentStatus
  | ClarifyData
  | ChatError

export interface ResultsData {
  type: 'results'
  results: ChatSubResult[]
}

export type ChatResponse = ChatSubResult | ResultsData
