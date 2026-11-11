import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import { getPaperStatus, retryPaper, reviewPaper, uploadPaper } from '../api/documents'
import type {
  Card as CardRecord,
  DFTParameters,
  DocumentStatus,
  HubbardU,
  PaperStatus,
  ReviewCandidate,
  ReviewDecision,
  ReviewRequest,
  SurfaceModel,
} from '../api/types'

const { Dragger } = Upload

const POLL_INTERVAL_MS = 5_000
const POLL_TIMEOUT_MS = 900_000

const STATUS_COLORS: Record<PaperStatus, string> = {
  queued: 'blue',
  parsing: 'blue',
  extracting: 'blue',
  validating: 'blue',
  needs_review: 'orange',
  saved: 'green',
  failed: 'red',
  method_not_found: 'red',
  rejected: 'red',
}

const SEVERITY_COLORS: Record<string, string> = {
  error: 'red',
  warning: 'orange',
  info: 'blue',
}

interface EditValues {
  encut_ev?: number | null
  mesh?: [number, number, number]
  hubbard_u?: HubbardU[]
  comment?: string
}

function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err
  return new ApiError(-1, err instanceof Error ? err.message : '未知错误', err)
}

function formatKpoints(kp: DFTParameters['kpoints']): string {
  if (!kp) return '未报告'
  if (kp.mesh) return `${kp.scheme} ${kp.mesh.join('×')}`
  if (kp.spacing) return `${kp.scheme} spacing ${kp.spacing}`
  return kp.scheme
}

function formatHubbardU(us: HubbardU[]): string {
  if (!us.length) return '无'
  return us.map((u) => `${u.element} ${u.orbital} ${u.u_ev} eV`).join('；')
}

function decisionTag(decision?: ReviewDecision) {
  if (!decision) return <Tag>待处理</Tag>
  if (decision.decision === 'approve') return <Tag color="green">已通过</Tag>
  if (decision.decision === 'edit') return <Tag color="blue">已修改</Tag>
  return <Tag color="red">已拒绝</Tag>
}

function formatSurface(surface: SurfaceModel | null): string {
  if (!surface) return '—'
  const layers = surface.layers != null ? `${surface.layers} 层` : '层数未报告'
  return `(${surface.miller.join('')}) · ${layers} · 真空 ${
    surface.vacuum_angstrom ?? '未报告'
  } Å · 终止面 ${surface.termination ?? '未指定'}`
}

export default function DocumentsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const queryPaperId = searchParams.get('paper_id')

  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [paperId, setPaperId] = useState<string | null>(null)
  const [doc, setDoc] = useState<DocumentStatus | null>(null)
  const [polling, setPolling] = useState(false)
  const [timedOut, setTimedOut] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  const [reviewRequest, setReviewRequest] = useState<ReviewRequest | null>(null)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [decisions, setDecisions] = useState<Record<number, ReviewDecision>>({})
  const [submitting, setSubmitting] = useState(false)
  const [reviewError, setReviewError] = useState<ApiError | null>(null)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [rejectTarget, setRejectTarget] = useState<number | null>(null)
  const [rejectComment, setRejectComment] = useState('')

  const pollRef = useRef<number | null>(null)
  const pollStartRef = useRef(0)

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
    setPolling(false)
  }, [])

  const applyStatus = useCallback(
    (status: DocumentStatus) => {
      setDoc(status)
      if (status.status === 'needs_review' && status.review_request) {
        setReviewRequest(status.review_request)
        setReviewOpen(true)
        setDecisions({})
        setEditingIndex(null)
        stopPolling()
        return
      }
      if (['saved', 'failed', 'method_not_found', 'rejected'].includes(status.status)) {
        stopPolling()
        if (status.status !== 'needs_review') {
          setReviewOpen(false)
          setReviewRequest(null)
        }
      }
      // needs_review 但 review_request 尚未落 checkpoint：继续轮询，避免弹空 Modal
    },
    [stopPolling],
  )

  const pollOnce = useCallback(
    async (id: string) => {
      try {
        const status = await getPaperStatus(id)
        applyStatus(status)
      } catch (err) {
        setError(toApiError(err))
        stopPolling()
      }
    },
    [applyStatus, stopPolling],
  )

  const startPolling = useCallback(
    (id: string) => {
      stopPolling()
      setTimedOut(false)
      setPolling(true)
      pollStartRef.current = Date.now()
      void pollOnce(id)
      pollRef.current = window.setInterval(() => {
        if (Date.now() - pollStartRef.current > POLL_TIMEOUT_MS) {
          stopPolling()
          setTimedOut(true)
          return
        }
        void pollOnce(id)
      }, POLL_INTERVAL_MS)
    },
    [pollOnce, stopPolling],
  )

  useEffect(
    () => () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current)
    },
    [],
  )

  // 支持 /documents?paper_id=... 直接加载（刷新恢复 / 分享链接）
  useEffect(() => {
    if (queryPaperId && queryPaperId !== paperId) {
      setPaperId(queryPaperId)
      setError(null)
      setTimedOut(false)
      startPolling(queryPaperId)
    }
  }, [queryPaperId, paperId, startPolling])

  const handleUpload = async () => {
    if (!file) {
      message.warning('请先选择一个 PDF 文件')
      return
    }
    stopPolling()
    setUploading(true)
    setError(null)
    setTimedOut(false)
    setDoc(null)
    setPaperId(null)
    setReviewRequest(null)
    setReviewOpen(false)
    setDecisions({})
    setSearchParams({}, { replace: true })
    try {
      const resp = await uploadPaper(file)
      setPaperId(resp.paper_id)
      setSearchParams({ paper_id: resp.paper_id }, { replace: true })
      message.success(resp.status === 'saved' ? '论文已存在，加载现有状态' : '上传成功，开始处理')
      startPolling(resp.paper_id)
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setUploading(false)
    }
  }

  const handleRetry = async () => {
    if (!paperId) return
    setError(null)
    setTimedOut(false)
    try {
      await retryPaper(paperId)
      message.info('已重新入队')
      startPolling(paperId)
    } catch (err) {
      setError(toApiError(err))
    }
  }

  const handleReviewSubmit = async (finalDecisions: ReviewDecision[]) => {
    if (!paperId) return
    setSubmitting(true)
    setReviewError(null)
    try {
      const resp = await reviewPaper(paperId, finalDecisions)
      if (resp.type === 'cards') {
        setReviewOpen(false)
        setReviewRequest(null)
        setDecisions({})
        setEditingIndex(null)
        message.success(`审核完成，共 ${resp.cards.length} 张卡片`)
        try {
          const latest = await getPaperStatus(paperId)
          applyStatus(latest)
        } catch (err) {
          setError(toApiError(err))
        }
        return
      }
      if (resp.type === 'review_request') {
        setReviewRequest(resp)
        setDecisions({})
        setEditingIndex(null)
        message.info('仍有未处理的候选，请继续审核')
        return
      }
      // document_status 兼容分支
      applyStatus(resp)
      if (resp.status !== 'needs_review') {
        setReviewOpen(false)
        setReviewRequest(null)
      }
    } catch (err) {
      setReviewError(toApiError(err))
    } finally {
      setSubmitting(false)
    }
  }

  const submitReview = () => {
    if (!reviewRequest) return
    const pending = reviewRequest.candidates.filter((c) => !decisions[c.index])
    if (pending.length > 0) {
      Modal.confirm({
        title: '有未处理的候选',
        content: `还有 ${pending.length} 个候选未处理，将默认标记为 reject。是否继续提交？`,
        okText: '继续提交',
        cancelText: '返回处理',
        onOk: () => {
          const filled: Record<number, ReviewDecision> = { ...decisions }
          for (const c of pending) {
            filled[c.index] = {
              index: c.index,
              decision: 'reject',
              comment: '未处理，默认拒绝',
            }
          }
          setDecisions(filled)
          void handleReviewSubmit(Object.values(filled))
        },
      })
      return
    }
    void handleReviewSubmit(Object.values(decisions))
  }

  const saveEdit = (candidate: ReviewCandidate, values: EditValues) => {
    const parameters: Partial<DFTParameters> = {}
    if (values.encut_ev !== undefined && values.encut_ev !== null) {
      parameters.encut_ev = values.encut_ev
    }
    if (candidate.parameters.kpoints && values.mesh) {
      parameters.kpoints = { ...candidate.parameters.kpoints, mesh: values.mesh }
    }
    if (values.hubbard_u) {
      parameters.hubbard_u = values.hubbard_u
    }
    setDecisions((prev) => ({
      ...prev,
      [candidate.index]: {
        index: candidate.index,
        decision: 'edit',
        parameters,
        comment: values.comment || undefined,
      },
    }))
    setEditingIndex(null)
  }

  const confirmReject = () => {
    if (rejectTarget === null) return
    setDecisions((prev) => ({
      ...prev,
      [rejectTarget]: {
        index: rejectTarget,
        decision: 'reject',
        comment: rejectComment || undefined,
      },
    }))
    setRejectTarget(null)
    setRejectComment('')
  }

  const columns: ColumnsType<CardRecord> = [
    { title: '体系', dataIndex: 'system_normalized', key: 'system_normalized' },
    { title: '计算类型', dataIndex: 'calculation_type', key: 'calculation_type' },
    { title: '泛函', dataIndex: 'functional', key: 'functional' },
    {
      title: '缺失字段',
      dataIndex: 'missing_fields',
      key: 'missing_fields',
      render: (value: string[]) =>
        value.length ? value.map((f) => <Tag key={f}>{f}</Tag>) : '—',
    },
    {
      title: '审核状态',
      dataIndex: 'review_status',
      key: 'review_status',
      render: (value: string) => (
        <Tag color={value === 'reviewed' ? 'green' : 'blue'}>{value}</Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (value?: string) => (value ? new Date(value).toLocaleString() : '—'),
    },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="上传论文">
        <Dragger
          accept=".pdf"
          maxCount={1}
          beforeUpload={(f) => {
            setFile(f)
            return false
          }}
          onChange={({ fileList }) => {
            if (fileList.length === 0) setFile(null)
          }}
          disabled={uploading || polling}
        >
          <p className="ant-upload-text">点击或拖拽 PDF 到此处</p>
          <p className="ant-upload-hint">仅支持文本型 PDF（后端上限 50MB）</p>
        </Dragger>
        <Button
          type="primary"
          onClick={() => void handleUpload()}
          loading={uploading}
          disabled={!file}
          style={{ marginTop: 12 }}
        >
          开始上传
        </Button>
      </Card>

      {error && (
        <Alert
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          message={`请求失败（code=${error.code}）`}
          description={error.message}
        />
      )}

      <Card title="文献处理">
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="状态">
            {doc ? (
              <Tag color={STATUS_COLORS[doc.status]}>{doc.status}</Tag>
            ) : paperId ? (
              <Tag color="blue">查询中</Tag>
            ) : (
              <Tag>未上传</Tag>
            )}
          </Descriptions.Item>
          {paperId && (
            <Descriptions.Item label="paper_id">
              <Typography.Text copyable>{paperId}</Typography.Text>
            </Descriptions.Item>
          )}
          {file && <Descriptions.Item label="文件名">{file.name}</Descriptions.Item>}
          {doc && (
            <Descriptions.Item label="pending_action">
              {doc.pending_action ?? '—'}
            </Descriptions.Item>
          )}
          {doc && (
            <Descriptions.Item label="card_ids">
              {doc.card_ids.length ? doc.card_ids.join(', ') : '—'}
            </Descriptions.Item>
          )}
          {doc?.error_msg && (
            <Descriptions.Item label="错误">
              {doc.error_code} {doc.error_msg}
            </Descriptions.Item>
          )}
        </Descriptions>

        {polling && (
          <div style={{ marginTop: 12 }}>
            <Spin size="small" /> <Typography.Text type="secondary">正在轮询处理状态…</Typography.Text>
          </div>
        )}
        {timedOut && (
          <Alert
            style={{ marginTop: 12 }}
            type="warning"
            showIcon
            message="轮询超时"
            description="可手动刷新或检查后端日志"
            action={
              <Button size="small" onClick={() => paperId && startPolling(paperId)}>
                重新轮询
              </Button>
            }
          />
        )}
        {doc && ['failed', 'method_not_found'].includes(doc.status) && (
          <Button onClick={() => void handleRetry()} style={{ marginTop: 12 }}>
            重试
          </Button>
        )}
        {doc?.status === 'needs_review' && !reviewOpen && (
          <Button type="primary" style={{ marginTop: 12 }} onClick={() => setReviewOpen(true)}>
            打开审核
          </Button>
        )}
      </Card>

      <Card title="参数卡片">
        <Table<CardRecord>
          rowKey="card_id"
          columns={columns}
          dataSource={doc?.cards ?? []}
          pagination={false}
          locale={{ emptyText: '暂无卡片' }}
        />
      </Card>

      <Modal
        title={reviewRequest ? `人工审核 · 第 ${reviewRequest.round} 轮` : '人工审核'}
        open={reviewOpen}
        width={900}
        styles={{ body: { maxHeight: '65vh', overflowY: 'auto' } }}
        onCancel={() => setReviewOpen(false)}
        footer={
          <Space>
            <Button onClick={() => setReviewOpen(false)}>稍后处理</Button>
            <Button type="primary" loading={submitting} onClick={submitReview}>
              提交审核
            </Button>
          </Space>
        }
      >
        {reviewError && (
          <Alert
            style={{ marginBottom: 12 }}
            type="error"
            showIcon
            message={`审核提交失败（code=${reviewError.code}）`}
            description={reviewError.message}
          />
        )}
        <List
          dataSource={reviewRequest?.candidates ?? []}
          locale={{ emptyText: '暂无审核候选' }}
          renderItem={(candidate) => (
            <List.Item key={candidate.index}>
              <Card
                style={{ width: '100%' }}
                title={`${candidate.system_formula} · ${candidate.parameters.calculation_type} · ${candidate.parameters.functional}`}
                extra={decisionTag(decisions[candidate.index])}
              >
                <Descriptions size="small" column={1} bordered>
                  <Descriptions.Item label="ENCUT">
                    {candidate.parameters.encut_ev != null
                      ? `${candidate.parameters.encut_ev} eV`
                      : '未报告'}
                  </Descriptions.Item>
                  <Descriptions.Item label="k 点">
                    {formatKpoints(candidate.parameters.kpoints)}
                  </Descriptions.Item>
                  <Descriptions.Item label="Hubbard U">
                    {formatHubbardU(candidate.parameters.hubbard_u)}
                  </Descriptions.Item>
                  {candidate.surface && (
                    <Descriptions.Item label="表面">
                      {formatSurface(candidate.surface)}
                    </Descriptions.Item>
                  )}
                  <Descriptions.Item label="缺失字段">
                    {candidate.missing_fields.length
                      ? candidate.missing_fields.map((f) => <Tag key={f}>{f}</Tag>)
                      : '—'}
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 8 }}>
                  <Typography.Text strong>校验结果：</Typography.Text>
                  {candidate.findings.length === 0 ? (
                    <Typography.Text type="secondary"> 无</Typography.Text>
                  ) : (
                    <Space direction="vertical" size={4} style={{ marginTop: 4 }}>
                      {candidate.findings.map((f, i) => (
                        <div key={`${f.rule_id}-${i}`}>
                          <Tag color={SEVERITY_COLORS[f.severity]}>{f.severity}</Tag>
                          <Typography.Text strong>{f.rule_id}</Typography.Text> {f.message}
                          {f.suggestion && (
                            <Typography.Text type="secondary">
                              （建议：{f.suggestion}）
                            </Typography.Text>
                          )}
                        </div>
                      ))}
                    </Space>
                  )}
                </div>

                {candidate.evidence.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <Typography.Text strong>证据：</Typography.Text>
                    <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                      {candidate.evidence.slice(0, 3).map((e, i) => (
                        <li key={i}>
                          <Typography.Text type="secondary">
                            {e.quote}
                            {e.page ? `（第 ${e.page} 页）` : ''}
                          </Typography.Text>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <Space style={{ marginTop: 12 }}>
                  <Button
                    type={decisions[candidate.index]?.decision === 'approve' ? 'primary' : 'default'}
                    onClick={() =>
                      setDecisions((prev) => ({
                        ...prev,
                        [candidate.index]: { index: candidate.index, decision: 'approve' },
                      }))
                    }
                  >
                    通过
                  </Button>
                  <Button
                    onClick={() =>
                      setEditingIndex(editingIndex === candidate.index ? null : candidate.index)
                    }
                  >
                    编辑
                  </Button>
                  <Button
                    danger
                    onClick={() => {
                      setRejectTarget(candidate.index)
                      setRejectComment('')
                    }}
                  >
                    拒绝
                  </Button>
                </Space>

                {editingIndex === candidate.index && (
                  <Card type="inner" title="修改参数（仅 INCAR 级字段）" style={{ marginTop: 12 }}>
                    <Form
                      layout="vertical"
                      initialValues={{
                        encut_ev: candidate.parameters.encut_ev ?? undefined,
                        mesh: candidate.parameters.kpoints?.mesh ?? [1, 1, 1],
                        hubbard_u: candidate.parameters.hubbard_u.map((u) => ({
                          element: u.element,
                          orbital: u.orbital,
                          u_ev: u.u_ev,
                        })),
                        comment: '',
                      }}
                      onFinish={(values) => saveEdit(candidate, values as EditValues)}
                    >
                      <Form.Item label="ENCUT (eV)" name="encut_ev">
                        <InputNumber min={50} max={1000} style={{ width: 180 }} />
                      </Form.Item>
                      {candidate.parameters.kpoints ? (
                        <Form.Item label="k 点网格">
                          <Space>
                            <Form.Item
                              name={['mesh', 0]}
                              noStyle
                              rules={[{ required: true, message: 'k 点' }]}
                            >
                              <InputNumber min={1} max={99} />
                            </Form.Item>
                            <Form.Item
                              name={['mesh', 1]}
                              noStyle
                              rules={[{ required: true, message: 'k 点' }]}
                            >
                              <InputNumber min={1} max={99} />
                            </Form.Item>
                            <Form.Item
                              name={['mesh', 2]}
                              noStyle
                              rules={[{ required: true, message: 'k 点' }]}
                            >
                              <InputNumber min={1} max={99} />
                            </Form.Item>
                          </Space>
                        </Form.Item>
                      ) : (
                        <Alert
                          style={{ marginBottom: 12 }}
                          type="info"
                          showIcon
                          message="论文未给出 k 点，审核阶段不允许新增网格"
                        />
                      )}
                      <Form.List name="hubbard_u">
                        {(fields, { add, remove }) => (
                          <Form.Item label="Hubbard U">
                            <Space direction="vertical" style={{ width: '100%' }}>
                              {fields.map(({ key, name, ...rest }) => (
                                <Space key={key} align="baseline">
                                  <Form.Item
                                    {...rest}
                                    name={[name, 'element']}
                                    rules={[{ required: true, message: '元素' }]}
                                  >
                                    <Input placeholder="Co" style={{ width: 90 }} />
                                  </Form.Item>
                                  <Form.Item
                                    {...rest}
                                    name={[name, 'orbital']}
                                    rules={[{ required: true, message: '轨道' }]}
                                  >
                                    <Select
                                      style={{ width: 70 }}
                                      options={[
                                        { value: 's', label: 's' },
                                        { value: 'p', label: 'p' },
                                        { value: 'd', label: 'd' },
                                        { value: 'f', label: 'f' },
                                      ]}
                                    />
                                  </Form.Item>
                                  <Form.Item
                                    {...rest}
                                    name={[name, 'u_ev']}
                                    rules={[{ required: true, message: 'U 值' }]}
                                  >
                                    <InputNumber min={0} max={20} addonAfter="eV" />
                                  </Form.Item>
                                  <Button danger type="link" onClick={() => remove(name)}>
                                    删除
                                  </Button>
                                </Space>
                              ))}
                              <Button
                                type="dashed"
                                onClick={() => add({ element: 'Co', orbital: 'd', u_ev: 0 })}
                              >
                                添加 U 值
                              </Button>
                            </Space>
                          </Form.Item>
                        )}
                      </Form.List>
                      <Form.Item label="备注" name="comment">
                        <Input.TextArea rows={2} placeholder="修改原因（可选）" />
                      </Form.Item>
                      <Space>
                        <Button type="primary" htmlType="submit">
                          保存修改
                        </Button>
                        <Button onClick={() => setEditingIndex(null)}>取消</Button>
                      </Space>
                    </Form>
                  </Card>
                )}
              </Card>
            </List.Item>
          )}
        />
      </Modal>

      <Modal
        title="拒绝候选"
        open={rejectTarget !== null}
        okText="确认拒绝"
        cancelText="取消"
        onOk={confirmReject}
        onCancel={() => setRejectTarget(null)}
      >
        <Input.TextArea
          rows={3}
          placeholder="拒绝原因（可选）"
          value={rejectComment}
          onChange={(e) => setRejectComment(e.target.value)}
        />
      </Modal>
    </Space>
  )
}
