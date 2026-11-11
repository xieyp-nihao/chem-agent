import {
  Alert,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  List,
  Select,
  Space,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { ApiError } from '../api/client'
import { confirmCalc, downloadFile, draftCalc, type ConfirmPayload } from '../api/calc'
import type {
  CalcSlots,
  ConfirmRequest,
  ConfirmResponse,
  HubbardU,
  KPoints,
  OutputFile,
} from '../api/types'

interface FormValues {
  calculation_type: 'relax' | 'static'
  functional: string
  formula?: string
  mp_id?: string
  include_surface: boolean
  miller?: [number, number, number]
  layers?: number
  vacuum_angstrom?: number
  encut_ev?: number
  mesh?: [number, number, number]
  hubbard_u?: HubbardU[]
}

interface PatchValues {
  encut_ev?: number
  mesh?: [number, number, number]
  hubbard_u?: HubbardU[]
}

const SEVERITY_COLORS: Record<string, string> = {
  error: 'red',
  warning: 'orange',
  info: 'blue',
}

const PRE_STYLE: CSSProperties = {
  maxHeight: 360,
  overflow: 'auto',
  background: '#fafafa',
  padding: 12,
  border: '1px solid #f0f0f0',
  borderRadius: 4,
  fontSize: 12,
  margin: 0,
}

function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err
  return new ApiError(-1, err instanceof Error ? err.message : '未知错误', err)
}

function functionalFamily(functional: string): CalcSlots['functional_family'] {
  if (functional === 'PBE') return 'PBE'
  if (functional === 'PBE+U') return 'PBE+U'
  return 'other'
}

export default function CalcPage() {
  const [form] = Form.useForm<FormValues>()
  const [patchForm] = Form.useForm<PatchValues>()

  const functional = (Form.useWatch('functional', form) as string | undefined) ?? 'PBE+U'
  const includeSurface =
    (Form.useWatch('include_surface', form) as boolean | undefined) ?? true

  const [draft, setDraft] = useState<ConfirmRequest | null>(null)
  const [submittedSlots, setSubmittedSlots] = useState<CalcSlots | null>(null)
  const [outputFiles, setOutputFiles] = useState<OutputFile[] | null>(null)
  const [loadingDraft, setLoadingDraft] = useState(false)
  const [loadingConfirm, setLoadingConfirm] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [activeTab, setActiveTab] = useState('INCAR')

  // 草案变化或 patch 提交后，用 submittedSlots 预填 patch Form
  useEffect(() => {
    if (!submittedSlots) return
    patchForm.setFieldsValue({
      encut_ev: submittedSlots.encut_ev ?? undefined,
      mesh: submittedSlots.kpoints?.mesh ?? [1, 1, 1],
      hubbard_u: submittedSlots.hubbard_u ?? [],
    })
  }, [submittedSlots, patchForm])

  const handleGenerate = async (values: FormValues) => {
    setError(null)
    if (!values.formula && !values.mp_id) {
      message.warning('化学式与 mp_id 至少填写一个')
      return
    }
    const family = functionalFamily(values.functional)
    const hubbardU = family === 'PBE+U' ? values.hubbard_u ?? [] : []
    if (family === 'PBE+U' && hubbardU.length === 0) {
      message.warning('PBE+U 至少需要一条 Hubbard U 值')
      return
    }
    const slots: CalcSlots = {
      calculation_type: values.calculation_type,
      functional: values.functional,
      functional_family: family,
      formula: values.formula || undefined,
      mp_id: values.mp_id || undefined,
      surface: values.include_surface
        ? {
            miller: values.miller as [number, number, number],
            layers: values.layers as number,
            vacuum_angstrom: values.vacuum_angstrom as number,
          }
        : undefined,
      encut_ev: values.encut_ev ?? undefined,
      kpoints: values.mesh
        ? { scheme: 'gamma', mesh: values.mesh }
        : undefined,
      hubbard_u: hubbardU,
    }

    setLoadingDraft(true)
    try {
      const resp = await draftCalc(slots)
      setDraft(resp)
      setSubmittedSlots(slots)
      setOutputFiles(null)
      setActiveTab('INCAR')
      message.success(`草案已生成（第 ${resp.round} 轮）`)
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setLoadingDraft(false)
    }
  }

  const applyResponse = (resp: ConfirmResponse) => {
    if (resp.type === 'confirm_request') {
      setDraft(resp)
      setActiveTab('INCAR')
      message.info(`仍有待处理项，进入第 ${resp.round} 轮`)
      return
    }
    if (resp.type === 'output_files') {
      setOutputFiles(resp.files)
      setDraft(null)
      setSubmittedSlots(null)
      form.resetFields()
      patchForm.resetFields()
      message.success(`已生成 ${resp.files.length} 个文件`)
      return
    }
    // ack
    setOutputFiles(null)
    setDraft(null)
    setSubmittedSlots(null)
    form.resetFields()
    patchForm.resetFields()
    message.info(resp.message || '已放弃草案')
  }

  const handleApprove = async () => {
    if (!draft) return
    setLoadingConfirm(true)
    setError(null)
    try {
      const resp = await confirmCalc(draft.session_id, { decision: 'approve' })
      applyResponse(resp)
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setLoadingConfirm(false)
    }
  }

  const handleAbort = async () => {
    if (!draft) return
    setLoadingConfirm(true)
    setError(null)
    try {
      const resp = await confirmCalc(draft.session_id, { decision: 'abort' })
      applyResponse(resp)
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setLoadingConfirm(false)
    }
  }

  const handlePatch = async (values: PatchValues) => {
    if (!draft || !submittedSlots) return
    setLoadingConfirm(true)
    setError(null)
    try {
      const patch: NonNullable<ConfirmPayload['patch']> = {}
      if (values.encut_ev !== undefined && values.encut_ev !== null) {
        patch.encut_ev = values.encut_ev
      }
      if (values.mesh) {
        const base: KPoints =
          submittedSlots.kpoints ?? { scheme: 'gamma', mesh: values.mesh }
        patch.kpoints = { ...base, mesh: values.mesh }
      }
      if (values.hubbard_u) {
        patch.hubbard_u = values.hubbard_u
      }

      const resp = await confirmCalc(draft.session_id, { decision: 'patch', patch })
      if (resp.type === 'confirm_request') {
        setDraft(resp)
        const next: CalcSlots = { ...submittedSlots }
        if (patch.encut_ev !== undefined) next.encut_ev = patch.encut_ev
        if (patch.kpoints) next.kpoints = patch.kpoints
        if (patch.hubbard_u) next.hubbard_u = patch.hubbard_u
        setSubmittedSlots(next)
        setActiveTab('INCAR')
        message.info(`仍有待处理项，进入第 ${resp.round} 轮`)
      } else {
        applyResponse(resp)
      }
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setLoadingConfirm(false)
    }
  }

  const handleDownload = async (file: OutputFile) => {
    setError(null)
    try {
      const blob = await downloadFile(file.url)
      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = file.name
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
    } catch (err) {
      setError(toApiError(err))
    }
  }

  const errData = (error?.data ?? null) as
    | { missing_slots?: string[]; suggestions?: string[] }
    | null

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {error && (
        <Alert
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          message={`请求失败（code=${error.code}）`}
          description={
            <Space direction="vertical" size={2}>
              <span>{error.message}</span>
              {error.code === 4003 && (
                <span>当前 session 有未完成的草案，请先确认或放弃。</span>
              )}
              {errData?.missing_slots && errData.missing_slots.length > 0 && (
                <span>缺少槽位：{errData.missing_slots.join('、')}</span>
              )}
              {errData?.suggestions && errData.suggestions.length > 0 && (
                <span>建议：{errData.suggestions.join('；')}</span>
              )}
            </Space>
          }
        />
      )}

      <Card title="计算参数">
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            calculation_type: 'relax',
            functional: 'PBE+U',
            formula: 'Co3O4',
            mp_id: 'mp-18748',
            include_surface: true,
            miller: [1, 1, 1],
            layers: 6,
            vacuum_angstrom: 15,
            encut_ev: 520,
            mesh: [3, 3, 1],
            hubbard_u: [{ element: 'Co', orbital: 'd', u_ev: 3.5 }],
          }}
          onFinish={(values) => void handleGenerate(values)}
        >
          <Divider orientation="left">基础</Divider>
          <Space wrap size="middle">
            <Form.Item
              label="计算类型"
              name="calculation_type"
              rules={[{ required: true, message: '请选择计算类型' }]}
            >
              <Select
                style={{ width: 180 }}
                options={[
                  { value: 'relax', label: 'relax（结构弛豫）' },
                  { value: 'static', label: 'static（单点）' },
                ]}
              />
            </Form.Item>
            <Form.Item
              label="泛函"
              name="functional"
              rules={[{ required: true, message: '请选择泛函' }]}
            >
              <Select
                style={{ width: 260 }}
                options={[
                  { value: 'PBE', label: 'PBE' },
                  { value: 'PBE+U', label: 'PBE+U' },
                  { value: 'HSE06', label: 'HSE06（第一版不支持，用于验证 3002）' },
                ]}
              />
            </Form.Item>
            <Form.Item
              label="化学式"
              name="formula"
              rules={[
                {
                  validator: (_, value) =>
                    value || form.getFieldValue('mp_id')
                      ? Promise.resolve()
                      : Promise.reject(new Error('化学式与 mp_id 至少填一个')),
                },
              ]}
            >
              <Input style={{ width: 160 }} placeholder="Co3O4" />
            </Form.Item>
            <Form.Item
              label="mp_id"
              name="mp_id"
              rules={[
                {
                  validator: (_, value) =>
                    value || form.getFieldValue('formula')
                      ? Promise.resolve()
                      : Promise.reject(new Error('化学式与 mp_id 至少填一个')),
                },
              ]}
            >
              <Input style={{ width: 160 }} placeholder="mp-18748" />
            </Form.Item>
          </Space>

          <Divider orientation="left">表面（可选）</Divider>
          <Form.Item name="include_surface" valuePropName="checked">
            <Checkbox>包含表面（slab）</Checkbox>
          </Form.Item>
          {includeSurface && (
            <Space wrap size="middle">
              <Form.Item label="Miller 指数" required>
                <Space>
                  <Form.Item
                    name={['miller', 0]}
                    noStyle
                    rules={[{ required: true, message: 'h' }]}
                  >
                    <InputNumber placeholder="h" />
                  </Form.Item>
                  <Form.Item
                    name={['miller', 1]}
                    noStyle
                    rules={[{ required: true, message: 'k' }]}
                  >
                    <InputNumber placeholder="k" />
                  </Form.Item>
                  <Form.Item
                    name={['miller', 2]}
                    noStyle
                    rules={[{ required: true, message: 'l' }]}
                  >
                    <InputNumber placeholder="l" />
                  </Form.Item>
                </Space>
              </Form.Item>
              <Form.Item
                label="层数"
                name="layers"
                rules={[{ required: true, message: '请填写层数' }]}
              >
                <InputNumber min={1} max={30} style={{ width: 110 }} />
              </Form.Item>
              <Form.Item
                label="真空层 (Å)"
                name="vacuum_angstrom"
                rules={[{ required: true, message: '请填写真空层' }]}
              >
                <InputNumber min={0} max={100} style={{ width: 130 }} />
              </Form.Item>
            </Space>
          )}

          <Divider orientation="left">参数</Divider>
          <Space wrap size="middle" align="start">
            <Form.Item label="ENCUT (eV)" name="encut_ev">
              <InputNumber min={50} max={1000} style={{ width: 140 }} />
            </Form.Item>
            <Form.Item label="k 点网格" required>
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
          </Space>

          {functional === 'PBE+U' && (
            <Form.List
              name="hubbard_u"
              rules={[
                {
                  validator: async (_, value) => {
                    if (!value || value.length === 0) {
                      throw new Error('PBE+U 至少需要一条 Hubbard U 值')
                    }
                  },
                },
              ]}
            >
              {(fields, { add, remove }, { errors }) => (
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
                      onClick={() => add({ element: 'Co', orbital: 'd', u_ev: 3.5 })}
                    >
                      添加 U 值
                    </Button>
                    <Form.ErrorList errors={errors} />
                  </Space>
                </Form.Item>
              )}
            </Form.List>
          )}

          <Button type="primary" htmlType="submit" loading={loadingDraft}>
            生成草案
          </Button>
        </Form>
      </Card>

      {draft && (
        <Card
          title={`草案预览（第 ${draft.round} 轮）`}
          extra={
            <Typography.Text type="secondary" copyable>
              session: {draft.session_id}
            </Typography.Text>
          }
        >
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: 'INCAR',
                label: 'INCAR',
                forceRender: true,
                children: (
                  <>
                    <pre style={PRE_STYLE}>{draft.draft.incar}</pre>
                    <Divider orientation="left">修改 INCAR 级参数（patch）</Divider>
                    <Form
                      form={patchForm}
                      layout="vertical"
                      onFinish={(values) => void handlePatch(values)}
                    >
                      <Space wrap size="middle" align="start">
                        <Form.Item label="ENCUT (eV)" name="encut_ev">
                          <InputNumber min={50} max={1000} style={{ width: 140 }} />
                        </Form.Item>
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
                      </Space>
                      {submittedSlots?.functional_family === 'PBE+U' && (
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
                                  onClick={() =>
                                    add({ element: 'Co', orbital: 'd', u_ev: 3.5 })
                                  }
                                >
                                  添加 U 值
                                </Button>
                              </Space>
                            </Form.Item>
                          )}
                        </Form.List>
                      )}
                    </Form>
                  </>
                ),
              },
              {
                key: 'POSCAR',
                label: 'POSCAR',
                children: <pre style={PRE_STYLE}>{draft.draft.poscar}</pre>,
              },
              {
                key: 'KPOINTS',
                label: 'KPOINTS',
                children: <pre style={PRE_STYLE}>{draft.draft.kpoints}</pre>,
              },
            ]}
          />

          <Divider orientation="left">校验结果</Divider>
          {draft.validation.findings.length === 0 ? (
            <Typography.Text type="secondary">无</Typography.Text>
          ) : (
            <Space direction="vertical" size={4}>
              {draft.validation.findings.map((f, i) => (
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
          {draft.validation.missing_fields.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <Typography.Text strong>缺失字段：</Typography.Text>
              {draft.validation.missing_fields.map((f) => (
                <Tag key={f}>{f}</Tag>
              ))}
            </div>
          )}
          {draft.draft.warnings && draft.draft.warnings.length > 0 && (
            <Alert
              style={{ marginTop: 12 }}
              type="warning"
              showIcon
              message="生成器警告"
              description={
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {draft.draft.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              }
            />
          )}

          <Divider orientation="left">POTCAR 顺序</Divider>
          {draft.draft.potcar_order.map((p) => (
            <Tag key={p}>{p}</Tag>
          ))}

          {draft.draft.overrides_applied &&
            Object.keys(draft.draft.overrides_applied).length > 0 && (
              <>
                <Divider orientation="left">参数覆盖来源</Divider>
                <Descriptions size="small" bordered column={2}>
                  {Object.entries(draft.draft.overrides_applied).map(([key, value]) => (
                    <Descriptions.Item key={key} label={key}>
                      {value}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </>
            )}
        </Card>
      )}

      <Card title="确认与输出">
        {draft ? (
          <Space direction="vertical" size="middle">
            <Space wrap>
              <Button type="primary" loading={loadingConfirm} onClick={() => void handleApprove()}>
                确认生成（approve）
              </Button>
              <Button
                loading={loadingConfirm}
                onClick={() => {
                  setActiveTab('INCAR')
                  patchForm.submit()
                }}
              >
                应用修改（patch）
              </Button>
              <Button danger loading={loadingConfirm} onClick={() => void handleAbort()}>
                放弃（abort）
              </Button>
            </Space>
            <Typography.Text type="secondary">
              第 {draft.round} 轮 · patch 只允许 ENCUT / k 点 / Hubbard U；结构级修改请重新生成
            </Typography.Text>
          </Space>
        ) : outputFiles ? (
          <List
            header={<Typography.Text strong>输出文件</Typography.Text>}
            dataSource={outputFiles}
            renderItem={(file) => (
              <List.Item
                actions={[
                  <Button key="download" onClick={() => void handleDownload(file)}>
                    下载
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={file.name}
                  description={`${file.size} bytes`}
                />
              </List.Item>
            )}
          />
        ) : (
          <Typography.Text type="secondary">先生成草案</Typography.Text>
        )}
      </Card>
    </Space>
  )
}
