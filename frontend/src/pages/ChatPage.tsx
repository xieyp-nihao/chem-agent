import {
  Alert,
  Button,
  Card,
  Collapse,
  Input,
  List,
  Space,
  Tag,
  Typography,
  message,
} from 'antd'
import { useState } from 'react'
import type { CSSProperties } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError } from '../api/client'
import { sendMessage } from '../api/chat'
import type {
  CardsData,
  ChatDocumentStatus,
  ChatError,
  ChatResponse,
  ChatSubResult,
  ClarifyData,
  ConfirmRequest,
  PaperStatus,
  ResultsData,
} from '../api/types'

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

const TYPE_LABELS: Record<string, string> = {
  cards: '文献卡片',
  confirm_request: '计算草案',
  document_status: '文献状态',
  clarify: '需要补充信息',
  error: '错误',
}

const PRE_STYLE: CSSProperties = {
  maxHeight: 220,
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

function sourceLabel(source: string): string {
  if (source === 'cache') return '精确缓存'
  if (source === 'semantic') return '语义检索'
  return source
}

export default function ChatPage() {
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [paperId, setPaperId] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  const toDocuments = (pid?: string) => {
    if (pid) {
      navigate(`/documents?paper_id=${encodeURIComponent(pid)}`)
    }
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text) {
      message.warning('请输入需求')
      return
    }
    setLoading(true)
    setError(null)
    setResponse(null)
    try {
      const resp = await sendMessage({
        message: text,
        paper_id: paperId.trim() || undefined,
        session_id: sessionId.trim() || undefined,
      })
      setResponse(resp)
      setInput('')
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setLoading(false)
    }
  }

  const renderCards = (data: CardsData) => {
    const firstPaper = data.cards.find((card) => card.paper_id)?.paper_id
    return (
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Typography.Text type="secondary">
          来源：{sourceLabel(data.source)}
        </Typography.Text>
        <List
          dataSource={data.cards}
          locale={{ emptyText: '暂无卡片' }}
          renderItem={(card) => (
            <List.Item
              actions={
                card.paper_id
                  ? [
                      <Button
                        key="view"
                        size="small"
                        onClick={() => card.paper_id && toDocuments(card.paper_id)}
                      >
                        查看
                      </Button>,
                    ]
                  : []
              }
            >
              <List.Item.Meta
                title={card.system_normalized ?? `卡片 ${card.card_id.slice(0, 8)}`}
                description={
                  <Space wrap>
                    {card.calculation_type && <Tag>{card.calculation_type}</Tag>}
                    {card.functional && <Tag>{card.functional}</Tag>}
                    {card.review_status && <Tag>{card.review_status}</Tag>}
                    {card.score !== undefined && (
                      <Tag color="blue">score {card.score.toFixed(3)}</Tag>
                    )}
                    {(card.missing_fields ?? []).map((field) => (
                      <Tag key={field}>{field}</Tag>
                    ))}
                  </Space>
                }
              />
            </List.Item>
          )}
        />
        <Space>
          <Button type="primary" disabled={!firstPaper} onClick={() => toDocuments(firstPaper)}>
            前往文献页查看
          </Button>
          {!firstPaper && (
            <Typography.Text type="secondary">
              语义检索结果未返回 paper_id，无法跳转
            </Typography.Text>
          )}
        </Space>
      </Space>
    )
  }

  const renderConfirm = (data: ConfirmRequest) => (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Typography.Text>
        第 {data.round} 轮 · session {data.session_id}
      </Typography.Text>
      <pre style={PRE_STYLE}>{data.draft.incar.split('\n').slice(0, 10).join('\n')}</pre>
      <div>
        <Typography.Text strong>校验结果：</Typography.Text>
        {data.validation.findings.length === 0 ? (
          <Typography.Text type="secondary"> 无</Typography.Text>
        ) : (
          <Space direction="vertical" size={4} style={{ marginTop: 4 }}>
            {data.validation.findings.map((f, i) => (
              <div key={`${f.rule_id}-${i}`}>
                <Tag color={SEVERITY_COLORS[f.severity]}>{f.severity}</Tag>
                <Typography.Text strong>{f.rule_id}</Typography.Text> {f.message}
              </div>
            ))}
          </Space>
        )}
      </div>
      {data.validation.missing_fields.length > 0 && (
        <div>
          <Typography.Text strong>缺失字段：</Typography.Text>
          {data.validation.missing_fields.map((field) => (
            <Tag key={field}>{field}</Tag>
          ))}
        </div>
      )}
      <div>
        <Typography.Text strong>POTCAR 顺序：</Typography.Text>
        {data.draft.potcar_order.map((p) => (
          <Tag key={p}>{p}</Tag>
        ))}
      </div>
      <Button type="primary" onClick={() => navigate('/calc')}>
        前往计算页继续
      </Button>
    </Space>
  )

  const renderDocumentStatus = (data: ChatDocumentStatus) => (
    <Space direction="vertical" size="small">
      <Space>
        <Tag color={STATUS_COLORS[data.status]}>{data.status}</Tag>
      </Space>
      <div>
        <Typography.Text strong>paper_id：</Typography.Text>
        <Typography.Text copyable>{data.paper_id}</Typography.Text>
      </div>
      <Typography.Text>pending_action：{data.pending_action ?? '—'}</Typography.Text>
      <Typography.Text>
        card_ids：{data.card_ids ? data.card_ids.length : '—'}
      </Typography.Text>
      <Button type="primary" onClick={() => toDocuments(data.paper_id)}>
        前往文献页查看
      </Button>
    </Space>
  )

  const renderClarify = (data: ClarifyData) => (
    <Space direction="vertical" size="small" style={{ width: '100%' }}>
      {data.message && <Alert type="info" showIcon message={data.message} />}
      {data.missing_slots && data.missing_slots.length > 0 && (
        <div>
          <Typography.Text strong>缺少槽位：</Typography.Text>
          {data.missing_slots.map((slot) => (
            <Tag key={slot}>{slot}</Tag>
          ))}
        </div>
      )}
      {data.suggestions && data.suggestions.length > 0 && (
        <List
          size="small"
          dataSource={data.suggestions}
          renderItem={(item) => <List.Item>{item}</List.Item>}
        />
      )}
      <Typography.Text type="secondary">请补充信息后重新输入。</Typography.Text>
    </Space>
  )

  const renderChatError = (data: ChatError) => (
    <Alert
      type="error"
      showIcon
      message={`错误 ${data.code}：${data.message}`}
      description={data.code === 5005 ? '请检查后端日志。' : undefined}
    />
  )

  const renderItem = (item: ChatSubResult) => {
    switch (item.type) {
      case 'cards':
        return renderCards(item)
      case 'confirm_request':
        return renderConfirm(item)
      case 'document_status':
        return renderDocumentStatus(item)
      case 'clarify':
        return renderClarify(item)
      case 'error':
        return renderChatError(item)
      default:
        return null
    }
  }

  const renderResults = (data: ResultsData) => (
    <Collapse
      defaultActiveKey={data.results.map((_, index) => String(index))}
      items={data.results.map((item, index) => ({
        key: String(index),
        label: (
          <Space>
            📌 问题 {index + 1}：{TYPE_LABELS[item.type] ?? item.type}
            {item.type === 'error' && <Tag color="red">错误</Tag>}
          </Space>
        ),
        children: renderItem(item),
      }))}
    />
  )

  const renderResponse = (data: ChatResponse) =>
    data.type === 'results' ? renderResults(data) : renderItem(data)

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="助手">
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Input.TextArea
            rows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="用自然语言描述你的需求，比如：提取这篇论文的 DFT 参数 / 帮我生成 Co3O4 (111) 的 INCAR"
          />
          <Space wrap>
            <Input
              style={{ width: 340 }}
              value={paperId}
              onChange={(e) => setPaperId(e.target.value)}
              placeholder="paper_id（可选，文献段）"
            />
            <Input
              style={{ width: 340 }}
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="session_id（可选，计算段）"
            />
            <Button type="primary" loading={loading} onClick={() => void handleSend()}>
              发送
            </Button>
          </Space>
          <Typography.Text type="secondary">
            每次请求独立，不保留对话历史；结果摘要可跳转到文献/计算页继续。
          </Typography.Text>
        </Space>
      </Card>

      <Card title="路由结果">
        {error ? (
          <Alert
            type="error"
            showIcon
            closable
            onClose={() => setError(null)}
            message={`请求失败（code=${error.code}）`}
            description={
              <Space direction="vertical" size={2}>
                <span>{error.message}</span>
                {error.code === 5005 && (
                  <span>两个请求都失败了，请检查后端日志。</span>
                )}
              </Space>
            }
          />
        ) : response ? (
          renderResponse(response)
        ) : (
          <Typography.Text type="secondary">发送请求后在这里查看摘要。</Typography.Text>
        )}
      </Card>
    </Space>
  )
}
