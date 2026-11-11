import { Alert, Button, Card, Descriptions, Space, Spin, Tag, Typography } from 'antd'
import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../api/client'
import { getHealth } from '../api/health'
import type { ComponentName, HealthData, HealthStatus } from '../api/types'

const COMPONENT_LABELS: Record<ComponentName, string> = {
  sqlite: 'SQLite',
  checkpointer: 'Checkpointer',
  llm: 'LLM',
  mp: 'Materials Project',
  chroma: 'ChromaDB',
  embedding_model: 'Embedding Model',
}

const HEALTH_COLORS: Record<HealthStatus, string> = {
  ok: 'green',
  degraded: 'orange',
  down: 'red',
}

function componentColor(status: string): string {
  if (status === 'ok') return 'green'
  if (status === 'unavailable') return 'red'
  return 'default'
}

export default function HealthPage() {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [updatedAt, setUpdatedAt] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHealth()
      setHealth(data)
      setUpdatedAt(new Date().toLocaleString())
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, '未知错误', err))
      setHealth(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  return (
    <Space direction="vertical" size="large" style={{ width: '100%', maxWidth: 720 }}>
      <Card
        title="系统健康"
        extra={
          <Button onClick={() => void load()} loading={loading}>
            刷新
          </Button>
        }
      >
        {loading && !health ? (
          <Spin />
        ) : error ? (
          <Alert
            type="error"
            showIcon
            message={`请求失败（code=${error.code}）`}
            description={error.message}
          />
        ) : health ? (
          <Descriptions column={1} bordered size="middle">
            <Descriptions.Item label="总体状态">
              <Tag color={HEALTH_COLORS[health.status]}>{health.status}</Tag>
            </Descriptions.Item>
            {(Object.keys(COMPONENT_LABELS) as ComponentName[]).map((name) => (
              <Descriptions.Item key={name} label={COMPONENT_LABELS[name]}>
                <Tag color={componentColor(health.components[name])}>
                  {health.components[name] ?? 'unknown'}
                </Tag>
              </Descriptions.Item>
            ))}
          </Descriptions>
        ) : (
          <Typography.Text type="secondary">暂无数据</Typography.Text>
        )}
      </Card>
      {updatedAt && <Typography.Text type="secondary">上次刷新：{updatedAt}</Typography.Text>}
    </Space>
  )
}
