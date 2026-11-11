import { Layout, Menu, Typography } from 'antd'
import type { MenuProps } from 'antd'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'

const { Header, Sider, Content } = Layout

const items: MenuProps['items'] = [
  { key: '/', label: '健康' },
  { key: '/chat', label: '助手' },
  { key: '/documents', label: '文献' },
  { key: '/calc', label: '计算' },
]

export default function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const section = location.pathname.split('/')[1]
  const selectedKey = section ? `/${section}` : '/'

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="dark" width={200}>
        <div
          style={{
            height: 48,
            margin: 16,
            color: '#fff',
            fontSize: 18,
            fontWeight: 600,
            lineHeight: '48px',
          }}
        >
          chem-agent
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', paddingInline: 24 }}>
          <Typography.Title level={4} style={{ margin: 0, lineHeight: '64px' }}>
            chem-agent
          </Typography.Title>
        </Header>
        <Content style={{ margin: 24 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
