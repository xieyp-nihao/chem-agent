"""统一异常体系（S8 迁移，§4.1 语义）。

- ServiceError：业务错误（S2 原样迁移；app/storage/files.py 以 re-export 保持
  旧 import 路径兼容，类对象唯一）
- AgentExecutionError：图/节点执行失败（5005 语义）
- PipelineError：编排管道失败（两段串行隔离）
- IntentRouteError：意图路由失败（llm_fallback 输出非法等）
"""


class BaseServiceError(Exception):
    """业务异常基类：code + message + details。"""

    def __init__(self, code: int, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ServiceError(BaseServiceError):
    """通用业务错误（API 层按 §7.9 映射 HTTP 状态）。"""


class AgentExecutionError(BaseServiceError):
    """图/节点执行失败（对应 5005 INTERNAL_ERROR）。"""


class PipelineError(BaseServiceError):
    """编排管道失败（如两段串行中某段崩溃）。"""


class IntentRouteError(BaseServiceError):
    """意图路由失败（llm_fallback 输出非法/超时，通常降级为 clarify）。"""
