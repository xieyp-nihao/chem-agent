"""规则校验 Service：薄包装 app.domain.rules.RuleValidator。

对应 docs/technical-design.md §4.4：
- 输入 (parameters, surface, calculation_type, structure?) → 构造 ValidationContext
  → 调 domain 层 validate → 返回 ValidationReport
- 不重复实现 R1–R7；不调 LLM、不写数据库、不抛业务异常（除非法输入）
"""
from __future__ import annotations

import logging

from app.domain.rules import RuleValidator as DomainRuleValidator
from app.domain.schemas import (
    CalculationType,
    DFTParameters,
    StructureInfo,
    SurfaceModel,
    ValidationContext,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class RuleValidatorService:
    """规则校验薄包装（同步、无状态）。"""

    def validate(
        self,
        parameters: DFTParameters,
        surface: SurfaceModel | None,
        calculation_type: CalculationType,
        structure: StructureInfo | None = None,
    ) -> ValidationReport:
        ctx = ValidationContext(
            parameters=parameters,
            surface=surface,
            structure=structure,
            calculation_type=calculation_type,
        )
        report = DomainRuleValidator().validate(ctx)
        logger.info(
            "rule validation done: findings=%d (error=%d warning=%d info=%d) missing=%d",
            len(report.findings),
            len(report.errors),
            len(report.warnings),
            len(report.infos),
            len(report.missing_fields),
        )
        return report
