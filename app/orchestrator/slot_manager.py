"""会话槽位管理（§2.4 编排层约定）。

- 读写 sessions.slots_json；只写 sessions，不写 papers/cards
- 计算槽位合并优先级：当前请求 > card_id 卡片 > session
- TTL：SESSION_TTL_HOURS（sessions.expires_at）
- 只存 CalcSlots JSON，不缓存 LLM 原始输出
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.domain.schemas import CalcSlots


def _naive_utc(dt) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _slots_from_card(card) -> CalcSlots:
    p = card.parameters
    return CalcSlots(
        calculation_type=p.calculation_type,
        functional=p.functional,
        functional_family=p.functional_family,
        hubbard_u=list(p.hubbard_u),
        encut_ev=p.encut_ev,
        kpoints=p.kpoints,
        ediff=p.ediff,
        ediffg=p.ediffg,
        spin_polarized=p.spin_polarized,
        vdw=p.vdw,
        surface=card.surface,
    )


def _overlay(base: CalcSlots, overlay: CalcSlots) -> CalcSlots:
    data = base.model_dump()
    for key, value in overlay.model_dump().items():
        if value is not None and value != []:
            data[key] = value
    return CalcSlots.model_validate(data)


class SlotManager:
    """会话槽位（异步、无状态；session factory 注入）。"""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def get_session_slots(self, session_id: str) -> CalcSlots | None:
        from app.storage.db import Session

        async with self._session_factory() as s:
            row = await s.get(Session, session_id)
            if row is None:
                return None
            if _naive_utc(row.expires_at) < _naive_utc(datetime.now(timezone.utc)):
                return None  # TTL 过期
            try:
                return CalcSlots.model_validate(json.loads(row.slots_json))
            except Exception:
                return None

    async def save_session_slots(self, session_id: str, slots: CalcSlots) -> None:
        from app.storage.db import Session

        now = _naive_utc(datetime.now(timezone.utc))
        expires = now + timedelta(hours=settings.SESSION_TTL_HOURS)
        async with self._session_factory() as s:
            row = await s.get(Session, session_id)
            if row is None:
                s.add(Session(
                    session_id=session_id, slots_json=slots.model_dump_json(),
                    expires_at=expires,
                ))
            else:
                row.slots_json = slots.model_dump_json()
                row.expires_at = expires
            await s.commit()

    async def merge(
        self,
        *,
        user_slots: CalcSlots,
        card: object | None = None,
        session_id: str | None = None,
    ) -> CalcSlots:
        """合并优先级：session < card < user。"""
        slots = CalcSlots()
        if session_id:
            session_slots = await self.get_session_slots(session_id)
            if session_slots is not None:
                slots = _overlay(slots, session_slots)
        if card is not None:
            slots = _overlay(slots, _slots_from_card(card))
        return _overlay(slots, user_slots)
