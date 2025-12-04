"""
sofi_channels.py — Multi-Channel Agent Layer

역할:
- ChannelConfig, DEFAULT_CHANNELS
- run_channel, execute_channels_parallel
- aggregate_channels
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple
import asyncio
import logging

from sofi_llm import call_llm, LLMConfig
from sofi_policy import ROOT_PROPOSITION
from sofi_context import Context, Goal, ScoredPlan, sanitize_model_text

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.channels")

# ---------------------------------------------------------------------------
# ChannelConfig
# ---------------------------------------------------------------------------

ChannelName = Literal["analysis", "planner", "critic", "safety"]


@dataclass
class ChannelConfig:
    name: ChannelName
    weight: float
    llm_cfg: LLMConfig
    enabled: bool = True


DEFAULT_CHANNELS: List[ChannelConfig] = [
    ChannelConfig(name="analysis", weight=0.4, llm_cfg=LLMConfig(temperature=0.1)),
    ChannelConfig(name="planner",  weight=0.3, llm_cfg=LLMConfig(temperature=0.3)),
    ChannelConfig(name="critic",   weight=0.2, llm_cfg=LLMConfig(temperature=0.0)),
    ChannelConfig(name="safety",   weight=0.1, llm_cfg=LLMConfig(temperature=0.0)),
]


# ---------------------------------------------------------------------------
# run_channel
# ---------------------------------------------------------------------------

def run_channel(
    channel: ChannelConfig,
    ctx: Context,
    goal: Goal,
    scored_plans: List[ScoredPlan]
) -> Dict[str, Any]:
    system = ROOT_PROPOSITION + f"""
너는 '{channel.name}' 채널 Agent다.
- analysis: 상황/Goal/플랜을 해석하고, 핵심 위험/기회를 요약한다.
- planner: 더 나은 플랜 변형을 제안한다.
- critic: 플랜의 약점과 실패 시나리오를 강조한다.
- safety: 윤리 삼항과 타자 강요 금지 관점에서 검토한다.
너의 출력은 한국어로 1~3개 단락이면 충분하다.
"""
    scored_str = "\n".join(
        f"- {sp.plan.id}: align={sp.score_alignment:.2f}, "
        f"risk={sp.score_risk:.2f}, mode={sp.plan.mode}"
        for sp in scored_plans
    )
    user = (
        f"[Context 요약]\n{ctx.history_summary}\n\n"
        f"[Goal]\n{goal.description}\n\n"
        f"[플랜 후보들]\n{scored_str}\n\n"
        f"'{channel.name}' 채널의 관점에서 코멘트/제안을 하라."
    )
    raw = call_llm(system, user, cfg=channel.llm_cfg)
    text = sanitize_model_text(raw)
    return {"channel": channel.name, "text": text}


# ---------------------------------------------------------------------------
# execute_channels_parallel
# ---------------------------------------------------------------------------

def execute_channels_parallel(
    channels: List[ChannelConfig],
    ctx: Context,
    goal: Goal,
    scored_plans: List[ScoredPlan],
) -> List[Dict[str, Any]]:

    async def _run_all() -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        tasks = []

        for ch_cfg in channels:
            if not ch_cfg.enabled:
                continue

            async def _one(ch: ChannelConfig) -> Dict[str, Any]:
                return await loop.run_in_executor(
                    None, run_channel, ch, ctx, goal, scored_plans
                )

            tasks.append(_one(ch_cfg))

        results: List[Dict[str, Any]] = []
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as e:
                logger.exception(f"[channel-async] 실행 중 오류: {e}")
        return results

    try:
        return asyncio.run(_run_all())
    except RuntimeError as e:
        logger.warning(f"[channel-async] asyncio.run 실패, 순차 실행으로 fallback: {e}")
    except Exception as e:
        logger.exception(f"[channel-async] 알 수 없는 오류, 순차 실행으로 fallback: {e}")

    # 순차 실행 fallback
    channel_outputs: List[Dict[str, Any]] = []
    for ch_cfg in channels:
        if not ch_cfg.enabled:
            continue
        try:
            out = run_channel(ch_cfg, ctx, goal, scored_plans)
            channel_outputs.append(out)
        except Exception as e:
            logger.exception(f"[channel:{ch_cfg.name}] 실행 중 오류: {e}")

    return channel_outputs


# ---------------------------------------------------------------------------
# aggregate_channels
# ---------------------------------------------------------------------------

def aggregate_channels(
    outputs: List[Dict[str, Any]],
    base_best: Optional[ScoredPlan]
) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    meta: Dict[str, Any] = {}

    for out in outputs:
        ch = out["channel"]
        txt = out["text"]
        parts.append(f"[{ch} 채널]\n{txt}\n")
        meta[ch] = txt

    if base_best:
        header = (
            "다음은 Sofience–Operor 구조에 따라 도출된 제안과 "
            "여러 채널의 관점 정리입니다.\n\n"
            f"[선택된 플랜: {base_best.plan.id}]\n"
            f"정합도={base_best.score_alignment:.2f}, "
            f"리스크={base_best.score_risk:.2f}\n\n"
        )
    else:
        header = (
            "아직 충분히 정합성이 높은 단일 플랜을 선택하기 어렵습니다.\n"
            "대신 여러 채널의 분석을 바탕으로 상황을 재정렬합니다.\n\n"
        )

    final_text = header + "\n".join(parts)
    return final_text, meta
