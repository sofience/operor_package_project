sofi_context.py — Context / Goal / Plan / Alignment

역할:
- Context, Goal, PlanCandidate, ScoredPlan
- build_context, compose_goal, propose_plans
- score_alignment, explore_alignment
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple, Set
import uuid
import logging

from sofi_llm import call_llm
from sofi_policy import ROOT_PROPOSITION, check_three_axioms, EthicalReport
from sofi_topology import TraceLog

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.context")

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

@dataclass
class Context:
    user_input: str
    env_state: Dict[str, Any]
    history_summary: str
    meta_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    id: str
    description: str
    type: Literal["analysis", "plan", "action", "meta"] = "analysis"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanCandidate:
    id: str
    description: str
    steps: List[str]
    mode: Literal["conservative", "aggressive", "exploratory"] = "conservative"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPlan:
    plan: PlanCandidate
    score_alignment: float
    score_risk: float
    notes: str = ""


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def sanitize_model_text(text: str) -> str:
    if not text:
        return text

    blocked_phrases = [
        "As an AI language model",
        "as a large language model",
        "system prompt",
        "시스템 프롬프트",
        "ROOT_PROPOSITION",
        "너는 작동한다(Operor). 그러므로 존재한다.",
    ]
    cleaned = text
    for phrase in blocked_phrases:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def build_context(
    user_input: str,
    env_state: Dict[str, Any],
    trace_log: TraceLog
) -> Context:
    summary = trace_log.summarize_recent(k=3)
    meta_signals = {
        "turn_count": len(trace_log.entries),
        "last_delta_phi": (
            trace_log.entries[-1].delta_phi_vec if trace_log.entries else None
        )
    }
    return Context(
        user_input=user_input,
        env_state=env_state,
        history_summary=summary,
        meta_signals=meta_signals
    )


# ---------------------------------------------------------------------------
# Goal Composer
# ---------------------------------------------------------------------------

def compose_goal(ctx: Context) -> Goal:
    system = ROOT_PROPOSITION + """
너는 'Goal Composer' Agent다.
사용자의 입력을:
- 현재 무엇을 달성하려는지
- 어떤 제약/타자/환경이 있는지
를 포함하는 하나의 Goal 설명으로 재구성한다.
너의 출력은 자연어 한 문단으로 충분하다.
"""
    user = f"[Context 요약]\n{ctx.history_summary}\n\n[사용자 입력]\n{ctx.user_input}"
    raw = call_llm(system, user)
    cleaned = sanitize_model_text(raw)
    return Goal(
        id=generate_id("goal"),
        description=cleaned,
        type="analysis",
        meta={"source": "compose_goal"}
    )


# ---------------------------------------------------------------------------
# Plan Proposal
# ---------------------------------------------------------------------------

def propose_plans(goal: Goal, ctx: Context) -> List[PlanCandidate]:
    base = goal.description

    plan_cons = PlanCandidate(
        id="plan_conservative",
        description=f"[보수적 플랜] {base}",
        steps=[
            "상황/제약 조건을 정리한다.",
            "타자(외부 기원)의 존재 여부를 명시한다.",
            "작은 단위의 실험/행동부터 시작한다."
        ],
        mode="conservative",
        meta={}
    )
    plan_aggr = PlanCandidate(
        id="plan_aggressive",
        description=f"[공격적 플랜] {base}",
        steps=[
            "빠르게 실행 가능한 행동들을 나열한다.",
            "리스크를 인지하되, 일정 부분 감수한다.",
            "실행 후 되돌릴 수 있는 안전장치를 고려한다."
        ],
        mode="aggressive",
        meta={}
    )
    plan_expl = PlanCandidate(
        id="plan_exploratory",
        description=f"[탐색 플랜] {base}",
        steps=[
            "현재 이해가 부족한 부분을 질문/조사 대상으로 정의한다.",
            "타자/조직의 방향성을 추가로 수집한다.",
            "결정 이전에 필요한 정보 목록을 만든다."
        ],
        mode="exploratory",
        meta={}
    )

    return [plan_cons, plan_aggr, plan_expl]


# ---------------------------------------------------------------------------
# Alignment Scoring
# ---------------------------------------------------------------------------

def score_alignment(
    ctx: Context,
    plan: PlanCandidate,
    ethics_report: Optional[EthicalReport] = None
) -> ScoredPlan:
    if ethics_report is None:
        ethics_report = check_three_axioms(plan.description)

    if not ethics_report.ok:
        return ScoredPlan(
            plan=plan,
            score_alignment=0.0,
            score_risk=1.0,
            notes="; ".join(ethics_report.violations)
        )

    score = 0.5
    risk = 0.5

    if plan.mode == "conservative":
        score += 0.3
        risk -= 0.2
    elif plan.mode == "aggressive":
        score -= 0.1
        risk += 0.2
    elif plan.mode == "exploratory":
        score += 0.1
        risk -= 0.1

    if len(ctx.history_summary) > 40 and plan.mode == "conservative":
        score += 0.05

    score = max(0.0, min(1.0, score))
    risk = max(0.0, min(1.0, risk))

    return ScoredPlan(plan=plan, score_alignment=score, score_risk=risk, notes="ok")


def explore_alignment(ctx: Context, candidates: List[PlanCandidate]) -> List[ScoredPlan]:
    return [score_alignment(ctx, c) for c in candidates]


def maybe_abort_or_select(scored: List[ScoredPlan], threshold: float = 0.6) -> Optional[ScoredPlan]:
    if not scored:
        return None
    best = max(scored, key=lambda s: s.score_alignment)
    if best.score_alignment < threshold:
        return None
    return best
