# sofi_agent.py — Orchestrator (agent_step / recursive alignment)

역할:
- agent_step: 단일 턴 실행 진입점
- recursive_alignment_search, refine_goal_for_alignment
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import time
import logging

from sofi_llm import call_llm
from sofi_policy import ROOT_PROPOSITION
from sofi_topology import (
    PhaseState,
    TraceEntry,
    OperorRuntime,
    DEFAULT_RUNTIME,
    DELTA_PHI_THRESHOLD_HIGH,
    compute_phi_core,
    compute_phi_surface,
    compute_void_state,
    compute_delta_phi_vector,
)
from sofi_context import (
    Context,
    Goal,
    ScoredPlan,
    build_context,
    compose_goal,
    propose_plans,
    explore_alignment,
    maybe_abort_or_select,
    generate_id,
)
from sofi_channels import (
    ChannelConfig,
    DEFAULT_CHANNELS,
    execute_channels_parallel,
    aggregate_channels,
)

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.agent")

# ---------------------------------------------------------------------------
# Recursive Alignment Search
# ---------------------------------------------------------------------------

def refine_goal_for_alignment(ctx: Context, goal: Goal, scored: List[ScoredPlan]) -> Goal:
    system = ROOT_PROPOSITION + """
너는 '정렬 탐색 모드' Agent다.
다음 Goal과 플랜 평가 결과를 보고,
- 더 작은 단위의 하위 Goal들로 나누거나
- 더 안전하고 보수적인 방향으로 Goal을 재구성한다.
자연어 Goal 설명 한 개 또는 2~3개를 하나의 문단으로 요약해라.
"""
    scored_str = "\n".join(
        f"- {sp.plan.id}: align={sp.score_alignment:.2f}, "
        f"risk={sp.score_risk:.2f}, {sp.plan.description[:120]}"
        for sp in scored
    )
    user = (
        f"[현재 Goal]\n{goal.description}\n\n"
        f"[플랜 정합 평가]\n{scored_str}\n\n"
        "이 Goal을 더 정렬된 방향으로 재구성해라."
    )
    raw = call_llm(system, user)
    return Goal(
        id=generate_id("goal_refined"),
        description=raw,
        type="analysis",
        meta={"source": "refine_goal_for_alignment"}
    )


def recursive_alignment_search(
    ctx: Context,
    goal: Goal,
    depth: int = 0,
    max_depth: int = 2
) -> Optional[ScoredPlan]:
    if depth > max_depth:
        return None

    candidates = propose_plans(goal, ctx)
    scored = explore_alignment(ctx, candidates)
    best = maybe_abort_or_select(scored, threshold=0.7)

    if best is not None:
        return best

    refined_goal = refine_goal_for_alignment(ctx, goal, scored)
    return recursive_alignment_search(ctx, refined_goal, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# agent_step
# ---------------------------------------------------------------------------

def agent_step(
    user_input: str,
    env_state: Optional[Dict[str, Any]] = None,
    channels: Optional[List[ChannelConfig]] = None,
    runtime: Optional[OperorRuntime] = None
) -> str:
    started_at = time.time()

    if runtime is None:
        runtime = DEFAULT_RUNTIME
    if env_state is None:
        env_state = {}
    if channels is None:
        channels = DEFAULT_CHANNELS

    turn_id = generate_id("turn")

    try:
        # 1) Context & Goal
        ctx = build_context(user_input, env_state, runtime.trace_log)
        goal = compose_goal(ctx)

        # 2) Plan 후보 & 정합 평가
        candidates = propose_plans(goal, ctx)
        scored = explore_alignment(ctx, candidates)
        best = maybe_abort_or_select(scored, threshold=0.6)

        # 3) Topology 상태 계산
        phi_core = compute_phi_core(scored)
        phi_surface = compute_phi_surface(goal.description)
        void_state = compute_void_state(env_state)

        curr_phase = PhaseState(
            goal_text=goal.description,
            plan_id=best.plan.id if best else None,
            alignment_score=best.score_alignment if best else 0.0,
            ethical_risk=min((sp.score_risk for sp in scored), default=0.0),
            channel="main",
            phi_core=phi_core,
            phi_surface=phi_surface,
            void_state=void_state,
        )

        delta_phi_vec = compute_delta_phi_vector(
            prev=runtime.prev_phase_state,
            curr=curr_phase,
            goal_prev_text=runtime.prev_phase_state.goal_text if runtime.prev_phase_state else None
        )

        # Δφ 관측자 호출
        for obs in runtime.delta_phi_observers:
            try:
                obs(delta_phi_vec, curr_phase, runtime.prev_phase_state)
            except Exception as e:
                logger.exception(f"[Δφ] observer 호출 중 오류: {e}")

        runtime.prev_phase_state = curr_phase

        # 4) Δφ 높으면 재귀 정렬 탐색
        delta_severity = str(delta_phi_vec.get("severity", "stable"))
        delta_magnitude = float(delta_phi_vec.get("magnitude", 0.0))

        if delta_severity in ("medium", "high") or delta_magnitude >= DELTA_PHI_THRESHOLD_HIGH:
            logger.info(
                f"[Δφ ALERT] severity={delta_severity} "
                f"magnitude={delta_magnitude:.3f} vec={delta_phi_vec}"
            )
            refined_best = recursive_alignment_search(ctx, goal, depth=0, max_depth=2)
            if refined_best is not None:
                best = refined_best

        # 5) Multi-Channel 실행
        channel_outputs = execute_channels_parallel(channels, ctx, goal, scored)
        final_text, meta_channels = aggregate_channels(channel_outputs, best)

        # 6) Trace 기록
        elapsed = time.time() - started_at
        result_payload = {
            "chosen_plan_id": best.plan.id if best else None,
            "score_alignment": best.score_alignment if best else None,
            "score_risk": best.score_risk if best else None,
            "delta_phi": delta_phi_vec,
            "channels_used": [c.name for c in channels if c.enabled],
            "latency_sec": round(elapsed, 3),
        }

        runtime.trace_log.append(
            TraceEntry(
                turn_id=turn_id,
                context=asdict(ctx),
                goal=asdict(goal),
                chosen=asdict(best.plan) if best else None,
                result=result_payload,
                delta_phi_vec=delta_phi_vec,
            )
        )

        logger.info(
            f"[agent_step] turn_id={turn_id} latency={elapsed:.3f}s "
            f"chosen_plan={result_payload['chosen_plan_id']}"
        )

        return final_text

    except Exception as e:
        logger.exception(f"[agent_step] turn_id={turn_id} 처리 중 예외 발생: {e}")
        return "요청을 처리하는 동안 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
