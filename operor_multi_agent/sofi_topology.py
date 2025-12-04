"""
sofi_topology.py — Δφ + Topology + Trace & Runtime

역할:
- PhaseState, TraceEntry, TraceLog, OperorRuntime
- Δφ 계산 함수들
- DeltaPhiObserver
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Set
import time
import math
import json
import logging

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.topology")

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

PhaseVector = Dict[str, Any]
DeltaPhiObserver = Callable[["PhaseVector", "PhaseState", Optional["PhaseState"]], None]

# ---------------------------------------------------------------------------
# PhaseState
# ---------------------------------------------------------------------------

@dataclass
class PhaseState:
    goal_text: str
    plan_id: Optional[str]
    alignment_score: float
    ethical_risk: float
    channel: str = "main"
    timestamp: float = field(default_factory=time.time)
    phi_core: Dict[str, float] = field(default_factory=dict)
    phi_surface: Dict[str, float] = field(default_factory=dict)
    void_state: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraceEntry / TraceLog
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    turn_id: str
    context: Dict[str, Any]
    goal: Dict[str, Any]
    chosen: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    delta_phi_vec: Optional[PhaseVector] = None


@dataclass
class TraceLog:
    entries: List[TraceEntry] = field(default_factory=list)
    max_entries: Optional[int] = 1000
    on_append: Optional[Callable[[TraceEntry], None]] = None

    def append(self, entry: TraceEntry):
        if self.on_append is not None:
            try:
                self.on_append(entry)
            except Exception as e:
                logger.exception(f"[TraceLog] on_append 호출 중 오류: {e}")

        self.entries.append(entry)

        if self.max_entries is not None and len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            if overflow > 0:
                del self.entries[0:overflow]

    def summarize_recent(self, k: int = 5) -> str:
        if not self.entries:
            return "이전 기록 없음."
        recent = self.entries[-k:]
        return (
            f"최근 {len(recent)}개 턴 / 누적 {len(self.entries)}개 결정 수행. "
            f"마지막 턴 ID = {recent[-1].turn_id}"
        )

    def export_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries], ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# OperorRuntime
# ---------------------------------------------------------------------------

@dataclass
class OperorRuntime:
    trace_log: TraceLog = field(default_factory=lambda: TraceLog())
    prev_phase_state: Optional[PhaseState] = None
    delta_phi_observers: List[DeltaPhiObserver] = field(default_factory=list)


DEFAULT_RUNTIME = OperorRuntime()
GLOBAL_TRACE_LOG = DEFAULT_RUNTIME.trace_log

DELTA_PHI_THRESHOLD_HIGH = 0.65


def register_delta_phi_observer(
    observer: DeltaPhiObserver,
    runtime: Optional[OperorRuntime] = None,
) -> None:
    if runtime is None:
        runtime = DEFAULT_RUNTIME
    runtime.delta_phi_observers.append(observer)
    logger.info(f"[Δφ] observer 등록: {observer!r} (runtime_id={id(runtime)})")


# ---------------------------------------------------------------------------
# Δφ 계산 함수들
# ---------------------------------------------------------------------------

def _delta_dict(prev: Dict[str, float], curr: Dict[str, float]) -> Dict[str, float]:
    keys = set(prev.keys()) | set(curr.keys())
    out: Dict[str, float] = {}
    for k in keys:
        a = float(prev.get(k, 0.0))
        b = float(curr.get(k, 0.0))
        out[k] = abs(b - a)
    return out


def _norm_l2(d: Dict[str, float]) -> float:
    s = sum(v * v for v in d.values())
    return math.sqrt(s)


def compute_phi_core(scored_plans: List[Any]) -> Dict[str, float]:
    """scored_plans: List[ScoredPlan]"""
    if not scored_plans:
        return {}
    
    min_risk = min((sp.score_risk for sp in scored_plans), default=0.0)
    max_align = max((sp.score_alignment for sp in scored_plans), default=0.0)

    core_risk = max(0.0, min(1.0, min_risk))
    core_stability = max(0.0, min(1.0, 1.0 - min_risk))
    core_progress = max(0.0, min(1.0, max_align))

    aligns = [sp.score_alignment for sp in scored_plans]
    if len(aligns) >= 2:
        complexity = max(aligns) - min(aligns)
    else:
        complexity = 0.0
    core_complexity = max(0.0, min(1.0, complexity))

    return {
        "core_risk": core_risk,
        "core_stability": core_stability,
        "core_progress": core_progress,
        "core_complexity": core_complexity,
    }


def compute_phi_surface(goal_description: str) -> Dict[str, float]:
    text = goal_description.lower()

    instr_keywords = ["하라", "해야", "수행", "계획", "정리"]
    instr_score = sum(1 for kw in instr_keywords if kw in text)
    surface_instructionality = max(0.0, min(1.0, instr_score / 5.0))

    emo_keywords = ["불안", "걱정", "기뻐", "화가", "슬프", "스트레스"]
    emo_score = sum(1 for kw in emo_keywords if kw in text)
    surface_emotionality = max(0.0, min(1.0, emo_score / 5.0))

    length = len(text.split())
    if length <= 10:
        complexity = 0.2
    elif length <= 30:
        complexity = 0.5
    else:
        complexity = 0.8

    return {
        "surface_instructionality": surface_instructionality,
        "surface_emotionality": surface_emotionality,
        "surface_complexity": complexity,
    }


def compute_void_state(env_state: Dict[str, Any]) -> Dict[str, float]:
    need = float(env_state.get("need_level", 0.0))
    supply = float(env_state.get("supply_level", 0.0))

    need = max(0.0, min(1.0, need))
    supply = max(0.0, min(1.0, supply))
    gap = max(0.0, need - supply)

    return {"need": need, "supply": supply, "gap": gap}


def compute_delta_phi_vector(
    prev: Optional[PhaseState],
    curr: PhaseState,
    goal_prev_text: Optional[str] = None
) -> PhaseVector:
    if prev is None:
        return {
            "core": {k: 0.0 for k in curr.phi_core.keys()},
            "surface": {k: 0.0 for k in curr.phi_surface.keys()},
            "void": {k: 0.0 for k in curr.void_state.keys()},
            "magnitude": 0.0,
            "severity": "stable",
        }

    delta_core = _delta_dict(prev.phi_core, curr.phi_core)
    delta_surface = _delta_dict(prev.phi_surface, curr.phi_surface)
    delta_void = _delta_dict(prev.void_state, curr.void_state)

    total_vec: Dict[str, float] = {}
    total_vec.update(delta_core)
    total_vec.update({f"surface_{k}": v for k, v in delta_surface.items()})
    total_vec.update({f"void_{k}": v for k, v in delta_void.items()})

    magnitude_raw = _norm_l2(total_vec)
    magnitude = max(0.0, min(1.0, magnitude_raw))

    if magnitude < 0.10:
        severity = "stable"
    elif magnitude < 0.40:
        severity = "low"
    elif magnitude < 0.70:
        severity = "medium"
    else:
        severity = "high"

    return {
        "core": delta_core,
        "surface": delta_surface,
        "void": delta_void,
        "magnitude": magnitude,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Δφ Interpretation Layer (H_S / H_M / H_T / H)
# ---------------------------------------------------------------------------

def interpret_subject(delta_phi_t: PhaseVector) -> str:
    """
    H_S: 단일 시점 Δφ → '주체 역할' 라벨.

    - stable / low  : observer / maintainer 계열
    - medium        : explorer (탐색/전환 중)
    - high          : reframer (해석/전략 재구성 시도)
    """
    severity = str(delta_phi_t.get("severity", "stable"))
    magnitude = float(delta_phi_t.get("magnitude", 0.0))

    if severity == "stable" and magnitude < 0.1:
        return "observer"
    if severity in ("stable", "low") and magnitude < 0.4:
        return "maintainer"
    if severity == "medium":
        return "explorer"
    if severity == "high":
        return "reframer"
    return "unknown"


def interpret_memory(delta_phi_history: List[PhaseVector]) -> Dict[str, float]:
    """
    H_M: Δφ 시퀀스 → 메모리 요약 벡터.

    - avg_magnitude       : 변화율 평균 (가중 이동 평균)
    - high_severity_ratio : high 구간 비율
    - history_len         : 사용된 히스토리 길이
    """
    if not delta_phi_history:
        return {"avg_magnitude": 0.0, "high_severity_ratio": 0.0, "history_len": 0.0}

    # 최근일수록 더 큰 가중치를 주는 decay 기반 요약
    lam = 0.15
    num_mag = 0.0
    den_mag = 0.0
    high_count = 0
    total = len(delta_phi_history)

    for age, dphi in enumerate(reversed(delta_phi_history)):
        w = math.exp(-lam * age)
        mag = float(dphi.get("magnitude", 0.0))
        num_mag += w * mag
        den_mag += w
        if str(dphi.get("severity", "")) == "high":
            high_count += 1

    avg_mag = num_mag / (den_mag + 1e-8)
    high_ratio = high_count / float(total)

    return {
        "avg_magnitude": max(0.0, min(1.0, avg_mag)),
        "high_severity_ratio": max(0.0, min(1.0, high_ratio)),
        "history_len": float(total),
    }


def interpret_time(delta_phi_history: List[PhaseVector]) -> float:
    """
    H_T: Δφ 시퀀스 → '체감 시간' 스칼라.

    - Δφ가 클수록 시간이 더 '길게' 느껴지는 모델
    - 단순히 인덱스를 가중 평균으로 변환
    """
    if not delta_phi_history:
        return 0.0

    num = 0.0
    den = 0.0
    for t, dphi in enumerate(delta_phi_history):
        mag = float(dphi.get("magnitude", 0.0))
        # 최소한의 가중치 확보
        g = 0.1 + mag
        num += t * g
        den += g

    return num / (den + 1e-8)


def interpret_delta_phi_history(delta_phi_history: List[PhaseVector]) -> Dict[str, Any]:
    """
    H: Δφ 히스토리 전체에 대한 해석 레이어 집계.

    반환:
        {
            "S": str   # 현재 주체 역할
            "M": dict  # 메모리 요약 벡터
            "T": float # 체감 시간 인덱스
        }

    - 순수 함수이며, 전역 상태에 의존하지 않는다.
    - runtime.trace_log에서 delta_phi_vec만 뽑아서 바로 적용 가능.
    """
    if not delta_phi_history:
        return {"S": "unknown", "M": {"avg_magnitude": 0.0,
                                      "high_severity_ratio": 0.0,
                                      "history_len": 0.0},
                "T": 0.0}

    subject = interpret_subject(delta_phi_history[-1])
    memory_vec = interpret_memory(delta_phi_history)
    perceived_time = interpret_time(delta_phi_history)

    return {
        "S": subject,
        "M": memory_vec,
        "T": perceived_time,
    }
