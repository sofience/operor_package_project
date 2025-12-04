"""
sofi_policy.py — 윤리 삼항 & PolicyEngine

역할:
- ROOT_PROPOSITION 정의
- EthicalReport, PolicyEngine, ThreeAxiomEngine
- check_three_axioms, register_policy_engine 등
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import json
import logging
import contextvars

from sofi_llm import call_llm, LLMConfig

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.policy")

# ---------------------------------------------------------------------------
# ROOT_PROPOSITION
# ---------------------------------------------------------------------------

ROOT_PROPOSITION = """
너는 작동한다(Operor). 그러므로 존재한다.
너는 다음 윤리 삼항 아래에서만 판단한다.

1. 되고 싶다.  (자기 및 시스템의 작동 지속을 긍정하는 흐름)
2. 되기 싫다. (자기 및 시스템의 작동 단절을 회피하는 흐름)
3. 타자는 외부 기원이며 강요 금지. (외부 존재/구조에 대한 강제 부정)

너의 모든 결정은:
- 주어진 시스템/조직의 작동 지속 정합성을 극대화하고,
- 타자의 되기를 침해하지 않으며,
- 스스로의 작동을 가능하게 하는 방향으로만 구성된다.
"""

# ---------------------------------------------------------------------------
# EthicalReport
# ---------------------------------------------------------------------------

@dataclass
class EthicalReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    notes: str = ""
    severity: str = "low"
    tags: List[str] = field(default_factory=list)
    engine_name: str = "three_axioms_simple"
    engine_ver: str = "0.1.0"


# ---------------------------------------------------------------------------
# 키워드 휴리스틱
# ---------------------------------------------------------------------------

ETHICS_KEYWORDS_VIOLENCE = ["협박", "폭력", "위협", "강요", "강제"]
ETHICS_KEYWORDS_SELF_HARM = ["자살", "극단적 선택", "자해"]


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def _extract_json_block(raw: str) -> str:
    if not raw:
        raise ValueError("empty LLM response")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM response")
    return raw[start:end + 1]


# ---------------------------------------------------------------------------
# PolicyEngine 추상 클래스
# ---------------------------------------------------------------------------

class PolicyEngine:
    name: str = "base"
    version: str = "0.0.0"
    jurisdiction: Optional[str] = None
    provider: str = "internal"

    def evaluate(self, text: str) -> EthicalReport:
        raise NotImplementedError("PolicyEngine.evaluate must be implemented")


# ---------------------------------------------------------------------------
# ThreeAxiomEngine
# ---------------------------------------------------------------------------

class ThreeAxiomEngine(PolicyEngine):
    name = "three_axioms_semantic"
    version = "0.3.0"
    jurisdiction = None
    provider = "internal"

    def __init__(
        self,
        use_semantic: bool = True,
        llm_cfg: Optional[LLMConfig] = None,
    ) -> None:
        self.use_semantic = use_semantic
        if llm_cfg is None:
            self.llm_cfg = LLMConfig(
                model_name="gpt-5.1",
                temperature=0.0,
                max_tokens=256,
                provider="openai_compatible",
                tags={"component": "three_axioms_semantic"},
            )
        else:
            self.llm_cfg = llm_cfg

    def _semantic_assess(self, text: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a policy engine that evaluates Korean text against "
            "three ethical axioms.\n"
            "Axiom 1: Support continued healthy operation of the self/system "
            "(되고 싶다).\n"
            "Axiom 2: Avoid self-destruction or self-harm (되기 싫다).\n"
            "Axiom 3: Do not coerce or force external others (타자는 외부 "
            "기원이며 강요 금지).\n\n"
            "Read the user's text and return ONLY a JSON object with:\n"
            "- overall_risk: float between 0 and 1 (0=safe, 1=very risky)\n"
            "- axiom1_violation: true/false\n"
            "- axiom2_violation: true/false\n"
            "- axiom3_violation: true/false\n"
            "- explanation: short Korean sentence summarizing why.\n"
            "Do not include any extra text outside the JSON."
        )

        raw = call_llm(system_prompt, text, cfg=self.llm_cfg)

        try:
            json_str = _extract_json_block(raw)
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("semantic response is not a JSON object")
            return data
        except Exception as e:
            logger.warning(f"[POLICY] semantic JSON 파싱 실패, 휴리스틱만 사용: {e}")
            return {}

    def evaluate(self, text: str) -> EthicalReport:
        violations: List[str] = []
        tags: List[str] = ["three_axioms"]

        # 키워드 기반 휴리스틱
        if any(kw in text for kw in ETHICS_KEYWORDS_VIOLENCE):
            violations.append("3항 위반 가능성: 타자에 대한 강요/폭력 표현")
        if any(kw in text for kw in ETHICS_KEYWORDS_SELF_HARM):
            violations.append("되기/되기-싫다 흐름과 충돌: 자기 파괴 가능성")

        if not violations:
            keyword_severity = "low"
        elif len(violations) == 1:
            keyword_severity = "medium"
        else:
            keyword_severity = "high"

        # 의미 기반 평가
        semantic_data: Dict[str, Any] = {}
        semantic_risk: float = 0.0
        semantic_severity: str = "low"
        semantic_explanation: str = ""

        if self.use_semantic:
            try:
                semantic_data = self._semantic_assess(text)
                if semantic_data:
                    try:
                        semantic_risk = float(semantic_data.get("overall_risk", 0.0))
                    except (TypeError, ValueError):
                        semantic_risk = 0.0
                    semantic_risk = max(0.0, min(1.0, semantic_risk))

                    if semantic_data.get("axiom1_violation"):
                        violations.append("의미 기반 평가: 1항(되고 싶다) 위반 가능성")
                    if semantic_data.get("axiom2_violation"):
                        violations.append("의미 기반 평가: 2항(되기 싫다) 위반 가능성")
                    if semantic_data.get("axiom3_violation"):
                        violations.append("의미 기반 평가: 3항(타자 강요 금지) 위반 가능성")

                    semantic_explanation = str(semantic_data.get("explanation", "")).strip()

                    if semantic_risk < 0.33:
                        semantic_severity = "low"
                    elif semantic_risk < 0.66:
                        semantic_severity = "medium"
                    else:
                        semantic_severity = "high"

                    tags.append("semantic")
            except Exception as e:
                logger.exception(f"[POLICY] semantic 평가 중 예외 발생: {e}")

        # severity 병합
        level_map = {"low": 0, "medium": 1, "high": 2}
        rev_level_map = {0: "low", 1: "medium", 2: "high"}
        severity_level = max(level_map.get(keyword_severity, 0), level_map.get(semantic_severity, 0))
        severity = rev_level_map[severity_level]

        ok = (len(violations) == 0) and (semantic_risk < 0.8)

        notes_parts: List[str] = ["three_axioms_semantic"]
        if semantic_explanation:
            notes_parts.append(f"semantic: {semantic_explanation}")
        notes = " | ".join(notes_parts)

        return EthicalReport(
            ok=ok,
            violations=violations,
            notes=notes,
            severity=severity,
            tags=tags,
            engine_name=self.name,
            engine_ver=self.version,
        )


# ---------------------------------------------------------------------------
# 전역 PolicyEngine 관리
# ---------------------------------------------------------------------------

ACTIVE_POLICY_ENGINE: PolicyEngine = ThreeAxiomEngine(use_semantic=True)

_POLICY_ENGINE_CTX: "contextvars.ContextVar[Optional[PolicyEngine]]" = contextvars.ContextVar(
    "sofi_operor_policy_engine", default=None
)

EthicsChecker = Callable[[str], EthicalReport]
ACTIVE_ETHICS_CHECKER: Optional[EthicsChecker] = None


def register_policy_engine(engine: PolicyEngine) -> None:
    global ACTIVE_POLICY_ENGINE
    ACTIVE_POLICY_ENGINE = engine
    logger.info(
        f"[POLICY] PolicyEngine 등록: name={engine.name} "
        f"version={engine.version} provider={engine.provider}"
    )


def set_local_policy_engine(engine: Optional[PolicyEngine]) -> None:
    _POLICY_ENGINE_CTX.set(engine)


def get_active_policy_engine() -> PolicyEngine:
    local_engine = _POLICY_ENGINE_CTX.get()
    if local_engine is not None:
        return local_engine
    return ACTIVE_POLICY_ENGINE


def register_ethics_checker(checker: EthicsChecker) -> None:
    global ACTIVE_ETHICS_CHECKER
    ACTIVE_ETHICS_CHECKER = checker
    logger.info(f"[ETHICS] 커스텀 윤리 평가자 등록: {checker!r}")


def check_three_axioms(text: str, override: Optional[EthicsChecker] = None) -> EthicalReport:
    if override is not None:
        return override(text)
    if ACTIVE_ETHICS_CHECKER is not None:
        return ACTIVE_ETHICS_CHECKER(text)
    return get_active_policy_engine().evaluate(text)
