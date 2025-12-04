"""
sofi_llm.py — LLM 클라이언트 & 캐시 레이어

역할:
- 외부 LLM 호출, 에러 타입, 후크, 캐시, provider dispatch
- 이 모듈은 Sofience 개념을 모르고, 순수 "LLM 서비스 클라이언트"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple, Callable
from collections import OrderedDict
import time
import json
import logging
import contextvars

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

logger = logging.getLogger("sofi-operor.llm")

# ---------------------------------------------------------------------------
# httpx 선택적 의존성
# ---------------------------------------------------------------------------

try:
    import httpx
except ImportError:
    httpx = None

# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model_name: str = "gpt-5.1"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 60
    provider: Literal["echo", "openai_compatible", "ollama"] = "echo"
    base_url: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    # 프로덕션용 제어 파라미터
    max_retry_cold: int = 3
    backoff_initial: float = 0.5
    backoff_multiplier: float = 2.0
    max_retry_warm: int = 1
    fallback_providers: List[Literal["echo", "openai_compatible", "ollama"]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLMError
# ---------------------------------------------------------------------------

@dataclass
class LLMError(Exception):
    provider: str
    attempt: int
    message: str
    cause: Optional[Exception] = None
    retryable: bool = True

    def __str__(self) -> str:
        return (
            f"LLMError(provider={self.provider!r}, attempt={self.attempt}, "
            f"retryable={self.retryable}, message={self.message})"
        )


# ---------------------------------------------------------------------------
# LLM Hook 시스템
# ---------------------------------------------------------------------------

LLMHook = Callable[[Dict[str, Any]], None]

_GLOBAL_LLM_HOOKS: List[LLMHook] = []

_LLM_HOOKS_CTX: "contextvars.ContextVar[Optional[List[LLMHook]]]" = (
    contextvars.ContextVar("sofi_operor_llm_hooks", default=None)
)


def _get_llm_hooks() -> List[LLMHook]:
    hooks = _LLM_HOOKS_CTX.get()
    if hooks is not None:
        return hooks
    return _GLOBAL_LLM_HOOKS


def set_llm_hooks(hooks: Optional[List[LLMHook]]) -> None:
    _LLM_HOOKS_CTX.set(list(hooks) if hooks is not None else [])


def register_llm_hook(hook: LLMHook, *, local: bool = True) -> None:
    if local:
        hooks = _LLM_HOOKS_CTX.get()
        if hooks is None:
            hooks = []
        else:
            hooks = list(hooks)
        hooks.append(hook)
        _LLM_HOOKS_CTX.set(hooks)
    else:
        _GLOBAL_LLM_HOOKS.append(hook)
    logger.info(f"[LLM] hook 등록: {hook!r} (local={local})")


def _emit_llm_event(event: Dict[str, Any]) -> None:
    for hook in _get_llm_hooks():
        try:
            hook(event)
        except Exception as e:
            logger.exception(f"[LLM] hook 호출 중 오류: {e}")


# ---------------------------------------------------------------------------
# LLM 캐시
# ---------------------------------------------------------------------------

@dataclass
class LLMCacheConfig:
    enabled: bool = True
    max_entries: int = 512
    ttl_sec: Optional[int] = 300


class LLMCache:
    def __init__(self, cfg: Optional[LLMCacheConfig] = None) -> None:
        self.cfg = cfg or LLMCacheConfig()
        self.store: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

    def make_key(
        self,
        provider: str,
        cfg: LLMConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        base = {
            "provider": provider,
            "model": cfg.model_name,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "system": system_prompt,
            "user": user_prompt,
            "tags": cfg.tags.get("cache_hint") if cfg.tags else None,
        }
        return json.dumps(base, ensure_ascii=False, sort_keys=True)

    def get(self, key: str) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        entry = self.store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if self.cfg.ttl_sec is not None:
            if time.time() - ts > self.cfg.ttl_sec:
                try:
                    del self.store[key]
                except KeyError:
                    pass
                return None
        self.store.move_to_end(key, last=True)
        return value

    def set(self, key: str, value: str) -> None:
        if not self.cfg.enabled:
            return
        now = time.time()
        if key in self.store:
            self.store.move_to_end(key, last=True)
        self.store[key] = (now, value)
        while len(self.store) > self.cfg.max_entries:
            self.store.popitem(last=False)


GLOBAL_LLM_CACHE = LLMCache()
_LLM_CACHE_POOLS: Dict[str, LLMCache] = {"default": GLOBAL_LLM_CACHE}


def get_llm_cache(cfg: LLMConfig) -> LLMCache:
    ns = "default"
    if cfg.tags and isinstance(cfg.tags.get("cache_ns"), str):
        ns = cfg.tags["cache_ns"]
    cache = _LLM_CACHE_POOLS.get(ns)
    if cache is None:
        cache = LLMCache(cfg=LLMCacheConfig(
            enabled=GLOBAL_LLM_CACHE.cfg.enabled,
            max_entries=GLOBAL_LLM_CACHE.cfg.max_entries,
            ttl_sec=GLOBAL_LLM_CACHE.cfg.ttl_sec,
        ))
        _LLM_CACHE_POOLS[ns] = cache
    return cache


# ---------------------------------------------------------------------------
# Provider 구현
# ---------------------------------------------------------------------------

def _call_llm_echo(system_prompt: str, user_prompt: str, cfg: LLMConfig) -> str:
    ts = int(time.time())
    snippet = user_prompt[:280].replace("\n", " ")
    logger.debug(
        f"[LLM ECHO] model={cfg.model_name} temp={cfg.temperature} "
        f"max_tokens={cfg.max_tokens} prompt_snippet={snippet!r}"
    )
    return f"[LLM-ECHO:{ts}] {user_prompt[:400]}"


def _call_llm_openai_compatible(system_prompt: str, user_prompt: str, cfg: LLMConfig) -> str:
    if httpx is None:
        logger.warning("httpx 미설치: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not api_key or not base_url:
        logger.warning("OPENAI_API_KEY/OPENAI_BASE_URL 미설정: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    payload = {
        "model": cfg.model_name,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception(f"[LLM ERROR] OpenAI compatible 호출 실패: {e}")
        return _call_llm_echo(system_prompt, user_prompt, cfg)


def _call_llm_ollama(system_prompt: str, user_prompt: str, cfg: LLMConfig) -> str:
    if httpx is None:
        logger.warning("httpx 미설치: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    base_url = cfg.base_url or "http://localhost:11434"
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            resp = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": cfg.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens,
                    },
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        logger.exception(f"[LLM ERROR] Ollama 호출 실패: {e}")
        return _call_llm_echo(system_prompt, user_prompt, cfg)


# ---------------------------------------------------------------------------
# call_llm 메인 엔트리포인트
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, cfg: Optional[LLMConfig] = None) -> str:
    if cfg is None:
        cfg = LLMConfig()

    def _dispatch(provider: str) -> str:
        if provider == "echo":
            return _call_llm_echo(system_prompt, user_prompt, cfg)
        elif provider == "openai_compatible":
            return _call_llm_openai_compatible(system_prompt, user_prompt, cfg)
        elif provider == "ollama":
            return _call_llm_ollama(system_prompt, user_prompt, cfg)
        else:
            logger.warning(f"알 수 없는 provider={provider!r}, echo 모드로 대체.")
            return _call_llm_echo(system_prompt, user_prompt, cfg)

    def _max_retry_for(provider: str) -> int:
        if provider == "echo":
            return cfg.max_retry_warm
        return cfg.max_retry_cold

    use_cache: bool = True
    if cfg.tags.get("no_cache") is True:
        use_cache = False

    cache = get_llm_cache(cfg)

    def _make_cache_key(provider: str) -> str:
        return cache.make_key(provider, cfg, system_prompt, user_prompt)

    provider_chain: List[str] = []
    if cfg.provider:
        provider_chain.append(cfg.provider)
    for p in cfg.fallback_providers:
        if p not in provider_chain:
            provider_chain.append(p)

    last_error: Optional[LLMError] = None

    for provider in provider_chain:
        is_cold_path = provider != "echo"
        max_retry = max(1, _max_retry_for(provider))
        backoff = cfg.backoff_initial

        for attempt in range(1, max_retry + 1):
            started_at = time.time()
            cache_key: Optional[str] = None

            if use_cache and cache.cfg.enabled:
                cache_key = _make_cache_key(provider)
                cached = cache.get(cache_key)
                if cached is not None:
                    elapsed = time.time() - started_at
                    _emit_llm_event({
                        "provider": provider,
                        "model": cfg.model_name,
                        "success": True,
                        "attempt": attempt,
                        "latency_sec": round(elapsed, 3),
                        "path": "cold" if is_cold_path else "warm",
                        "tags": dict(cfg.tags),
                        "error": None,
                        "from_cache": True,
                    })
                    return cached

            try:
                text = _dispatch(provider)
                elapsed = time.time() - started_at

                if use_cache and cache.cfg.enabled:
                    if cache_key is None:
                        cache_key = _make_cache_key(provider)
                    cache.set(cache_key, text)

                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": True,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": None,
                    "from_cache": False,
                })

                return text

            except Exception as e:
                elapsed = time.time() - started_at
                err = LLMError(
                    provider=provider,
                    attempt=attempt,
                    message=str(e),
                    cause=e,
                    retryable=is_cold_path,
                )
                last_error = err

                logger.error(
                    f"[LLM ERROR] provider={provider} attempt={attempt}/{max_retry} "
                    f"path={'cold' if is_cold_path else 'warm'} error={e}"
                )

                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": False,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": {"type": type(e).__name__, "message": str(e)},
                    "from_cache": False,
                })

                if attempt < max_retry and err.retryable:
                    time.sleep(backoff)
                    backoff *= cfg.backoff_multiplier
                    continue
                else:
                    break

    logger.warning("[LLM] 모든 provider 체인 호출 실패 → echo fallback 시도")

    try:
        started_at = time.time()
        text = _call_llm_echo(system_prompt, user_prompt, cfg)
        elapsed = time.time() - started_at

        _emit_llm_event({
            "provider": "echo",
            "model": cfg.model_name,
            "success": True,
            "attempt": 1,
            "latency_sec": round(elapsed, 3),
            "path": "warm",
            "tags": dict(cfg.tags),
            "error": None,
            "from_cache": False,
        })

        return text

    except Exception as e:
        logger.exception(f"[LLM] echo fallback까지 실패: {e}")
        raise last_error or LLMError(
            provider="echo",
            attempt=1,
            message=str(e),
            cause=e,
            retryable=False,
        )
