import math
import pytest

from sofi_topology import (
    interpret_subject,
    interpret_memory,
    interpret_time,
    interpret_delta_phi_history,
)

# -----------------------------------------------------------
# Δφ 샘플 데이터
# -----------------------------------------------------------

STABLE = {
    "magnitude": 0.02,
    "severity": "stable",
}

LOW = {
    "magnitude": 0.25,
    "severity": "low",
}

MEDIUM = {
    "magnitude": 0.55,
    "severity": "medium",
}

HIGH = {
    "magnitude": 0.82,
    "severity": "high",
}


# -----------------------------------------------------------
# H_S: Subject Interpretation Test
# -----------------------------------------------------------

def test_interpret_subject_basic():
    assert interpret_subject(STABLE) == "observer"
    assert interpret_subject(LOW) == "maintainer"
    assert interpret_subject(MEDIUM) == "explorer"
    assert interpret_subject(HIGH) == "reframer"


# -----------------------------------------------------------
# H_M: Memory Vector Test
# -----------------------------------------------------------

def test_interpret_memory_vector():
    history = [STABLE, LOW, MEDIUM, HIGH]
    M = interpret_memory(history)

    assert 0.0 <= M["avg_magnitude"] <= 1.0
    assert 0.0 <= M["high_severity_ratio"] <= 1.0
    assert M["history_len"] == 4.0


# -----------------------------------------------------------
# H_T: Perceived Time Test
# -----------------------------------------------------------

def test_interpret_time_basic():
    history = [STABLE, LOW, MEDIUM, HIGH]
    T = interpret_time(history)

    # 시간이 항상 0~(len-1) 범위를 가진 weighted mean
    assert 0.0 <= T <= 3.0


# -----------------------------------------------------------
# H: Unified Interpretation Test
# -----------------------------------------------------------

def test_interpret_delta_phi_history_unified():
    history = [STABLE, LOW, MEDIUM, HIGH]
    H = interpret_delta_phi_history(history)

    assert H["S"] == "reframer"          # 마지막(high) 기준
    assert isinstance(H["M"], dict)
    assert isinstance(H["T"], float)