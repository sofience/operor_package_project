"""
sofience_operor 패키지
"""

from sofi_llm import (
    LLMConfig,
    LLMError,
    LLMCache,
    LLMCacheConfig,
    call_llm,
    register_llm_hook,
)

from sofi_policy import (
    ROOT_PROPOSITION,
    EthicalReport,
    PolicyEngine,
    ThreeAxiomEngine,
    check_three_axioms,
    register_policy_engine,
    set_local_policy_engine,
    get_active_policy_engine,
)

from sofi_topology import (
    PhaseState,
    PhaseVector,
    TraceEntry,
    TraceLog,
    OperorRuntime,
    DEFAULT_RUNTIME,
    GLOBAL_TRACE_LOG,
    DeltaPhiObserver,
    register_delta_phi_observer,
    compute_phi_core,
    compute_phi_surface,
    compute_void_state,
    compute_delta_phi_vector,
)

from sofi_context import (
    Context,
    Goal,
    PlanCandidate,
    ScoredPlan,
    build_context,
    compose_goal,
    propose_plans,
    score_alignment,
    explore_alignment,
    maybe_abort_or_select,
)

from sofi_channels import (
    ChannelConfig,
    ChannelName,
    DEFAULT_CHANNELS,
    run_channel,
    execute_channels_parallel,
    aggregate_channels,
)

from sofi_agent import (
    agent_step,
    recursive_alignment_search,
    refine_goal_for_alignment,
)

__all__ = [
    # LLM
    "LLMConfig", "LLMError", "LLMCache", "LLMCacheConfig",
    "call_llm", "register_llm_hook",
    # Policy
    "ROOT_PROPOSITION", "EthicalReport", "PolicyEngine", "ThreeAxiomEngine",
    "check_three_axioms", "register_policy_engine",
    "set_local_policy_engine", "get_active_policy_engine",
    # Topology
    "PhaseState", "PhaseVector", "TraceEntry", "TraceLog",
    "OperorRuntime", "DEFAULT_RUNTIME", "GLOBAL_TRACE_LOG",
    "DeltaPhiObserver", "register_delta_phi_observer",
    "compute_phi_core", "compute_phi_surface", "compute_void_state", "compute_delta_phi_vector",
    # Context
    "Context", "Goal", "PlanCandidate", "ScoredPlan",
    "build_context", "compose_goal", "propose_plans",
    "score_alignment", "explore_alignment", "maybe_abort_or_select",
    # Channels
    "ChannelConfig", "ChannelName", "DEFAULT_CHANNELS",
    "run_channel", "execute_channels_parallel", "aggregate_channels",
    # Agent
    "agent_step", "recursive_alignment_search", "refine_goal_for_alignment",
]
