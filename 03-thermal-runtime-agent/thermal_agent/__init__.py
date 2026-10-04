"""Thermal aware runtime optimization agent.

A runtime (closed loop) extension of an agentic optimizer with a thermal
objective, built around the idea that on edge devices the optimizer has to
watch temperature, not only throughput and energy.
"""
from .dataset import ConfigPerfDataset, Record
from .evaluation import Assessment, BaselineAgent, EvaluationAgent, known_good_config
from .harness import SessionMetrics, objective_value, oracle_static_levels, run_comparison, run_model_level_comparison, run_session
from .knobs import (
    Knob, KnobLevel, knob_from_sweep, kv_precision_knob, power_cap_knob, quantization_knob,
    token_pacing_knob,
)
from .objectives import Objectives
from .observer import Observer
from .optimizer import Advisor, OptimizationReasoningAgent, Plan
from .predictor import ConfigPredictor
from .runtime import AgentConfig, Decision, ThermalAgent
from .sim import PHONE_SCENARIOS, PRESETS, SimParams, SimulatedDevice, make_phone_device
from .surrogate import Surrogate
from .thermal_model import ThermalParams, fit_thermal_model
from .transformer import DeploymentMonitor, Transformer
from .types import Observation, THERMAL_STATES

__all__ = [
    "ConfigPerfDataset", "Record", "Assessment", "BaselineAgent", "EvaluationAgent",
    "SessionMetrics", "run_comparison", "run_session", "Knob", "KnobLevel", "knob_from_sweep",
    "power_cap_knob", "Objectives", "Observer", "ConfigPredictor", "AgentConfig", "Decision",
    "ThermalAgent", "PRESETS", "SimParams", "SimulatedDevice", "ThermalParams",
    "fit_thermal_model", "DeploymentMonitor", "Transformer", "Observation", "THERMAL_STATES",
    "objective_value", "oracle_static_levels", "run_model_level_comparison", "kv_precision_knob", "quantization_knob",
    "token_pacing_knob", "Advisor", "OptimizationReasoningAgent", "Plan", "PHONE_SCENARIOS",
    "make_phone_device", "Surrogate", "known_good_config",
]
