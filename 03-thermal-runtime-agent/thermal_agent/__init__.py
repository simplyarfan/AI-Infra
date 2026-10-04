"""Thermal aware runtime optimization agent.

A runtime (closed loop) extension of an agentic optimizer with a thermal
objective, built around the idea that on edge devices the optimizer has to
watch temperature, not only throughput and energy.
"""
from .dataset import ConfigPerfDataset, Record
from .evaluation import Assessment, BaselineAgent, EvaluationAgent
from .harness import SessionMetrics, run_comparison, run_session
from .knobs import Knob, KnobLevel, knob_from_sweep, power_cap_knob
from .objectives import Objectives
from .observer import Observer
from .predictor import ConfigPredictor
from .runtime import AgentConfig, Decision, ThermalAgent
from .sim import PRESETS, SimParams, SimulatedDevice
from .thermal_model import ThermalParams, fit_thermal_model
from .transformer import DeploymentMonitor, Transformer
from .types import Observation, THERMAL_STATES

__all__ = [
    "ConfigPerfDataset", "Record", "Assessment", "BaselineAgent", "EvaluationAgent",
    "SessionMetrics", "run_comparison", "run_session", "Knob", "KnobLevel", "knob_from_sweep",
    "power_cap_knob", "Objectives", "Observer", "ConfigPredictor", "AgentConfig", "Decision",
    "ThermalAgent", "PRESETS", "SimParams", "SimulatedDevice", "ThermalParams",
    "fit_thermal_model", "DeploymentMonitor", "Transformer", "Observation", "THERMAL_STATES",
]
