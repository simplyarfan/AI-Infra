"""Print the path each tick takes through the AutoOptAgent lifecycle.

Usage:  python scripts/run_lifecycle_demo.py [phone_quant_flat | phone_quant_cools]

Runs the agent on the simulated phone and prints, for the ticks where
something interesting happens, which stages ran and what they decided. This is
the quickest way to see the slide's boxes and arrows as running code.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from thermal_agent import AgentConfig, Objectives, ThermalAgent, make_phone_device  # noqa: E402


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "phone_quant_flat"
    obj = Objectives()
    dev = make_phone_device(scenario)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=obj))
    names = [k.name for k in agent.knobs]
    print("scenario: %s   knobs: %s" % (scenario, ", ".join(names)))
    print("thermal budget %.0f C (limit with margin %.0f C)\n" % (obj.max_temp_c, obj.limit_c))
    shown_quiet = False
    for _ in range(600):
        dev.advance(1.0)
        d = agent.tick()
        quiet = not d.applied and d.stages[-1] == "8 deploy and hold" and not d.drift
        if quiet:
            if not shown_quiet and d.t_s > 10:
                print("t=%4.0fs  %s" % (d.t_s, d.trace()))
                print("          (stable: the gate says no optimization is needed, Agent 4 is not called)\n")
                shown_quiet = True
            continue
        if d.applied or d.drift or d.advisor_vetoed:
            temp = dev.temp
            print("t=%4.0fs  temp %.1f C  %s" % (d.t_s, temp, d.trace()))
            print("          %s -> %s" % (
                ", ".join("%s=%d" % kv for kv in d.before.items()),
                ", ".join("%s=%d" % kv for kv in d.after.items())))
            print("          %s\n" % d.note)
    print("failed explorations: %d   re-optimizations: %d   final config: %s" % (
        agent.optimizer.failed_explorations, agent.reoptimizations, agent.transformer.current()))


if __name__ == "__main__":
    main()
