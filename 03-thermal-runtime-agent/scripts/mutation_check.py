"""Mutation check: break the framework on purpose and confirm the tests notice.

A passing test suite only means something if it can fail. For each mutation this
runs the full suite with one deliberate bug injected (see tests/mutations.py)
and reports whether any test caught it. A mutation that no test catches is a
gap in the tests and is reported as SURVIVED.

Usage:  python scripts/mutation_check.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTATIONS = [
    ("agent_never_acts", "the agent decides but never applies a configuration"),
    ("evaluator_ignores_heat", "the evaluator always reports everything is fine"),
    ("model_never_fitted", "the thermal model is never identified from live data"),
    ("drift_never_detected", "the deployment monitor never detects drift"),
    ("step_up_jumps_to_top", "stepping up jumps straight to full power (unsafe exploration)"),
    ("coarse_state_ignored", "coarse thermal states are ignored on devices without temperature"),
    ("software_cap_counted_as_throttle", "the agent's own power cap is mistaken for hardware throttling"),
    ("back_off_one_level_only", "backing off is limited to one level at a time"),
    ("lowering_waits_for_dwell", "backing off has to wait for the dwell timer"),
]


def run(mut):
    env = dict(os.environ)
    env["MUT"] = mut
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "tests"), ROOT, env.get("PYTHONPATH", "")])
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "mutations", "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    m = re.search(r"(\d+) failed", tail)
    failed = int(m.group(1)) if m else 0
    return p.returncode != 0, failed, tail


def main():
    base = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    last = base.stdout.strip().splitlines()[-1] if base.stdout.strip() else ""
    print("Unmodified suite: " + last)
    if base.returncode != 0:
        print("The unmodified suite must pass before a mutation check means anything.")
        return 2
    print()
    print("%-34s %-9s %s" % ("mutation", "result", "tests that failed"))
    survived = 0
    for name, why in MUTATIONS:
        caught, failed, _tail = run(name)
        if not caught:
            survived += 1
        print("%-34s %-9s %d    (%s)" % (name, "CAUGHT" if caught else "SURVIVED", failed, why))
    print()
    print("%d of %d mutations caught" % (len(MUTATIONS) - survived, len(MUTATIONS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
