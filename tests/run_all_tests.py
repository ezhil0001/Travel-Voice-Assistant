"""Runs the full test suite and writes a timestamped report to logs/test_results.log.

Each module targets a specific layer — config first (fast, no mocks), then tools,
middleware, model routing, voice, agent routing, graph integration, and finally
the HTTP endpoints. Running them in dependency order makes failure attribution easier.
"""
import subprocess
import os
import sys
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "test_results.log")

# Ordered by dependency depth so a failure in config immediately explains
# failures lower down without having to chase unrelated errors.
TEST_MODULES = [
    "tests/test_config_and_env.py",
    "tests/test_api_tools.py",
    "tests/test_middleware_pipeline.py",
    "tests/test_model_routing.py",
    "tests/test_voice_pipeline.py",
    "tests/test_agent_routing.py",
    "tests/test_graph_routing.py",
    "tests/test_server_endpoints.py",
]


def run_all():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "w") as log:
        log.write(f"=== Test Run: {datetime.now().isoformat()} ===\n\n")

        all_passed = True
        for test_file in TEST_MODULES:
            log.write(f"--- {test_file} ---\n")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
            )
            log.write(result.stdout)
            log.write(result.stderr)
            log.write("\n")

            if result.returncode != 0:
                all_passed = False

        status = "ALL TESTS PASSED ✅" if all_passed else "SOME TESTS FAILED ❌"
        log.write(f"\n=== {status} ===\n")
        print(f"\n{status}  →  see {LOG_PATH}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
