"""
run_all_tests.py — Runs the full test suite and writes a timestamped report to logs/test_results.log.

Each test module covers a specific concern (config, tools, middleware, model routing, voice).
Running them together makes it easy to spot regressions before committing.
"""
import subprocess
import os
import sys
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "test_results.log")

# Keep this list ordered by dependency depth — config first, server last.
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
