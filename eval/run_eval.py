import json
import subprocess
import time
import csv
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

TASK_FILE = "eval/tasks.json"

REPORT_DIR = "eval/reports"

REPORT_FILE = f"{REPORT_DIR}/eval_report.csv"

# Ensure report directory exists
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD TASKS
# ==========================================

with open(TASK_FILE, "r", encoding="utf-8") as f:
    tasks = json.load(f)

# ==========================================
# CSV SETUP
# ==========================================

fieldnames = [
    "task_id",
    "task",
    "expected",
    "status",
    "iterations",
    "runtime_sec",
    "recovered_after_failure",
    "log_file",
    "diff_file"
]

# Create CSV with headers
with open(REPORT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# ==========================================
# HELPERS
# ==========================================

def run_cmd(cmd, timeout=120):
    """
    Safe subprocess helper
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )

def reset_repo():
    """
    Restore clean git state between tasks
    """
    print("\n[RESET] Cleaning repository state...")

    run_cmd(["git", "reset", "--hard"])
    run_cmd(["git", "clean", "-fd"])

    # git clean deletes generated folders
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

def save_diff(task_id):
    """
    Save git diff for current task
    """
    diff_path = f"{REPORT_DIR}/{task_id}_diff.patch"

    result = run_cmd(["git", "diff"])

    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    return diff_path

# ==========================================
# EVAL LOOP
# ==========================================

for task in tasks:

    print("\n===================================")
    print(f"Running {task['id']}")
    print(task["task"])
    print("===================================")

    # --------------------------------------
    # RESET BEFORE TASK
    # --------------------------------------

    reset_repo()

    start = time.time()

    try:

        # --------------------------------------
        # RUN AGENT
        # --------------------------------------

        result = subprocess.run(
            [
                "python",
                "main.py",
                "--task",
                task["task"]
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=900
        )

        runtime = round(time.time() - start, 2)

        output = result.stdout + "\n" + result.stderr

        # --------------------------------------
        # SAVE RAW LOG
        # --------------------------------------

        log_path = f"{REPORT_DIR}/{task['id']}_log.txt"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output)

        # --------------------------------------
        # SAVE GIT DIFF
        # --------------------------------------

        diff_path = save_diff(task["id"])

        # --------------------------------------
        # PARSE STATUS
        # --------------------------------------

        if "✅ Task Successfully Completed and Reviewed!" in output:
            status = "PASS"

        elif "FINAL FAILURE REPORT" in output:
            status = "SAFE_FAILURE"

        elif "STATUS: ALL CHECKS PASSED." in output:
            status = "PARTIAL_PASS"

        else:
            status = "FAIL"

        # --------------------------------------
        # ITERATION COUNT
        # --------------------------------------

        iterations = output.count("[PHASE 4")

        # --------------------------------------
        # RECOVERY DETECTION
        # --------------------------------------

        recovered = (
            iterations > 1 and status in ["PASS", "PARTIAL_PASS"]
        )

        # --------------------------------------
        # BUILD RESULT ROW
        # --------------------------------------

        row = {
            "task_id": task["id"],
            "task": task["task"],
            "expected": task["expected"],
            "status": status,
            "iterations": iterations,
            "runtime_sec": runtime,
            "recovered_after_failure": recovered,
            "log_file": log_path,
            "diff_file": diff_path
        }

        # --------------------------------------
        # SAVE ROW IMMEDIATELY
        # --------------------------------------

        with open(REPORT_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

        # --------------------------------------
        # CONSOLE OUTPUT
        # --------------------------------------

        print(f"STATUS: {status}")
        print(f"ITERATIONS: {iterations}")
        print(f"RUNTIME: {runtime}s")

    except subprocess.TimeoutExpired:

        print("STATUS: TIMEOUT")

        row = {
            "task_id": task["id"],
            "task": task["task"],
            "expected": task["expected"],
            "status": "TIMEOUT",
            "iterations": 0,
            "runtime_sec": 900,
            "recovered_after_failure": False,
            "log_file": "TIMEOUT",
            "diff_file": "TIMEOUT"
        }

        with open(REPORT_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

    except Exception as e:

        print(f"STATUS: ERROR -> {e}")

        row = {
            "task_id": task["id"],
            "task": task["task"],
            "expected": task["expected"],
            "status": f"ERROR: {str(e)}",
            "iterations": 0,
            "runtime_sec": 0,
            "recovered_after_failure": False,
            "log_file": "ERROR",
            "diff_file": "ERROR"
        }

        with open(REPORT_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

    finally:

        # --------------------------------------
        # RESET AFTER TASK
        # --------------------------------------

        reset_repo()

# ==========================================
# DONE
# ==========================================

print("\n✅ Sequential Eval Complete")
print(f"Report saved to: {REPORT_FILE}")