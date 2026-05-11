import json
import subprocess
import time
import csv
from pathlib import Path

TASK_FILE = "eval/tasks.json"
REPORT_FILE = "eval/reports/eval_report.csv"

Path("eval/reports").mkdir(parents=True, exist_ok=True)

with open(TASK_FILE, "r") as f:
    tasks = json.load(f)

results = []

for task in tasks:
    print(f"\n=== Running {task['id']} ===")

    start = time.time()

    try:
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
            timeout=600
        )

        runtime = round(time.time() - start, 2)

        output = result.stdout + "\n" + result.stderr

        # -----------------------------
        # PARSE STATUS
        # -----------------------------

        if "✅ Task Successfully Completed!" in output:
            status = "PASS"

        elif "FINAL FAILURE REPORT" in output:
            status = "SAFE_FAILURE"

        else:
            status = "FAIL"

        # -----------------------------
        # PARSE ITERATIONS
        # -----------------------------

        iterations = 0

        for line in output.splitlines():
            if "[PHASE 4: Tester]" in line:
                iterations += 1

        # -----------------------------
        # DETECT RECOVERY
        # -----------------------------

        recovered = (
            iterations > 1 and status == "PASS"
        )

        results.append({
            "task_id": task["id"],
            "task": task["task"],
            "expected": task["expected"],
            "status": status,
            "iterations": iterations,
            "runtime_sec": runtime,
            "recovered_after_failure": recovered
        })

        print(f"STATUS: {status}")

    except subprocess.TimeoutExpired:
        results.append({
            "task_id": task["id"],
            "task": task["task"],
            "expected": task["expected"],
            "status": "TIMEOUT",
            "iterations": 0,
            "runtime_sec": 600,
            "recovered_after_failure": False
        })

# -----------------------------
# SAVE CSV REPORT
# -----------------------------

with open(REPORT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = results[0].keys()

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in results:
        writer.writerow(row)

print("\n✅ Eval Complete")
print(f"Report saved to: {REPORT_FILE}")