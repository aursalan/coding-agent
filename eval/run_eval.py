import json
import subprocess
import time
import sys
from pathlib import Path

from langsmith import Client

# ==========================================
# CONFIG
# ==========================================

client = Client()

REPORT_DIR = Path("eval/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TASK_FILE = "eval/tasks.json"

with open(TASK_FILE, "r", encoding="utf-8") as f:
    tasks = json.load(f)

PYTHON_EXECUTABLE = sys.executable

all_results = []

# ==========================================
# TASK LOOP
# ==========================================

for index, task in enumerate(tasks, start=1):

    print("\n===================================")
    print(f"TASK {index}/{len(tasks)}")
    print(f"ID: {task['id']}")
    print(f"TASK: {task['task']}")
    print("===================================\n")

    start = time.time()

    try:

        result = subprocess.run(
            [
                PYTHON_EXECUTABLE,
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
        # EXTRACT RUN ID
        # --------------------------------------

        run_id = None

        for line in output.splitlines():
            if line.startswith("LANGSMITH_RUN_ID="):
                run_id = line.split("=")[1].strip()
                break

        # --------------------------------------
        # FETCH LANGSMITH DATA
        # --------------------------------------

        total_tokens = 0
        trace_url = "N/A"

        if run_id:

            project_name = f"bindu-eval-{run_id}"

            runs = list(client.list_runs(
                project_name=project_name
            ))

            if runs:

                root_run = runs[0]

                total_tokens = (
                    root_run.total_tokens
                    if hasattr(root_run, "total_tokens")
                    else 0
                )

                trace_url = (
                    f"https://smith.langchain.com/public/"
                    f"{root_run.id}"
                )

        # --------------------------------------
        # STATUS
        # --------------------------------------

        if "✅ Task Successfully Completed and Reviewed!" in output:
            status = "PASS"

        elif "FINAL FAILURE REPORT" in output:
            status = "SAFE_FAILURE"

        else:
            status = "FAIL"

        # --------------------------------------
        # ITERATIONS
        # --------------------------------------

        iterations = output.count("[PHASE 4")

        task_result = {
            "task_id": task["id"],
            "task": task["task"],
            "status": status,
            "iterations": iterations,
            "runtime_sec": runtime,
            "tokens_burned": total_tokens,
            "trace_link": trace_url
        }

        all_results.append(task_result)

        # --------------------------------------
        # TERMINAL REPORT
        # --------------------------------------

        print("\n========== TASK REPORT ==========")
        print(f"STATUS: {status}")
        print(f"ITERATIONS: {iterations}")
        print(f"RUNTIME: {runtime}s")
        print(f"TOKENS: {total_tokens}")
        print(f"TRACE: {trace_url}")
        print("=================================\n")

    except subprocess.TimeoutExpired:

        print("\n⏰ TASK TIMED OUT")

# ==========================================
# SAVE FINAL REPORT
# ==========================================

report_path = REPORT_DIR / "eval_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\n===================================")
print("✅ ALL EVAL TASKS COMPLETED")
print(f"Saved report to: {report_path}")
print("===================================")