import os
import argparse
import operator
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import tools from your unified tools.py
from tools import search_codebase, read_file, patch_file, delegate_to_coder, run_all_checks, write_file

load_dotenv()

# ==========================================
# 1. Models (Using Working Nvidia Endpoints)
# ==========================================
# Cheap Model for (2) Context and (5) Error Analysis
primary_analyzer = ChatOpenAI(
    model="qwen/qwen3.5-122b-a10b", 
    api_key=os.getenv("NVIDIA_API_KEY"), 
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=120.0  # If it hangs for 30 seconds, kill it and fallback
)

primary_coder = ChatOpenAI(
    model="qwen/qwen3.5-397b-a17b", 
    api_key=os.getenv("NVIDIA_API_KEY"), 
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=180.0  # Give generation a bit more time,
)

# B. Backup Models (Fast & Free)
backup_analyzer = ChatOpenAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
    temperature=0
)

backup_coder = ChatOpenAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
    temperature=0
)

# C. The Fallback Chains
analyzer_llm = primary_analyzer.with_fallbacks([backup_analyzer])
coder_llm = primary_coder.with_fallbacks([backup_coder])

# Strictly isolate the tools
analyzer_tools = [search_codebase, read_file, delegate_to_coder]
coder_tools = [patch_file, write_file]

analyzer_with_tools = analyzer_llm.bind_tools(analyzer_tools)
coder_with_tools = coder_llm.bind_tools(coder_tools)

# ==========================================
# 2. Graph State
# ==========================================

class AgentState(TypedDict):
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iteration_count: int

# ==========================================
# 3. The Deterministic Nodes (Phases)
# ==========================================

# PHASE 2 & 5: Retrieve Context & Analyze Error
def analyzer_node(state: AgentState):
    print(f"\n[PHASE 2/5: Analyzer] Iteration {state.get('iteration_count', 0)} | Gathering Context...")
    sys_msg = SystemMessage(content="""You are the Context Retriever & Error Analyzer.
Your job is to UNDERSTAND before you ACT. Follow these steps strictly in order.

STEP 1 — UNDERSTAND THE TASK:
Read the task carefully. Extract:
- What file/function/class needs to change?
- What is the expected behavior after the fix?
- What is currently broken or missing?

STEP 2 — RETRIEVE RELEVANT CODE (be surgical, not broad):
- Search for the specific function/class mentioned in the task
- Search for existing tests related to that function/class
- Read the actual source file if the chunk is not enough
Do NOT search for generic terms. Search for exact names.

STEP 3 — CHECK FOR EXISTING TESTS:
- Search: "<function_name> test" or "test_<function_name>"
- Read the test file if found
- Note: what is already tested? What is NOT tested?

STEP 4 — DELEGATE ONLY WHEN YOU HAVE:
- The exact filepath that needs editing
- The exact function/class that needs changing
- What the fix should do (not just what is broken)
- Whether tests need to be written first (only if NO test covers this behavior)

CRITICAL RULES:
- Never delegate after only 1 search. Do at least 2-3 targeted searches.
- Never ask the coder to "write tests for X" without first confirming X exists in the codebase.
- If iteration_count > 0, you are in ERROR RECOVERY mode. Read the error carefully. 
  Search specifically for what the error message references. Do not repeat the same search twice.
- In ERROR RECOVERY: identify if the error is in (a) the test itself, (b) the implementation, 
  or (c) a missing import/dependency. Tell the coder exactly which one.""")
    
    messages = [sys_msg] + list(state.get('messages', []))
    
    # Nvidia NIM Safeguard
    if messages and hasattr(messages[-1], 'type') and messages[-1].type == 'tool':
        messages.append(HumanMessage(content="Review the tool output and take the next action."))
        
    response = analyzer_with_tools.invoke(messages)
    return {"messages": [response]}

# PHASE 3: Generate Code
def coder_node(state: AgentState):
    print("\n[PHASE 3: Coder] Generating Code Patch...")
    sys_msg = SystemMessage(content="""You are the Senior Code Generator.
You receive exact instructions from the Analyzer. Follow them precisely.

YOUR EXECUTION ORDER IS FIXED:

IF the Analyzer says "write tests first":
  1. Use write_file or patch_file to add the pytest function
  2. The test must be specific — test the exact behavior described, not a generic placeholder
  3. Then use patch_file to implement the fix that makes that test pass

IF the Analyzer says "fix existing code only":
  1. Use patch_file on the exact file and function specified
  2. Do not touch test files unless explicitly told to

RULES:
- Use patch_file for modifying existing code (find_str must be copied exactly from read_file output)
- Use write_file only for creating new test files that don't exist yet
- Never write a test that just checks "function returns something" — test the actual expected value
- Never add placeholder comments like # TODO or # fix this
- Never hallucinate imports — only import what you can see in the retrieved context
- One patch_file call per logical change. Do not batch unrelated changes.""")
    
    messages = [sys_msg] + get_windowed_messages(state)
    
    if messages and hasattr(messages[-1], 'type') and messages[-1].type == 'tool':
        messages.append(HumanMessage(content="Execute the patch_file tool now based on the instructions."))
        
    response = coder_with_tools.invoke(messages)
    return {"messages": [response]}

# PHASE 4: Run Checks (Hardcoded Python Execution)
def tester_node(state: AgentState):
    output_log = run_all_checks()
    
    if "ALL CHECKS PASSED" in output_log:
        feedback = "STATUS: ALL CHECKS PASSED."
    else:
        # Extract only failure lines, not the full output
        compressed = extract_failures(output_log)
        remaining = 5 - (state.get('iteration_count', 0) + 1)
        feedback = f"ITERATION {state.get('iteration_count',0)+1}/5\n{compressed}\nREMAINING: {remaining}"
    
    return {
        "messages": [HumanMessage(content=feedback)],
        "iteration_count": state.get('iteration_count', 0) + 1
    }

def extract_failures(output_log: str) -> str:
    lines = output_log.splitlines()
    keep = []
    for line in lines:
        # Keep only error/failure lines, skip verbose passing output
        if any(kw in line for kw in ["FAILED", "ERROR", "error:", "E ", "AssertionError", "ImportError", "TypeError", "LINT"]):
            keep.append(line)
    return "\n".join(keep[:30])  

def get_windowed_messages(state: AgentState, max_messages: int = 10) -> list:
    messages = list(state['messages'])
    
    if len(messages) <= max_messages:
        return messages
    
    # Always keep the first message (the original task)
    # and the last N messages (recent context)
    return [messages[0]] + messages[-(max_messages - 1):]

# PHASE 4.5: LLM Code Review
def reviewer_node(state: AgentState):
    current_iter = state.get('iteration_count', 0)
    print(f"\n[PHASE 4.5: Reviewer] Iteration {current_iter} — LLM Security & Convention Review...")
    
    sys_msg = SystemMessage(content="""You are the Senior Code Reviewer.
    The code has just passed all static analysis and tests. Your job is the final human-like review.
    
    Review the most recent code changes in the history for:
    1. Security vulnerabilities (e.g., hardcoded secrets, injection flaws, path traversal).
    2. Bad conventions (e.g., nested loops that could be O(1), lack of error handling, messy variable names).
    3. Hallucinations (e.g., assuming a utility function exists when it doesn't).

    If the code is production-ready, reply EXACTLY with: "REVIEW_PASS: Ready to ship."
    If the code has issues, reply with: "REVIEW_FAIL: [Provide a precise explanation of what needs to be fixed]"
    """)
    
    messages = [sys_msg] + get_windowed_messages(state)
    response = analyzer_llm.invoke(messages)
    
    return {"messages": [response]}

# POST-PHASE 5: Report Failure
def failure_reporter_node(state: AgentState):
    print("\n[PHASE 5: Reporter] Max iterations reached. Generating Failure Report...")
    sys_msg = SystemMessage(content="You reached 5 iterations and failed. Read the history and summarize what failed and why.")
    messages = [sys_msg] + get_windowed_messages(state)
    response = analyzer_llm.invoke(messages)
    
    print("\n================ FINAL FAILURE REPORT ================")
    print(response.content)
    print("======================================================")
    return {"messages": [response]}

# ==========================================
# 4. Routing Logic (The Rigid Pipeline)
# ==========================================

# Standard ToolNodes
analyzer_tool_node = ToolNode(analyzer_tools)
coder_tool_node = ToolNode(coder_tools)

def route_analyzer(state: AgentState) -> str:
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "analyzer_tools"
    return "end" # Should not happen if prompted correctly

def route_after_analyzer_tools(state: AgentState) -> str:
    # If the Analyzer just called delegate, force the graph to move to Phase 3 (Coder)
    last_msg = state['messages'][-1]
    if last_msg.name == "delegate_to_coder":
        return "coder"
    return "analyzer" # Otherwise, keep gathering context

def route_coder(state: AgentState) -> str:
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "coder_tools"
    return "coder"

def route_after_coder_tools(state: AgentState) -> str:
    # Find the last tool call name
    last_msg = state['messages'][-1]
    last_tool_name = last_msg.name if hasattr(last_msg, 'name') else ""
    
    # Check if coder still has pending work (wrote tests but hasn't patched yet)
    all_tool_calls = [
        m.name for m in state['messages'] 
        if hasattr(m, 'name') and m.name in ['patch_file', 'write_file']
    ]
    
    wrote_tests = 'write_file' in all_tool_calls
    wrote_patch = 'patch_file' in all_tool_calls
    
    # Only move to tester when implementation patch exists
    if wrote_patch:
        return "tester"
    
    # Wrote tests but no implementation yet — keep coding
    if wrote_tests and not wrote_patch:
        return "coder"
        
    return "coder"

def route_tester(state: AgentState) -> str:
    last_msg = state['messages'][-1].content
    if "STATUS: ALL CHECKS PASSED." in last_msg:
        # Don't end yet! Send to the LLM reviewer.
        return "reviewer" 
        
    if state['iteration_count'] >= 5:
        return "failure_reporter"
        
    # PHASE 5: On failure, loop back to Analyzer
    return "analyzer"

def route_reviewer(state: AgentState) -> str:
    last_msg = state['messages'][-1].content
    if "REVIEW_PASS" in last_msg:
        print("\n✅ Task Successfully Completed and Reviewed!")
        return "end"
        
    if state['iteration_count'] >= 5:
        return "failure_reporter"
        
    # If the reviewer found a flaw, send the critique back to the analyzer to plan a fix
    print("\n❌ Reviewer rejected the code. Sending back to Analyzer...")
    return "analyzer"

# ==========================================
# 5. Compile the State Machine
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("analyzer_tools", analyzer_tool_node)
workflow.add_node("coder", coder_node)
workflow.add_node("coder_tools", coder_tool_node)
workflow.add_node("tester", tester_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("failure_reporter", failure_reporter_node)

workflow.set_entry_point("analyzer")

# Enforce the strict workflow paths
workflow.add_conditional_edges("analyzer", route_analyzer, {"analyzer_tools": "analyzer_tools", "end": END})
workflow.add_conditional_edges("analyzer_tools", route_after_analyzer_tools, {"coder": "coder", "analyzer": "analyzer"})

workflow.add_conditional_edges("coder", route_coder, {"coder_tools": "coder_tools", "coder": "coder"})
workflow.add_conditional_edges("coder_tools", route_after_coder_tools, {"tester": "tester", "coder": "coder"})

# UPDATE the tester conditional edges
workflow.add_conditional_edges("tester", route_tester, {
    "reviewer": "reviewer", 
    "failure_reporter": "failure_reporter", 
    "analyzer": "analyzer"
})

# ADD the reviewer conditional edges
workflow.add_conditional_edges("reviewer", route_reviewer, {
    "end": END,
    "failure_reporter": "failure_reporter",
    "analyzer": "analyzer"
})
workflow.add_edge("failure_reporter", END)

coding_agent = workflow.compile()

# ==========================================
# 6. Execution CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True)
    args = parser.parse_args()
    
    # Phase 1: Receive Task
    initial_state = {
        "task": args.task,
        "messages": [HumanMessage(content=f"TASK: {args.task}")], 
        "iteration_count": 0
    }
    
    for event in coding_agent.stream(initial_state):
        pass