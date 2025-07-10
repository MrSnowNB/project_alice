import json
import requests
import sys
import os
import time  # For backoff in retries (Phase 3)
import yaml  # For YAML frontmatter parsing (Phase 3)
import click
import importlib.util
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage

from src.state import AgentState
from src.tools import (
    retrieve_from_memory, 
    search_the_web, 
    write_file, 
    execute_script, 
    run_shell_command, 
    add_to_memory, 
    request_human_assistance
)

# --- Dynamic Tool Loading with Hybrid Metadata (Phase 3) ---
TOOL_DIR = Path(project_root) / "src" / "generated_tools"
MANIFEST_FILE = TOOL_DIR / "manifest.yaml"  # Central manifest for quick loading

def extract_frontmatter(file_path: Path) -> dict:
    """Extracts YAML frontmatter from a .py file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if content.startswith('---\n'):
        end_idx = content.find('\n---\n', 4)
        if end_idx != -1:
            yaml_str = content[4:end_idx]
            try:
                return yaml.safe_load(yaml_str) or {}
            except yaml.YAMLError:
                return {}
    return {}

def update_manifest(metadata: dict):
    """Updates or creates the manifest with tool metadata."""
    TOOL_DIR.mkdir(exist_ok=True)
    manifest = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f) or {}
    tool_name = metadata.get('name')
    if tool_name:
        manifest[tool_name] = metadata
        with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(manifest, f)

def load_generated_tools(plan_text: str = "") -> list:
    """Dynamically loads tools from generated_tools dir, filtered by plan relevance using hybrid metadata."""
    TOOL_DIR.mkdir(exist_ok=True)
    (TOOL_DIR / "__init__.py").touch(exist_ok=True)
    
    # Load manifest for quick metadata scan (efficient single-file read)
    metadata_map = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            metadata_map = yaml.safe_load(f) or {}
    
    # Prune stale tools (e.g., last_used >30 days; optional, configurable)
    import datetime
    today = datetime.date.today()
    for tool_name, meta in list(metadata_map.items()):
        last_used = meta.get('last_used')
        if last_used and (today - datetime.date.fromisoformat(last_used)).days > 30:
            del metadata_map[tool_name]  # Prune from manifest; optionally delete file
    
    loaded_tools = []
    relevant_tools = []  # Filter based on plan (simple keyword match for efficiency)
    if plan_text:
        plan_lower = plan_text.lower()
        for tool_name, meta in metadata_map.items():
            tags = meta.get('tags', [])
            desc = meta.get('description', '').lower()
            if any(tag.lower() in plan_lower for tag in tags) or plan_lower in desc:
                relevant_tools.append(tool_name)
    else:
        relevant_tools = list(metadata_map.keys())  # Load all if no plan filter
    
    for tool_name in relevant_tools:
        file_path = TOOL_DIR / f"{tool_name}.py"
        if file_path.exists():
            module_name = f"src.generated_tools.{tool_name}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                tool_function = getattr(module, tool_name, None)
                if callable(tool_function):
                    print(f"Dynamically loaded tool: {tool_name}")
                    loaded_tools.append(tool_function)
                    # Update last_used in manifest
                    metadata = metadata_map.get(tool_name, {})
                    metadata['last_used'] = datetime.date.today().isoformat()
                    update_manifest(metadata)
    
    # Rebuild manifest if any files missing (fallback sync)
    for file_path in TOOL_DIR.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        tool_name = file_path.stem
        if tool_name not in metadata_map:
            metadata = extract_frontmatter(file_path)
            if metadata:
                update_manifest(metadata)
    
    return loaded_tools

# --- Initial Tool Setup ---
base_tools = [
    retrieve_from_memory, search_the_web, write_file, 
    execute_script, run_shell_command, add_to_memory, request_human_assistance
]
DANGEROUS_TOOLS = ["write_file", "execute_script", "run_shell_command"]

MAX_REPLAN_ATTEMPTS = 3

# --- API Configuration ---
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model" 

def format_messages_for_api(messages: list[BaseMessage]) -> list[dict]:
    api_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            api_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                api_tool_calls = []
                for tool_call in message.tool_calls:
                    api_tool_calls.append({
                        "id": tool_call.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name"),
                            "arguments": json.dumps(tool_call.get("args", {}))
                        }
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": api_tool_calls
                })
            else:
                api_messages.append({"role": "assistant", "content": message.content})
        elif isinstance(message, ToolMessage):
            api_messages.append({
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": str(message.content)
            })
    return api_messages

def invoke_llm(messages: list[BaseMessage], available_tools: list) -> AIMessage:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    api_messages = format_messages_for_api(messages)
    
    payload = {"model": MODEL_NAME, "messages": api_messages, "temperature": 0}
    if available_tools:
        payload["tools"] = [convert_to_openai_tool(tool) for tool in available_tools]

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        message_data = response_data['choices'][0]['message']

        parsed_tool_calls = []
        if raw_tool_calls := message_data.get("tool_calls"):
            for call in raw_tool_calls:
                function_call = call.get("function", {})
                try:
                    args = json.loads(function_call.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(
                    {
                        "name": function_call.get("name"),
                        "args": args,
                        "id": call.get("id"),
                    }
                )
        
        return AIMessage(content=message_data.get("content", ""), tool_calls=parsed_tool_calls)
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return AIMessage(content=f"Error: The API call failed with an exception: {e}")

# --- Node Definitions ---

def create_plan(state: AgentState):
    """Creates a multi-step plan grounded in the available tools."""
    available_tools = base_tools + load_generated_tools()
    tool_names = ", ".join([tool.__name__ for tool in available_tools])
    
    plan_prompt = f"""Based on the user's goal and the available tools, create a concise, step-by-step plan to achieve the goal.
You can only use the tools provided. Do not make assumptions about information you do not have. Your first step should almost always be to gather information.

Available Tools: {tool_names}

User Goal: {state['user_goal']}

Respond with only the plan, formatted as a numbered list."""
    
    response = invoke_llm([HumanMessage(content=plan_prompt)], available_tools=[])
    return {"plan": response.content, "messages": []}

def format_history_for_prompt(messages: list[BaseMessage]) -> str:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_info = msg.tool_calls[0] if msg.tool_calls else {}
                history.append(f"AI (Tool Call): {tool_info.get('name', 'unknown_tool')}({tool_info.get('args', {})})")
            else:
                history.append(f"AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            history.append(f"Tool ({msg.name}): {msg.content}")
    return "\n".join(history)

def replan(state: AgentState):
    """Creates a new, grounded plan after a failure or human feedback."""
    current_attempts = state.get('replan_attempts', 0)
    available_tools = base_tools + load_generated_tools(state['plan'])  # Phase 3: Pass plan for filtered loading
    tool_names = ", ".join([tool.__name__ for tool in available_tools])
    history_str = format_history_for_prompt(state.get('messages', []))
    failed_actions_str = ", ".join(state.get('failed_actions', [])) or "None"

    replan_prompt = f"""You are an AI agent's planning module. The agent has hit an error or received new human feedback.
Review the conversation history, the user's goal, the available tools, and the list of failed actions. Create a new, actionable, step-by-step plan to achieve the goal.

**Critically analyze the history and failed_actions: {failed_actions_str}.** The previous plan failed. The new plan *must* introduce a new approach.
- If the same tool or category failed multiple times (e.g., >2 API auth errors), do not repeat it. Instead, your plan MUST include steps to create a new tool or escalate.
- For tool creation: 1. Use `search_the_web` to find a Python code snippet for the needed functionality (e.g., keyless weather scraper). 2. Use `write_file` to save it as a new tool in `src/generated_tools/` (filename = function name, e.g., get_weather_free.py). Include YAML frontmatter at the top for metadata (name, description, version, tags, last_used). 3. Use the new tool in later steps.
- If no viable path, plan to call `request_human_assistance` with a summary of failures.

Available Tools: {tool_names}
Original User Goal: {state['user_goal']}

Conversation History:
---
{history_str}
---

Respond with only the new, revised plan, formatted as a numbered list."""
    
    response = invoke_llm([HumanMessage(content=replan_prompt)], available_tools=[])
    print("--- NEW PLAN CREATED ---")
    print(response.content)
    
    # Prune messages if bloated (>10) to prevent context overflow
    messages = state.get('messages', [])
    if len(messages) > 10:
        summary = "Summary of early history: Initial attempts failed due to tool errors; proceeding with revised plan."
        messages = [HumanMessage(content=summary)] + messages[-10:]
    
    return {
        "plan": response.content, 
        "replan_attempts": current_attempts + 1,
        "messages": messages
    }

def planner(state: AgentState):
    """The planner node. It invokes the LLM with the current state to decide the next action."""
    current_messages = state.get('messages', []) 
    completed_steps_str = "\n".join(f"- {step}" for step in state.get('completed_plan_steps', []))
    if not completed_steps_str:
        completed_steps_str = "No steps completed yet."
    failed_actions_str = ", ".join(state.get('failed_actions', [])) or "None"

    plan_context = HumanMessage(content=f"""You are an autonomous AI agent. Your job is to execute a plan to fulfill the user's goal.
Review the plan, the conversation history, the steps you have already completed, and failed_actions: {failed_actions_str}.

**Analyze the most recent tool results very carefully.**
- If the last tool call returned useful information that helps achieve the user's goal, continue with the next step in the plan.
- If the last tool call failed or returned unhelpful, irrelevant, or generic information (like a help page), you MUST recognize this as a failure. Do not proceed with the plan. Instead, your response should be a call to the `request_human_assistance` tool, explaining that the previous approach failed and that you need a new strategy.
- If failed_actions show patterns (e.g., repeated API failures), suggest creating a new tool or escalating.

Decide on the next step:
1. If you have gathered all the necessary information and can now directly answer the user's goal, provide the final answer as a concise summary.
2. Otherwise, select the single best tool to execute for the *next incomplete* step in the plan, unless the previous step failed as described above.

The overall plan is:
{state['plan']}

You have already completed the following steps:
{completed_steps_str}""")

    if not current_messages:
        messages_for_llm = [HumanMessage(content=state['user_goal']), plan_context]
    else:
        messages_for_llm = current_messages + [plan_context]

    available_tools = base_tools + load_generated_tools(state['plan'])  # Phase 3: Pass plan for filtered loading
    response = invoke_llm(messages_for_llm, available_tools=available_tools)
    return {"messages": current_messages + [response]}

def route_after_planner(state: AgentState):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return "end"
    
    planned_tool = last_message.tool_calls[0].get("name")
    if planned_tool == "request_human_assistance":
        return "assistance"
    else:
        return "permission"

def request_permission(state: AgentState):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return state

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call.get("name")

    if tool_name in DANGEROUS_TOOLS:
        print("\n--- PERMISSION REQUEST ---")
        print(f"The agent wants to run the following dangerous tool:")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_call.get('args')}")
        
        while True:
            response = input("Do you approve? (y/n): ").lower()
            if response == 'y':
                print("Permission granted.")
                return state
            elif response == 'n':
                print("Permission denied.")
                denial_message = HumanMessage(content=f"The user has denied permission to run the '{tool_name}' tool. Please choose a different approach.")
                messages = state['messages'][:-1] + [denial_message]
                return {"messages": messages}
    
    return state

def check_for_exit(state: AgentState):
    last_human_message = ""
    for message in reversed(state.get('messages', [])):
        if isinstance(message, HumanMessage):
            last_human_message = message.content.lower()
            break
            
    exit_keywords = ["conclude", "final answer", "stop", "end", "exit"]
    
    if any(keyword in last_human_message for keyword in exit_keywords):
        return "end"
    else:
        return "replan"

def check_for_tool_error(state: AgentState):
    last_message = state['messages'][-1]
    if not isinstance(last_message, ToolMessage):
        return "planner"

    content_str = str(last_message.content)
    if '"status": "error"' in content_str or '"error":' in content_str or '"result": "No' in content_str or "Search returned an empty URL" in content_str:
        return "handle_error"
    else:
        return "mark_step_complete"

def after_permission_check(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        return "planner"
    return "tool_executor"

def handle_circuit_breaker(state: AgentState):
    """Handles the circuit breaker logic when the agent is stuck in a loop."""
    print("\n--- CIRCUIT BREAKER TRIPPED ---")
    print(f"The agent has failed to create a working plan after {MAX_REPLAN_ATTEMPTS} attempts.")
    print("Please provide guidance on how to proceed. You can suggest a new plan, a specific tool to use, or type 'exit' to terminate.")
    
    response = input("Your guidance: ")
    
    if response.lower() == 'exit':
        return {"messages": state['messages'] + [HumanMessage(content="exit")]}

    return {
        "messages": state['messages'] + [HumanMessage(content=response)],
        "replan_attempts": 0 
    }

def handle_human_assistance(state: AgentState):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return state

    request_text = last_message.tool_calls[0].get("args", {}).get("request", "")
    
    print("\n--- HUMAN ASSISTANCE REQUESTED ---")
    print(f"Agent's Request: {request_text}")
    
    response = input("Please provide your response: ")
    
    messages = state['messages'][:-1] + [HumanMessage(content=response)]
    
    return {"messages": messages}

def mark_step_complete(state: AgentState):
    if len(state.get('messages', [])) < 2:
        return state

    last_ai_message = state['messages'][-2]
    
    if not isinstance(last_ai_message, AIMessage) or not last_ai_message.tool_calls:
        return state
        
    tool_call = last_ai_message.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args")
    
    completed_step_summary = f"Executed tool `{tool_name}` with arguments `{tool_args}`."
    
    completed_steps = state.get('completed_plan_steps', [])
    return {"completed_plan_steps": completed_steps + [completed_step_summary]}

def handle_error(state: AgentState):
    last_message = state['messages'][-1]

    # Log the failed action
    failed_tool_name = "unknown_tool"
    if len(state['messages']) > 1:
        ai_message_with_tool_call = state['messages'][-2]
        if isinstance(ai_message_with_tool_call, AIMessage) and ai_message_with_tool_call.tool_calls:
            failed_tool_name = ai_message_with_tool_call.tool_calls[0].get("name", "unknown_tool")

    failed_actions = state.get('failed_actions', [])
    if failed_tool_name not in failed_actions:
        failed_actions.append(failed_tool_name)

    # Classify error (e.g., parse HTTP codes)
    content_str = str(last_message.content)
    error_category = "Unknown error"
    if "401" in content_str or "Unauthorized" in content_str:
        error_category = "Authentication failure (e.g., invalid API key)"
    elif "403" in content_str or "Forbidden" in content_str:
        error_category = "Access forbidden (e.g., blocked site)"
    elif "429" in content_str or "Too Many Requests" in content_str:
        error_category = "Rate limit exceeded"
    elif "404" in content_str or "Not Found" in content_str:
        error_category = "Resource not found"

    error_message = f"The last tool call failed with the following output:\n\n{last_message.content}\n\nError Category: {error_category}\nPlease analyze this error and create a plan to recover."

    return {
        "messages": state['messages'] + [HumanMessage(content=error_message)],
        "failed_actions": failed_actions
    }

def execute_tools(state: AgentState):
    available_tools = base_tools + load_generated_tools(state['plan'])  # Phase 3: Filtered by plan
    tool_node = ToolNode(available_tools)
    
    result_dict = tool_node.invoke({"messages": state['messages']})
    new_tool_messages = result_dict.get('messages', [])
    messages = state.get('messages', []) + new_tool_messages
    return {"messages": messages}

def generate_final_report(state: AgentState):
    user_goal = state['user_goal']
    history_str = format_history_for_prompt(state.get('messages', []))

    report_prompt = f"""You are the summarization module for an AI agent. 
Your task is to write the "Agent's Final Answer" section of a final report.
Based on the original user goal and the full conversation history, write a concise, final answer that summarizes the outcome of the task.

Original User Goal: {user_goal}

Full Conversation History:
---
{history_str}
---
Your response should be only the text of the final answer, without any prefixes.
"""
    
    final_answer_message = invoke_llm([HumanMessage(content=report_prompt)], available_tools=[])
    final_answer = final_answer_message.content

    completed_steps_str = "\n".join(f"- {step}" for step in state.get('completed_plan_steps', []))
    if not completed_steps_str:
        completed_steps_str = "No steps were executed."

    report = f"""
# Agent Final Report

## User Goal
> {user_goal}

## Executed Steps
{completed_steps_str}

## Agent's Final Answer
{final_answer}
"""
    return {"final_report": report.strip()}

def route_after_replan(state: AgentState):
    """Routes to the planner or trips the circuit breaker based on replan attempts."""
    if state.get('replan_attempts', 0) >= MAX_REPLAN_ATTEMPTS:
        return "circuit_breaker"
    return "planner"

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("create_plan", create_plan)
workflow.add_node("planner", planner)
workflow.add_node("replan", replan)
workflow.add_node("tool_executor", execute_tools)
workflow.add_node("mark_step_complete", mark_step_complete)
workflow.add_node("handle_human_assistance", handle_human_assistance)
workflow.add_node("request_permission", request_permission)
workflow.add_node("handle_error", handle_error)
workflow.add_node("handle_circuit_breaker", handle_circuit_breaker)
workflow.add_node("check_for_exit", check_for_exit)
workflow.add_node("final_report_generator", generate_final_report)

workflow.set_entry_point("create_plan")
workflow.add_edge("create_plan", "planner")

workflow.add_conditional_edges("planner", route_after_planner, {
    "assistance": "handle_human_assistance", 
    "permission": "request_permission", 
    "end": "final_report_generator"
})
workflow.add_conditional_edges("handle_human_assistance", check_for_exit, {
    "replan": "replan", "end": "final_report_generator"
})
workflow.add_conditional_edges("replan", route_after_replan, {
    "planner": "planner", "circuit_breaker": "handle_circuit_breaker"
})
workflow.add_edge("handle_circuit_breaker", "check_for_exit")
workflow.add_conditional_edges("request_permission", after_permission_check, {
    "planner": "planner", "tool_executor": "tool_executor"
})
workflow.add_conditional_edges("tool_executor", check_for_tool_error, {
    "mark_step_complete": "mark_step_complete", "handle_error": "handle_error"
})
workflow.add_edge("handle_error", "replan")
workflow.add_edge("mark_step_complete", "planner")
workflow.add_edge("final_report_generator", END)

app = workflow.compile()

# --- CLI Application ---
def run_agent_task(goal: str, session_messages: list):
    """Runs a single task through the agent graph."""
    inputs = {
        "user_goal": goal,
        "messages": session_messages,
        "completed_plan_steps": [],
        "replan_attempts": 0,
        "failed_actions": []
    }
    
    final_state = None
    for output in app.stream(inputs, {"recursion_limit": 100}):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            if key == "final_report_generator":
                print(value.get("final_report"))
                final_state = value
            else:
                print(value)
        print("\n---\n")

    return final_state.get('messages', []) if final_state else session_messages

@click.command()
def cli():
    """An interactive CLI for the Alice agent."""
    print("Welcome to the Alice Agent CLI. Type your goal and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")
    
    session_messages = []
    while True:
        try:
            goal = input("Alice>: ")
            if goal.lower() in ['exit', 'quit']:
                break
            session_messages = run_agent_task(goal, session_messages)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    cli()