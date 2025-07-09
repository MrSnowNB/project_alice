import json
import requests
import sys
import os

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.state import AgentState
from src.tools import retrieve_from_memory, search_the_web, write_file, execute_script, run_shell_command, add_to_memory, request_human_assistance

# Define the tools the agent can use
tools = [retrieve_from_memory, search_the_web, write_file, execute_script, run_shell_command, add_to_memory, request_human_assistance]

# Define a list of tools that require user confirmation before execution
DANGEROUS_TOOLS = ["write_file", "execute_script", "run_shell_command"]

# --- Custom LLM Invocation (No LangChain Model Wrapper) ---
API_URL = "http://localhost:1234/v1/chat/completions"
# Note: The model name might need to be adjusted based on what's running in your LM Studio
MODEL_NAME = "local-model" 

def format_messages_for_api(messages: list[BaseMessage]) -> list[dict]:
    """Converts LangChain message objects to a list of dictionaries for the API."""
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
                "content": message.content
            })
    return api_messages

def invoke_llm(messages: list[BaseMessage], use_tools: bool = False) -> AIMessage:
    """Invokes the LLM via a direct API call."""
    api_messages = format_messages_for_api(messages)
    
    payload = {"model": MODEL_NAME, "messages": api_messages, "temperature": 0}
    if use_tools:
        payload["tools"] = [convert_to_openai_tool(tool) for tool in tools]

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
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
    """Creates a multi-step plan to achieve the user's goal."""
    plan_prompt = f"""Based on the user's goal, create a concise, step-by-step plan to achieve it.
Each step should be a clear action for the agent.

User Goal: {state['user_goal']}

Respond with only the plan, formatted as a numbered list."""
    
    messages = [HumanMessage(content=plan_prompt)]
    response = invoke_llm(messages, use_tools=False)
    return {"plan": response.content}

def format_history_for_prompt(messages: list[BaseMessage]) -> str:
    """Formats the message history into a readable string for a prompt."""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_info = msg.tool_calls[0] if msg.tool_calls else {}
                tool_name = tool_info.get('name', 'unknown_tool')
                tool_args = tool_info.get('args', {})
                history.append(f"AI (Tool Call): {tool_name}({tool_args})")
            else:
                history.append(f"AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            history.append(f"Tool ({msg.name}): {msg.content}")
    return "\n".join(history)

def replan(state: AgentState):
    """Creates a new plan after a failure or human feedback."""
    history_str = format_history_for_prompt(state.get('messages', []))
    replan_prompt = f"""You are an AI agent's planning module. The agent has hit an error or received new human feedback.
Review the entire conversation history and the original user goal. Your task is to create a new, actionable, step-by-step plan to achieve the original goal.

**Critically analyze the history.** The previous plan failed. The new plan *must* introduce a new approach to get past the error. For example, if a tool failed, try calling it with different parameters or use a different tool entirely. If a web search yields no results, try a different, more general search query.

Original User Goal: {state['user_goal']}

Conversation History:
---
{history_str}
---

Respond with only the new, revised plan, formatted as a numbered list."""
    
    messages = [HumanMessage(content=replan_prompt)]
    response = invoke_llm(messages, use_tools=False)
    print("--- NEW PLAN CREATED ---")
    print(response.content)
    return {"plan": response.content}

def planner(state: AgentState):
    """The planner node. It invokes the LLM with the current state to decide the next action."""
    current_messages = state.get('messages', []) 
    completed_steps_str = "\n".join(f"- {step}" for step in state.get('completed_plan_steps', []))
    if not completed_steps_str:
        completed_steps_str = "No steps completed yet."

    plan_context = HumanMessage(content=f"""You are an autonomous AI agent. Your job is to execute a plan to fulfill the user's goal.
Review the plan, the conversation history, and the steps you have already completed. 

Decide on the next step:
1. If you have gathered all the necessary information and can now directly answer the user's goal, provide the final answer as a concise summary.
2. Otherwise, select the single best tool to execute for the *next incomplete* step in the plan.

The overall plan is:
{state['plan']}

You have already completed the following steps:
{completed_steps_str}""")

    if not current_messages:
        messages_for_llm = [HumanMessage(content=state['user_goal']), plan_context]
    else:
        messages_for_llm = current_messages + [plan_context]

    response = invoke_llm(messages_for_llm, use_tools=True)
    return {"messages": current_messages + [response]}

def route_after_planner(state: AgentState):
    """Decides the next step after the planner has made a decision."""
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return "end"
    
    planned_tool = last_message.tool_calls[0].get("name")
    if planned_tool == "request_human_assistance":
        return "assistance"
    else:
        return "permission"

def request_permission(state: AgentState):
    """Checks if the planned tool call is dangerous and asks for user permission."""
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
    """Checks the user's last message for an exit command."""
    last_human_message = ""
    for message in reversed(state.get('messages', [])):
        if isinstance(message, HumanMessage):
            last_human_message = message.content.lower()
            break
            
    exit_keywords = ["conclude", "final answer", "stop", "end", "exit"]
    
    if any(keyword in last_human_message for keyword in exit_keywords):
        print("--- EXIT COMMAND DETECTED ---")
        return "end"
    else:
        return "replan"

def check_for_tool_error(state: AgentState):
    """
    Checks the last message for a tool error and routes accordingly.
    Also treats unhelpful results as errors to be handled.
    """
    last_message = state['messages'][-1]
    if not isinstance(last_message, ToolMessage):
        return "planner"

    content_str = last_message.content
    if '"status": "error"' in content_str or '"error":' in content_str or '"result": "No' in content_str or "Search returned an empty URL" in content_str:
        return "handle_error"
    else:
        return "mark_step_complete"

def after_permission_check(state: AgentState):
    """Routes to planner if permission was denied, otherwise to the tool executor."""
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        return "planner"
    return "tool_executor"

def handle_human_assistance(state: AgentState):
    """Handles the agent's request for human help."""
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
    """Marks the last executed step as complete."""
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
    """A dedicated node to process errors and formulate a recovery plan."""
    last_message = state['messages'][-1]
    error_message = f"The last tool call failed with the following output:\n\n{last_message.content}\n\nPlease analyze this error and create a plan to recover."
    return {"messages": state['messages'] + [HumanMessage(content=error_message)]}

def execute_tools(state: AgentState):
    """A robust custom node that executes tools and correctly appends the result."""
    tool_node = ToolNode(tools)
    result_dict = tool_node.invoke({"messages": state['messages']})
    new_tool_messages = result_dict.get('messages', [])
    messages = state.get('messages', []) + new_tool_messages
    return {"messages": messages}

def generate_final_report(state: AgentState):
    """Generates a structured final report."""
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

Provide only the final summary answer for the report.
"""
    
    final_answer_message = invoke_llm([HumanMessage(content=report_prompt)], use_tools=False)
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
workflow.add_node("check_for_exit", check_for_exit)
workflow.add_node("final_report_generator", generate_final_report)

workflow.set_entry_point("create_plan")
workflow.add_edge("create_plan", "planner")

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {"assistance": "handle_human_assistance", "permission": "request_permission", "end": "final_report_generator"}
)

workflow.add_conditional_edges(
    "handle_human_assistance",
    check_for_exit,
    {"replan": "replan", "end": "final_report_generator"}
)
workflow.add_edge("replan", "planner")

workflow.add_conditional_edges(
    "request_permission",
    after_permission_check,
    {"planner": "planner", "tool_executor": "tool_executor"}
)

workflow.add_conditional_edges(
    "tool_executor",
    check_for_tool_error,
    {"mark_step_complete": "mark_step_complete", "handle_error": "handle_error"}
)

# FINAL BUG FIX: Route errors to the replan node to force a new strategy.
workflow.add_edge("handle_error", "replan")
workflow.add_edge("mark_step_complete", "planner")
workflow.add_edge("final_report_generator", END)

app = workflow.compile()

# Run the agent
if __name__ == "__main__":
    inputs = {
        "user_goal": "Search the web to find out who the current CEO of OpenAI is. Then, add this information to your long-term memory. Finally, retrieve the information from your memory and report it as your final answer."
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
