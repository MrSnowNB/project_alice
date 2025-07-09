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
MODEL_NAME = "mistralai/Devstral-Small-2505_gguf"

# Convert our Python functions into a format the API understands
formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

def format_messages_for_api(messages: list[BaseMessage]) -> list[dict]:
    """Converts LangChain message objects to a list of dictionaries for the API."""
    api_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            api_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            # Handle tool calls in AIMessage
            if message.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": message.content or "", # content can be None
                    "tool_calls": message.tool_calls
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
    """
    Invokes the LLM via a direct API call, bypassing LangChain model wrappers.
    """
    api_messages = format_messages_for_api(messages)
    
    payload = {"model": MODEL_NAME, "messages": api_messages, "temperature": 0}
    if use_tools:
        payload["tools"] = formatted_tools

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        message_data = response_data['choices'][0]['message']

        # Manually parse the tool calls from the API response into the format
        # that LangChain's AIMessage expects. The API returns an OpenAI-like
        # structure with a nested 'function' dictionary.
        parsed_tool_calls = []
        if raw_tool_calls := message_data.get("tool_calls"):
            for call in raw_tool_calls:
                function_call = call.get("function", {})
                # Safely parse the arguments string, defaulting to an empty dict on error
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
    
    # We create a HumanMessage to pass to our custom invoke function
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
                # Safely access tool call information
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
    """Creates a new plan based on the full conversation history after receiving human feedback."""
    history_str = format_history_for_prompt(state.get('messages', []))
    replan_prompt = f"""You are an AI agent's planning module. The agent has been executing a task and has received new instructions from a human. 
Review the entire conversation history and the original user goal. Your task is to create a new, actionable, step-by-step plan to achieve the original goal.

**Critically analyze the history.** If the agent is stuck in a loop or not making progress, the new plan *must* introduce a new approach. For example, if the agent needs a new capability, the plan should include a step to use the `search_the_web` tool to find out how to build it.

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
    # 1. Safely get the current message history from the state.
    current_messages = state.get('messages', []) 

    # 2. Get the list of completed steps to provide context.
    completed_steps_str = "\n".join(f"- {step}" for step in state.get('completed_plan_steps', []))
    if not completed_steps_str:
        completed_steps_str = "No steps completed yet."

    # 3. Create the context message containing the plan and completed steps.
    # This prompt is more forceful, instructing the LLM that its only valid output is a tool call.
    plan_context = HumanMessage(content=f"""You are an autonomous AI agent. Your job is to execute a plan to fulfill the user's goal.
Review the plan, the conversation history, and the steps you have already completed. Then, select the single best tool to execute for the *next incomplete* step.
Do not explain your reasoning or ask for clarification. Your output must be a tool call.

The overall plan is:
{state['plan']}

You have already completed the following steps:
{completed_steps_str}""")

    # 4. Construct the messages to be sent to the LLM for this specific turn.
    if not current_messages:
        # First turn: user goal + plan context
        messages_for_llm = [HumanMessage(content=state['user_goal']), plan_context]
    else:
        # Subsequent turns: The full history + the plan context as the final instruction.
        messages_for_llm = current_messages + [plan_context]

    # 5. Invoke the LLM.
    response = invoke_llm(messages_for_llm, use_tools=True)

    # 6. Append the LLM's response to the original message history to update the state.
    return {"messages": current_messages + [response]}

def route_after_planner(state: AgentState):
    """Decides the next step after the planner has made a decision."""
    last_message = state['messages'][-1]
    # If the LLM didn't call a tool, it means it has a final answer.
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
        # This should not be reached if graph is correct, but as a safeguard
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
                # Return state unchanged to proceed
                return state
            elif response == 'n':
                print("Permission denied.")
                # Replace the AI's tool call with a message indicating denial
                denial_message = HumanMessage(content=f"The user has denied permission to run the '{tool_name}' tool. Please choose a different approach or ask for clarification.")
                # We replace the last message
                messages = state['messages'][:-1] + [denial_message]
                return {"messages": messages}
    
    # If the tool is not dangerous, just pass through
    return state


def check_for_tool_error(state: AgentState):
    """Checks the last message for a tool error and routes accordingly."""
    last_message = state['messages'][-1]
    if not isinstance(last_message, ToolMessage):
        return "planner" # Should not happen, but as a safeguard

    # The content of a ToolMessage is a string representation of the tool's return dict.
    # We check for common error signatures.
    if '"status": "error"' in last_message.content or '"error":' in last_message.content:
        return "handle_error"
    else:
        return "mark_step_complete"

def after_permission_check(state: AgentState):
    """Routes to planner if permission was denied, otherwise to the tool executor."""
    last_message = state['messages'][-1]
    # If the last message is a HumanMessage, it means permission was denied and we injected a new prompt.
    if isinstance(last_message, HumanMessage):
        return "planner"
    return "tool_executor"

def handle_human_assistance(state: AgentState):
    """Handles the agent's request for human help."""
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return state # Should not happen

    request_text = last_message.tool_calls[0].get("args", {}).get("request", "")
    
    print("\n--- HUMAN ASSISTANCE REQUESTED ---")
    print(f"Agent's Request: {request_text}")
    
    response = input("Please provide your response: ")
    
    # We replace the tool call with a new HumanMessage to guide the next planning step.
    messages = state['messages'][:-1] + [HumanMessage(content=response)]
    
    return {"messages": messages}

def mark_step_complete(state: AgentState):
    """Marks the last executed step as complete."""
    # The tool call that was just executed is in the second-to-last message.
    # The last message is the ToolMessage with the result.
    last_ai_message = state['messages'][-2]
    
    if not isinstance(last_ai_message, AIMessage) or not last_ai_message.tool_calls:
        return state # Should not happen, but safeguard
        
    tool_call = last_ai_message.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args")
    
    completed_step_summary = f"Executed tool `{tool_name}` with arguments `{tool_args}`."
    
    return {"completed_plan_steps": state.get('completed_plan_steps', []) + [completed_step_summary]}

def handle_error(state: AgentState):
    """A dedicated node to process errors and formulate a recovery plan."""
    last_message = state['messages'][-1]
    error_message = f"The last tool call failed with the following output:\n\n{last_message.content}\n\nPlease analyze this error and create a plan to recover. You can retry the tool with different parameters, use a different tool, or ask for help if you are stuck."
    # We add this as a new HumanMessage to force the LLM to address it directly.
    return {"messages": state['messages'] + [HumanMessage(content=error_message)]}

def generate_final_report(state: AgentState):
    """Generates a structured final report upon task completion."""
    user_goal = state['user_goal']
    plan = state.get('plan', 'No plan was generated.')
    completed_steps = state.get('completed_plan_steps', [])
    final_response = state['messages'][-1].content

    completed_steps_str = "\n".join(f"- {step}" for step in completed_steps)
    if not completed_steps_str:
        completed_steps_str = "No steps were executed."

    report = f"""
# Agent Final Report

## User Goal
> {state['user_goal']}

## Final Plan
{plan}

## Executed Steps
{completed_steps_str}

## Agent's Final Answer
{final_response}
"""
    return {"final_report": report.strip()}

# --- Graph Definition ---
# Define the graph
workflow = StateGraph(AgentState)

# The planner node decides the next action
workflow.add_node("create_plan", create_plan)
workflow.add_node("planner", planner)
workflow.add_node("replan", replan)
# The tool_node executes the tools chosen by the planner
tool_node = ToolNode(tools)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("mark_step_complete", mark_step_complete)
workflow.add_node("handle_human_assistance", handle_human_assistance)
workflow.add_node("request_permission", request_permission)
workflow.add_node("handle_error", handle_error)
workflow.add_node("final_report_generator", generate_final_report)

# Set the entry point
workflow.set_entry_point("create_plan")

# The plan is created first, then passed to the planner.
workflow.add_edge("create_plan", "planner")

# Define the conditional logic
workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {"assistance": "handle_human_assistance", "permission": "request_permission", "end": "final_report_generator"}
)

# After getting human help, the agent should replan.
workflow.add_edge("handle_human_assistance", "replan")
workflow.add_edge("replan", "planner")

# After requesting permission, decide where to go
workflow.add_conditional_edges(
    "request_permission",
    after_permission_check,
    {"planner": "planner", "tool_executor": "tool_executor"}
)

# After executing a tool, check for errors
workflow.add_conditional_edges(
    "tool_executor",
    check_for_tool_error,
    {"mark_step_complete": "mark_step_complete", "handle_error": "handle_error"}
)

# The error handler routes back to the planner to re-evaluate
workflow.add_edge("handle_error", "planner")
# After a step is marked complete, go back to the planner
workflow.add_edge("mark_step_complete", "planner")
# The final report generator is the last step
workflow.add_edge("final_report_generator", END)

# Compile the graph
app = workflow.compile()

# Run the agent
if __name__ == "__main__":
    # This goal will test the new human assistance request mechanism.
    inputs = {"user_goal": "I need to convert a PDF file named 'document.pdf' to a text file. I don't think you have a tool for that. Please request assistance and ask for a new tool or a shell command to accomplish this."}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
