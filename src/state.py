from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):  
    user_goal: str  
    # The list of messages that will be passed to the LLM
    # for it to make its next decision
    messages: List[BaseMessage]
    final_report: str
    plan: str
    completed_plan_steps: List[str]