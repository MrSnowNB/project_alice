### **Project Alice: An AI First Architectural Plan**

Version: 1.0  
Date: July 7, 2025  
Author: Gemini  
Objective: To provide a comprehensive, step-by-step guide for building "Alice," a resilient, local-first AI agent capable of complex task execution, self-correction, and long-term memory. This document is intended to be a shareable artifact for the AI community.

### **1\. Core Philosophy: The "AI First" Approach**

The development of Alice is guided by a core philosophy that treats the Large Language Model (LLM) not as an all-knowing oracle, but as a specialized cognitive resource within a larger, more robust software architecture.

* **Separation of Concerns:** The application's logic, state management, and control flow are the responsibility of deterministic Python code. The LLM's role is limited to tasks it excels at: language understanding, parameter extraction, and code generation/correction within a highly focused context.  
* **Resilience over Perfection:** We assume that errors are inevitable. The architecture is not designed to prevent all failures but to gracefully handle them. The agent must be able to detect when a step has failed, analyze the error, and formulate a plan to correct it.  
* **Statelessness and Modularity:** The core reasoning loop should be as stateless as possible from the LLM's perspective. Complex tasks are broken down into a series of simple, atomic steps, preventing the "runaway memory" and context pollution issues that plague monolithic conversational agents.

### **2\. Environment Setup & Configuration (The Foundation)**

This is the most critical and often overlooked step. A clean, predictable environment is a non-negotiable prerequisite for building a stable agent.

**2.1. Project Structure (From a Clean Folder)**

Create the following directory and file structure to ensure a clean workspace.

/Alice  
|-- src/  
|   |-- main.py             \# The main application entry point (our LangGraph kernel)  
|   |-- tools.py            \# Tool definitions (e.g., get\_coordinates)  
|   |-- state.py            \# The definition of our AgentState class  
|-- knowledge\_base/         \# The directory for our future vector database  
|-- .gitignore  
|-- requirements.txt

**2.2. Local LLM Server Configuration (LM Studio)**

This is the source of the most significant instability in early development. The following configuration **must** be applied to ensure the server acts as a pure, pass-through inference engine.

1. **Launch LM Studio** and navigate to the **Local Server** tab (↔️).  
2. Load the target model (e.g., gemma-3-27b-abliterated-dpo.gguf).  
3. On the right-hand panel, find the **"System Prompt"** text box and ensure it is **completely empty**.  
4. Click the **"Preset"** dropdown menu and select (or create) the **LangChain** preset. The critical configuration for this preset is "prompt\_format": "raw". This tells LM Studio not to modify or re-template the prompts sent by our application.  
5. Restart the server to ensure all settings are applied.

**2.3. Python Dependencies (requirements.txt)**

This file will contain all necessary libraries.

langchain  
langchain-openai  
langgraph  
beautifulsoup4  
googlesearch-python  
pyyaml  
\# For Phase 2 (RAG System)  
\# sentence-transformers  
\# chromadb

Install with: pip install \-r requirements.txt

### **3\. Phase 1: The Self-Correcting State Graph (Working Memory)**

This phase focuses on building the agent's core "executive function" using LangGraph. This is the agent's "Working Memory" or "Workbench."

**3.1. The State (src/state.py)**

Define the graph's state object. This object will be passed between every node, carrying the full context of the current task.

from typing import List, Dict, Any

class AgentState(TypedDict):  
    user\_goal: str  
    file\_path: str  
    city\_to\_test: str  
    current\_code: str  
    error\_message: str  
    correction\_attempts: int

**3.2. The Tools (src/tools.py)**

Define the agent's capabilities as simple Python functions. They should be stateless and only operate on the inputs they are given.

* get\_coordinates(city: str) \-\> dict  
* get\_weather(latitude: float, longitude: float, temp\_unit: str) \-\> dict  
* write\_file(file\_path: str, content: str) \-\> dict  
* execute\_script(file\_path: str, args: List\[str\]) \-\> dict

**3.3. The Graph Definition (src/main.py)**

This is the heart of the application. We will use LangGraph to define the agent's "mind" as a flowchart.

* **Nodes:** Each node will be a Python function that takes the AgentState as input and returns a dictionary with the updated state values.  
  * extract\_parameters: Uses the LLM once to populate the initial state.  
  * write\_initial\_script: Writes the first version of the code.  
  * execute\_script: Runs the code and captures success or failure.  
  * request\_correction: If execute\_script fails, this node uses the LLM to generate a corrected version of the code.  
* **Edges:** The logic that connects the nodes.  
  1. **Entry Point:** The graph will start at extract\_parameters.  
  2. **Linear Flow:** It will proceed extract\_parameters \-\> write\_initial\_script \-\> execute\_script.  
  3. **Conditional Edge (The Correction Loop):** After the execute\_script node, a conditional edge will check the error\_message in the state.  
     * **If error\_message is None:** The script succeeded. The graph transitions to the **END**.  
     * **If error\_message exists:** The graph transitions to the request\_correction node.  
  4. **Loop:** After request\_correction, the graph transitions back to write\_initial\_script to write the *new* code, creating the self-correction loop. The loop will terminate after a set number of attempts.

### **4\. Phase 2: The Reranked Vector Database (Reference Memory)**

Once the agent is stable and can reliably execute complex plans, we will give it a long-term memory.

**4.1. The Indexer**

A new script, indexer.py, will be created to build the knowledge base. It will:

1. Use our established AST-based chunking method to parse a target codebase.  
2. For each code chunk, generate a vector embedding using a sentence-transformer model.  
3. Store the chunk's content, its vector embedding, and its rich YAML metadata in a local ChromaDB vector store located in the /knowledge\_base directory.

**4.2. The retrieve\_from\_memory Tool**

A new tool will be added to src/tools.py. This tool will be the agent's interface to its long-term memory.

1. It will take a natural language query and optional metadata filters as input.  
2. It will perform a **hybrid search**: first a semantic vector search, followed by a metadata filter.  
3. The top results from the retrieval step will be passed to a **cross-encoder model for reranking**.  
4. Only the top 3-5 most relevant, reranked results will be returned.

**4.3. Integration with the Graph**

The LangGraph agent will be taught to use this new tool. When faced with a novel problem or a difficult error, its plan will now include a step to query its own memory, allowing it to learn from past experiences and apply previously known solutions to new problems.

### **Project Alice: Final Status Report & Action Plan**

Date of Record: Monday, July 7, 2025, 7:33 PM EDT  
Project Goal: To build a stable, resilient, local-first AI agent ("Alice") capable of executing complex, multi-step development tasks.  
Core Technology: gemma-3-27b (via LM Studio), LangChain, and LangGraph.

### **1\. Summary of Achievements & Key Learnings**

Our extensive debugging session was highly productive. We successfully diagnosed and solved several critical, foundational issues, leading us to a robust final architecture.

**What We Successfully Accomplished:**

* **Solved Environment Conflicts:** We definitively identified and fixed the "Two Chefs" problem. By creating a custom LangChain preset in LM Studio and clearing the UI's system prompt, we ensured our application has exclusive control over the prompts sent to the model. This was a critical breakthrough.  
* **Solved "Runaway Memory":** We proved that a simple conversational loop overwhelms the local model's context window. We successfully architected and implemented a LangGraph state machine, which solves this problem by design.  
* **Validated the LLM's Capability:** We proved that the gemma-3-27b model is highly capable of reasoning and tool use when given focused, single-purpose tasks (like parameter extraction). The failures were not due to the model's intelligence but our application's architecture.  
* **Built a Working Correction Loop:** We successfully built a "Human-in-the-Loop" graph that can execute a script, detect a failure, and correctly pause to ask the user for help.

### **2\. The Final Bug: The Sticking Point**

Despite our progress, the agent is currently stuck in an infinite debugging loop. The log analysis has revealed the single, specific bug responsible for this failure.

**What Is Not Working:**

* **The Symptom:** When the agent's script fails and the human is prompted to fix the file, after the human saves the corrected code and types retry, the agent re-executes the script and fails with the *exact same error*.  
* **The Root Cause:** The agent is using a "cached" or stale version of the code. The request\_human\_correction node in our graph is not correctly updating the agent's internal memory (AgentState) after the human fixes the file on the disk. It clears the error but fails to re-load the corrected code, causing it to re-execute the original, broken script from its memory.

### **3\. The Final Implementation Plan**

We have a clear, definitive plan to fix this final bug. This plan represents the culmination of all our learning and will result in a stable, working agent.

**The Architecture: The "Resilient Co-Pilot" (v26/v28)**

The LangGraph architecture is correct. We only need to make one crucial change to the logic of the request\_human\_correction node.

**The Core Fix:**

The request\_human\_correction function in src/main.py must be modified. When the user types retry, it must perform two actions in this specific order:

1. **Re-read the File:** It must open the script file (e.g., weather\_tool.py) from the hard disk and read its contents. This ensures it gets the version of the code that the human has just fixed and saved.  
2. **Update State:** It must update the current\_code key in the AgentState dictionary with the new code it just read from the file.

With this change, when the graph loops back to the execute\_script\_node, it will be operating on the fresh, corrected code from the state, breaking the loop and allowing the agent to proceed.

**Immediate Next Steps:**

1. **Implement the Fix:** Apply the change described above to the request\_human\_correction function in src/main.py.  
2. **Run the Final Test:** Execute the script and provide the definitive test prompt:  
   Create a Python script named \`weather\_tool.py\`. This script must be a reusable command-line tool that accepts a city name as an argument. When run, it should print the current temperature for that city in both Celsius and Fahrenheit. After creating the script, test it by running it for "Lawrence Township, New Jersey".

3. **Perform the Human-in-the-Loop Step:** When the agent pauses and asks for help, open weather\_tool.py, fix the URL encoding bug, **save the file**, and then type retry in the terminal.

This plan directly addresses the final, identified bug. It is the most logical and robust path to successfully completing our goal. Enjoy your break.

