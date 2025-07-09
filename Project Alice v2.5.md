Project Alice v2.5: The Advanced CLI Agent
Version: 2.5
Date: July 9, 2025
Objective: To enhance Alice into a fully-featured, interactive command-line agent capable of self-improvement and complex local task execution, inspired by tools like GeminiCLI. This phase prioritizes core capability and interactivity over voice or proactive assistance.
1. Core Philosophy: The Self-Improving Agent
This phase introduces a new core principle:
Runtime Extensibility: The agent should not be limited to the tools it was programmed with. It must be able to identify missing capabilities, find solutions, and dynamically integrate new tools into its own skillset during a single session.
2. Revised Development Phases
The path to a Gemini-like CLI involves three key upgrades: making it interactive, making it smarter, and making it testable.
Phase 3: The Interactive CLI Shell
The agent must be a persistent, interactive tool, not a single-run script.
3.1. Build the CLI Wrapper:
Goal: Refactor main.py to be an interactive command-line application.
Key Libraries: Use argparse or, preferably, Click to create a professional CLI experience.
Functionality:
When a user runs python -m src.main, they should be dropped into an "Alice>" prompt.
The user can type a goal and hit Enter. The agent will execute the goal and print the final report.
The agent will then return to the "Alice>" prompt, ready for a new goal, maintaining the conversation history from the session.
Phase 4: Dynamic Tool Generation
This is the most critical step toward a truly intelligent agent. It gives Alice the ability to learn.
4.1. Create the generated_tools Directory:
Goal: Create a new directory, src/generated_tools/, where Alice can save new Python scripts that she writes.
4.2. Enhance the replan Node:
Goal: Make the agent's recovery process more intelligent.
Logic: The prompt for the replan node in main.py will be updated with a new instruction: "If you are stuck because you lack a specific capability, your new plan must include steps to create that tool. First, use search_the_web to find a Python code snippet that accomplishes the task. Then, use write_file to save this code as a new tool in the src/generated_tools/ directory."
4.3. Implement Dynamic Tool Loading:
Goal: Modify the main application loop in main.py so that at the beginning of each new task, it automatically discovers and loads any .py files from the src/generated_tools/ directory into its list of available tools. This allows the agent to immediately use the tools it creates.
Phase 5: Building an Evaluation Harness
To ensure the agent is getting better and not regressing, we need a systematic way to test its capabilities.
5.1. Create Benchmark Tasks:
Goal: Define a suite of standardized tests in a new evaluation/ directory. Each test will be a YAML file containing a user_goal and a success_condition (e.g., "a file named report.txt must exist" or "the command ls -l must return a specific string").
5.2. Build the Evaluator (evaluate.py):
Goal: Create a new script, evaluate.py, that automatically runs the agent against all benchmark tasks.
Logic: The script will loop through the YAML files in the evaluation/ directory, run the agent with each user_goal, and then programmatically check if the success_condition for that task was met. It will then produce a final report showing the percentage of tasks passed.
3. Updated Project Structure
/Alice_v2.5
|-- src/
|   |-- main.py               # Will be refactored into an interactive CLI
|   |-- tools.py
|   |-- state.py
|   |-- generated_tools/      # New directory for self-written tools
|
|-- sub_agents/
|   |-- memory_service.py
|
|-- evaluation/               # New directory for benchmark tests
|   |-- test_summarize_file.yaml
|   |-- test_create_script.yaml
|
|-- evaluate.py               # New script for running benchmarks
|
|-- knowledge_base/
|-- source_documents/
|-- indexer.py
|-- requirements.txt          # Will be updated with 'click'


This plan provides a direct, focused path to creating a powerful and extensible CLI agent. The immediate next step is to begin Phase 3.1: Build the CLI Wrapper to make the agent interactive.
