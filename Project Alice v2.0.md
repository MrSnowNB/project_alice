### **Project Alice v2.0: A Modular, Sub-Agent Architecture**

Version: 2.0  
Date: July 9, 2025  
Objective: To refactor "Alice" into a multi-component system, separating the core reasoning engine from specialized, high-load services. This plan introduces a dedicated Memory Sub-Agent to handle all knowledge base operations, optimizing for performance and modularity.

### **1\. Core Philosophy: The "AI First" Approach v2.0**

The foundational principles of Separation of Concerns and Resilience remain. We now add a third, critical principle:

* **Specialized, Independent Agents:** Complex, resource-intensive, or highly specialized tasks should be encapsulated within their own independent services or "sub-agents." The main agent acts as an orchestrator, delegating tasks to these specialists. This prevents the main agent from becoming a monolith, improves testability, and allows for more efficient resource management (like VRAM).

### **2\. System Components**

The project will now consist of two primary components running as separate processes.

**2.1. The Main CLI Agent (The Orchestrator)**

* **Role:** This remains the "brain" of the operation, responsible for user interaction, planning, replanning, and general tool orchestration.  
* **Implementation:** The existing LangGraph application (main.py, state.py).  
* **Key Change:** It will **no longer** directly manage the vector database or the embedding models. Its tools.py will be refactored. The retrieve\_from\_memory and add\_to\_memory functions, which load heavy models, will be removed.

**2.2. The Memory Sub-Agent (The Retriever)**

* **Role:** A dedicated, lightweight service whose **only** responsibility is to perform knowledge base operations. It owns the vector database, the embedding model, and the cross-encoder.  
* **Implementation:** This will be a new, standalone web service (e.g., using Flask or FastAPI) that runs alongside the main agent. It will expose simple API endpoints.  
  * POST /query: Accepts a JSON payload with a query string and returns the reranked, relevant context.  
  * POST /add: Accepts a JSON payload with text\_to\_remember and adds it to the vector store.  
* **Benefit:** This isolates the VRAM-heavy models (sentence-transformer, cross-encoder) into a single, predictable process that can be managed and tested independently.

### **3\. Revised Development Phases**

The project is now divided into two new parallel phases, focusing on building and integrating these two components.

**Phase 1: Build & Validate the Memory Sub-Agent**

This phase is dedicated entirely to creating a robust, standalone retrieval service.

* **1.1. Create the Service (memory\_service.py):**  
  * Build a simple Flask/FastAPI application.  
  * On startup, it will load the sentence-transformer and cross-encoder models into memory.  
  * It will initialize the connection to the ChromaDB vector store.  
  * It will expose the /query and /add endpoints as described above.  
* **1.2. Adapt the Indexer (indexer.py):**  
  * The existing indexer.py script remains largely the same. Its purpose is to build the chroma\_db directory that the Memory Sub-Agent will use.  
* **1.3. Execute the Formal Test Plan:**  
  * Rigorously execute the "Test Plan for Sub-Agent Vector Database Retrieval" against the new service's API endpoints.  
  * Use tools like curl or a simple Python requests script to simulate queries.  
  * Monitor nvidia-smi to validate VRAM usage and ensure it meets the efficiency goals (\< 12GB).  
  * **This phase is not complete until all test cases in the plan are passed.**

**Phase 2: Refactor the Main CLI Agent**

Once the Memory Sub-Agent is validated, the Main Agent will be updated to communicate with it.

* **2.1. Refactor the Toolset (tools.py):**  
  * Remove the old retrieve\_from\_memory and add\_to\_memory functions.  
  * Create **new** functions that act as simple API clients:  
    * query\_memory\_service(query: str) \-\> dict: This function will not load any models. It will simply make a requests.post call to the Memory Sub-Agent's /query endpoint and return the JSON response.  
    * add\_to\_memory\_service(text\_to\_remember: str) \-\> dict: This function will make a requests.post call to the /add endpoint.  
* **2.2. Update the Main Graph (main.py):**  
  * The list of tools available to the agent will be updated to include the new query\_memory\_service and add\_to\_memory\_service.  
  * The core LangGraph logic remains the same, but it will now be calling the lightweight API client tools instead of the heavy, model-loading functions.  
* **2.3. End-to-End Integration Testing:**  
  * Run the full agent with a goal that requires memory retrieval.  
  * Verify that the Main Agent correctly calls the Memory Sub-Agent, receives the context, and incorporates it into its plan.

### **4\. Updated Project Structure**

/Alice\_v2  
|-- src/                      \# The Main CLI Agent (Orchestrator)  
|   |-- main.py  
|   |-- tools.py              \# Contains lightweight API client tools now  
|   |-- state.py  
|  
|-- sub\_agents/               \# The new home for specialized services  
|   |-- memory\_service.py     \# The Flask/FastAPI Memory Sub-Agent  
|  
|-- knowledge\_base/  
|   |-- chroma\_db/  
|  
|-- source\_documents/  
|  
|-- indexer.py  
|-- requirements.txt  
|-- test\_plan.md              \# The formal test plan document  