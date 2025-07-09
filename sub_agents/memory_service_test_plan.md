# Test Plan for Memory Sub-Agent

**Version:** 1.0  
**Objective:** To validate the functionality, correctness, and error handling of the `memory_service.py` Flask application.

---

## 1. Pre-Test Setup

1.  **Install Dependencies:** Ensure all dependencies from `requirements.txt` are installed in the virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Build the Knowledge Base:** Run the `indexer.py` script to create an initial ChromaDB vector store in the `knowledge_base/chroma_db` directory. This is required for the `/query` endpoint to have data to search.
    ```bash
    python -m src.indexer
    ```
3.  **Start the Memory Service:** Launch the Flask application from the project root directory.
    ```bash
    python -m sub_agents.memory_service
    ```
    The service should start on `http://127.0.0.1:5001` and log that it has successfully loaded the models and connected to the database.

---

## 2. Test Cases

All tests will be performed using a separate terminal window with a tool like `curl` or a REST client like Postman.

### 2.1. `/query` Endpoint Tests

**Test Case 2.1.1: Successful Query**
*   **Description:** Send a valid query that should have relevant results in the indexed documents.
*   **Request:** `POST http://127.0.0.1:5001/query` with JSON body `{"query": "What is the agent's state object?"}`
*   **Expected Outcome:**
    *   HTTP Status Code: `200 OK`
    *   JSON Response: A `relevant_context` key containing a list of 1 to 3 document objects, each with `content` and `metadata`.

**Test Case 2.1.2: Query with No Results**
*   **Description:** Send a valid query that is unlikely to have relevant results.
*   **Request:** `POST http://127.0.0.1:5001/query` with JSON body `{"query": "Information about culinary arts"}`
*   **Expected Outcome:**
    *   HTTP Status Code: `200 OK`
    *   JSON Response: A `relevant_context` key with an empty list `[]`.

**Test Case 2.1.3: Malformed Request (Missing Query)**
*   **Description:** Send a request with an invalid JSON body.
*   **Request:** `POST http://127.0.0.1:5001/query` with JSON body `{}`
*   **Expected Outcome:**
    *   HTTP Status Code: `400 Bad Request`
    *   JSON Response: An `error` key with a message like "Missing 'query' in request body".

### 2.2. `/add` Endpoint Tests

**Test Case 2.2.1: Successful Add**
*   **Description:** Add a new piece of text to the knowledge base.
*   **Request:** `POST http://127.0.0.1:5001/add` with JSON body `{"text_to_remember": "The current CEO of OpenAI is Sam Altman."}`
*   **Expected Outcome:**
    *   HTTP Status Code: `200 OK`
    *   JSON Response: A `status` key with value `"success"`.

**Test Case 2.2.2: Verify Persistence**
*   **Description:** After a successful "add", query for the newly added information to ensure it was persisted.
*   **Request:** `POST http://127.0.0.1:5001/query` with JSON body `{"query": "Who is the CEO of OpenAI?"}`
*   **Expected Outcome:**
    *   HTTP Status Code: `200 OK`
    *   JSON Response: The `relevant_context` should contain the text added in Test Case 2.2.1.

**Test Case 2.2.3: Malformed Request (Missing Text)**
*   **Description:** Send a request with an invalid JSON body.
*   **Request:** `POST http://127.0.0.1:5001/add` with JSON body `{}`
*   **Expected Outcome:**
    *   HTTP Status Code: `400 Bad Request`
    *   JSON Response: An `error` key with a message like "Missing 'text_to_remember' in request body".