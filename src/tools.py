import os
import shlex
import subprocess
import requests
from googlesearch import search
from bs4 import BeautifulSoup

# --- Configuration for the Memory Sub-Agent Service ---
MEMORY_SERVICE_URL = "http://127.0.0.1:5001"

def retrieve_from_memory(query: str) -> dict:
    """
    Searches the agent's knowledge base for information relevant to the query by calling the memory sub-agent.
    """
    print(f"Querying memory service for: '{query}'")
    try:
        response = requests.post(f"{MEMORY_SERVICE_URL}/query", json={"query": query}, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        response_data = response.json()
        if "error" in response_data:
            return {"error": response_data["error"]}
            
        # Format the context for the LLM
        context = "\n\n---\n\n".join([doc['content'] for doc in response_data.get("relevant_context", [])])
        if not context:
            return {"result": "No relevant information found in memory."}
            
        return {"relevant_context": context}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to memory service: {e}"}

def add_to_memory(text_to_remember: str) -> dict:
    """
    Adds a piece of text to the agent's long-term memory by calling the memory sub-agent.
    """
    print(f"Sending to memory service: '{text_to_remember[:100]}...'")
    try:
        response = requests.post(f"{MEMORY_SERVICE_URL}/add", json={"text_to_remember": text_to_remember}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to memory service: {e}"}

def http_get(url: str, params: dict = None) -> dict:
    """Makes an HTTP GET request to the specified URL and returns the response text."""
    print(f"Making HTTP GET request to: {url} with params: {params}")
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        # Return the raw text content. The LLM can parse it.
        return {"response_text": response.text}
    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP GET request failed: {e}"}

def search_the_web(query: str) -> dict:
    """Searches the web for a query and returns the text content of the top search result."""
    print(f"Searching the web for: '{query}'")
    try:
        # Get the first URL from the search results
        try:
            # ROBUST FIX: Remove the problematic argument and just get the iterator.
            search_results = search(query)
            # Then, take the first result from the iterator.
            url = next(search_results)
        except StopIteration:
            return {"result": "No search results found."}

        # Add a check to ensure the URL is not empty or None
        if not url:
            return {"result": "Search returned an empty URL."}

        print(f"Scraping content from: {url}")
        # Scrape the content from the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # A simple way to extract text, removing script and style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Return a snippet to avoid overwhelming the context
        return {"retrieved_content": clean_text[:4000]}
    except Exception as e:
        return {"error": f"An error occurred during web search: {e}"}

def write_file(file_path: str, content: str) -> dict:
    """Writes content to a file."""
    try:
        # Ensure the directory for the file exists.
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        return {"error": f"Failed to write to file '{file_path}': {e}"}

def execute_script(file_path: str, args: list[str] = None) -> dict:
    """Executes a script and returns a structured output."""
    if args is None:
        args = []
    print(f"Executing script: {file_path} with args: {args}")
    if not os.path.exists(file_path):
        return {"status": "error", "error": f"Script not found at path: {file_path}"}
    try:
        result = subprocess.run(
            ["python", file_path] + args, 
            capture_output=True, 
            text=True, 
            check=True,
            encoding="utf-8"
        )
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "script_error", "stdout": e.stdout, "stderr": e.stderr, "return_code": e.returncode}
    except FileNotFoundError:
        return {"status": "execution_error", "error": "The 'python' command was not found. Is Python installed and in your PATH?"}
    except Exception as e:
        return {"status": "execution_error", "error": str(e)}

def run_shell_command(command: str) -> dict:
    """Executes a shell command and returns the output."""
    print(f"Executing shell command: {command}")
    try:
        args = shlex.split(command)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "command_error", "stdout": e.stdout, "stderr": e.stderr, "return_code": e.returncode}
    except FileNotFoundError:
        return {"status": "execution_error", "error": f"Command not found: '{args[0]}'. Please ensure it is installed and in your PATH."}
    except Exception as e:
        return {"status": "execution_error", "error": str(e)}

def request_human_assistance(request: str) -> str:
    """
    Asks for human assistance when the agent is stuck or needs a new capability.
    """
    return "The human has been notified. Please await their response."
