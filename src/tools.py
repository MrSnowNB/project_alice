
import os
import shlex
import subprocess
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import CrossEncoder
from googlesearch import search
from bs4 import BeautifulSoup

# Define constants consistent with the indexer
KNOWLEDGE_BASE_DIR = "knowledge_base"
DB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
 
def retrieve_from_memory(query: str) -> dict:
    """Searches the agent's knowledge base for information relevant to the query."""
    print(f"Searching memory for: '{query}'")
    if not os.path.exists(DB_DIR):
        return {"error": "Knowledge base not found. Please run the indexer first."}

    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        
        # 1. Retrieve a larger set of documents for reranking (e.g., top 10)
        print("Step 1: Retrieving initial candidates from vector store...")
        retrieved_docs = db.similarity_search(query, k=10)
        
        if not retrieved_docs:
            return {"result": "No relevant information found in memory."}
        
        # 2. Rerank the results using a cross-encoder model
        print("Step 2: Reranking candidates with a cross-encoder...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        doc_contents = [doc.page_content for doc in retrieved_docs]
        
        # The model expects a list of [query, document] pairs
        pairs = [[query, doc] for doc in doc_contents]
        scores = cross_encoder.predict(pairs)
        
        # Combine documents with their new scores and sort
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        
        # 3. Return the top 3 reranked documents
        top_docs = [doc for score, doc in scored_docs[:3]]
        context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
        return {"relevant_context": context}
        
    except Exception as e:
        return {"error": f"An error occurred while retrieving from memory: {e}"}

def add_to_memory(text_to_remember: str) -> dict:
    """
    Adds a piece of text to the agent's long-term memory (knowledge base).
    Use this to remember important facts, user preferences, or successful solutions.
    """
    print(f"Adding to memory: '{text_to_remember}'")
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_function
        )
        db.add_texts(texts=[text_to_remember])
        return {"status": "success", "message": "Information successfully added to memory."}
    except Exception as e:
        return {"error": f"An error occurred while adding to memory: {e}"}

def search_the_web(query: str) -> dict:
    """Searches the web for a query and returns the text content of the top search result."""
    print(f"Searching the web for: '{query}'")
    try:
        # Get the first URL from the search results
        try:
            url = next(search(query, stop=1))
        except StopIteration:
            return {"result": "No search results found."}

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

def execute_script(file_path: str, args: list[str]) -> dict:
    """Executes a script and returns a structured output."""
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
        # This exception is raised when the script has a non-zero exit code.
        return {"status": "script_error", "stdout": e.stdout, "stderr": e.stderr, "return_code": e.returncode}
    except FileNotFoundError:
        # This error means the 'python' command itself was not found.
        return {"status": "execution_error", "error": "The 'python' command was not found. Is Python installed and in your PATH?"}
    except Exception as e:
        return {"status": "execution_error", "error": str(e)}

def run_shell_command(command: str) -> dict:
    """Executes a shell command and returns the output."""
    print(f"Executing shell command: {command}")
    try:
        # Use shlex.split to safely parse the command and avoid shell=True
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
        # Command returned a non-zero exit code
        return {"status": "command_error", "stdout": e.stdout, "stderr": e.stderr, "return_code": e.returncode}
    except FileNotFoundError:
        return {"status": "execution_error", "error": f"Command not found: '{args[0]}'. Please ensure it is installed and in your PATH."}
    except Exception as e:
        return {"status": "execution_error", "error": str(e)}

def request_human_assistance(request: str) -> str:
    """
    Asks for human assistance when the agent is stuck or needs a new capability.
    Use this to request a new tool, ask for clarification, or report an unrecoverable error.
    The human's response will be provided in the next step.
    """
    # This is a placeholder. The graph will intercept this call.
    return "The human has been notified. Please await their response."
