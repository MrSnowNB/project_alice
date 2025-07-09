from flask import Flask, request, jsonify
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import CrossEncoder

# --- Configuration ---
app = Flask(__name__)

# Assuming the service is run from the project root
KNOWLEDGE_BASE_DIR = "knowledge_base"
DB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Global Variables to hold loaded models ---
embedding_function = None
db = None
cross_encoder = None

def load_models_and_db():
    """Loads all necessary models and the DB connection into memory on startup."""
    global embedding_function, db, cross_encoder
    
    print("Memory Sub-Agent: Loading models and initializing database connection...")
    
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    
    if os.path.exists(DB_DIR):
        print(f"Memory Sub-Agent: Connecting to existing DB at {DB_DIR}")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    else:
        print(f"Memory Sub-Agent: Warning - DB directory not found at {DB_DIR}. The '/query' endpoint will fail until the indexer is run.")
        # Create a new DB instance. It will be persisted to when the /add endpoint is called.
        db = Chroma(embedding_function=embedding_function, persist_directory=DB_DIR)

    print("Memory Sub-Agent: Models and database loaded successfully.")

@app.route('/query', methods=['POST'])
def query_memory():
    """
    Handles a query to the knowledge base.
    Expects a JSON payload: {"query": "your search query"}
    """
    if db is None:
        return jsonify({"error": "Database not initialized. Please run the indexer."}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query = data['query']
    print(f"Memory Sub-Agent: Received query: '{query}'")

    try:
        retrieved_docs = db.similarity_search(query, k=10)
        
        if not retrieved_docs:
            return jsonify({"relevant_context": []})
        
        doc_contents = [doc.page_content for doc in retrieved_docs]
        pairs = [[query, doc] for doc in doc_contents]
        scores = cross_encoder.predict(pairs)
        
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        
        top_docs = [
            {"content": doc.page_content, "metadata": doc.metadata} 
            for score, doc in scored_docs[:3]
        ]
        return jsonify({"relevant_context": top_docs})
        
    except Exception as e:
        print(f"Memory Sub-Agent: Error during query: {e}")
        return jsonify({"error": f"An error occurred during retrieval: {e}"}), 500

@app.route('/add', methods=['POST'])
def add_to_memory_service():
    """
    Adds text to the knowledge base.
    Expects a JSON payload: {"text_to_remember": "the text to add"}
    """
    if db is None:
        return jsonify({"error": "Database not initialized."}), 500

    data = request.get_json()
    if not data or 'text_to_remember' not in data:
        return jsonify({"error": "Missing 'text_to_remember' in request body"}), 400
    
    text = data['text_to_remember']
    print(f"Memory Sub-Agent: Received text to add: '{text[:100]}...'")

    try:
        db.add_texts(texts=[text])
        db.persist() # Persist changes explicitly after adding
        return jsonify({"status": "success", "message": "Information successfully added to memory."})
    except Exception as e:
        print(f"Memory Sub-Agent: Error during add: {e}")
        return jsonify({"error": f"An error occurred while adding to memory: {e}"}), 500

if __name__ == '__main__':
    load_models_and_db()
    # Using a specific port for the memory service to avoid conflicts
    app.run(port=5001, debug=True)