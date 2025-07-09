import os
import chromadb
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Define constants
KNOWLEDGE_BASE_DIR = "knowledge_base"
DB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")
SOURCE_DOCS_DIR = "source_documents" # A directory to hold raw files for indexing
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    """
    Main function to build or update the vector store.
    This script will:
    1. Check for source documents.
    2. Load them.
    3. Split them into chunks.
    4. Generate embeddings and store them in a persistent ChromaDB.
    """
    print("Starting indexer...")

    # --- 1. Setup ---
    # Ensure source and DB directories exist
    if not os.path.exists(SOURCE_DOCS_DIR):
        os.makedirs(SOURCE_DOCS_DIR)
        print(f"Created source documents directory at '{SOURCE_DOCS_DIR}'.")
        # Create a placeholder file to prevent errors on the first run
        with open(os.path.join(SOURCE_DOCS_DIR, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder file. Add text documents here to be indexed.")
        print("Please add documents to be indexed in this directory.")

    # --- 2. Load Documents ---
    print(f"Loading documents from '{SOURCE_DOCS_DIR}'...")
    documents = []
    for filename in os.listdir(SOURCE_DOCS_DIR):
        file_path = os.path.join(SOURCE_DOCS_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        # Add other loaders here (e.g., for .docx, .csv)
    
    if not documents:
        print("No documents found to index. Exiting.")
        return

    # --- 3. Chunk Documents ---
    # The plan mentions AST-based chunking; we'll start with a simpler text splitter.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # --- 4. Create Embeddings and Store in ChromaDB ---
    print(f"Creating embeddings using '{EMBEDDING_MODEL}' and storing in ChromaDB at '{DB_DIR}'...")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_function,
        persist_directory=DB_DIR,
    )

    print(f"Indexing complete. Vector store with {db._collection.count()} items saved to '{DB_DIR}'.")

if __name__ == "__main__":
    main()