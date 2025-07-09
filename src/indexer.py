import os
import chromadb
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Define constants
KNOWLEDGE_BASE_DIR = "knowledge_base"
DB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")
SOURCE_DIRS = ["source_documents", "src"] # A list of directories to index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    """
    Main function to build or update the vector store.
    This script will:
    1. Check for source documents in multiple directories.
    2. Load them based on file type.
    3. Split them into chunks.
    4. Generate embeddings and store them in a persistent ChromaDB.
    """
    print("Starting indexer...")

    # --- 1. Setup ---
    # Ensure source directories exist
    for directory in SOURCE_DIRS:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created source directory at '{directory}'.")

    # --- 2. Load and Chunk Documents ---
    print(f"Loading and chunking documents from {SOURCE_DIRS}...")
    chunked_docs = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )

    for source_dir in SOURCE_DIRS:
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    chunked_docs.extend(text_splitter.split_documents(docs))
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    chunked_docs.extend(text_splitter.split_documents(docs))
                elif filename.endswith(".py"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    chunked_docs.extend(python_splitter.split_documents(docs))
            except Exception as e:
                print(f"Warning: Failed to load or chunk file '{file_path}'. Error: {e}")
                # Continue to the next file
                continue
    
    if not chunked_docs:
        print("No documents found to index. Exiting.")
        return

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