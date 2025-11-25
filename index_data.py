from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# NOTE: The modern package for Chroma integration
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings



# --- Load Environment Variables ---
# Loads DISCORD_BOT_TOKEN, OPENAI_API_KEY, and NEW Chroma Cloud variables
load_dotenv()

# --- Configuration ---
DATA_DIR = "./articles"  # Directory containing your .txt files
# CHROMA_PATH is now unused as data is remote
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "accurate_articles" # This must match the collection name in your Chroma Cloud DB

# --- Chroma Cloud Credentials (Fetched from .env) ---
# Ensure these are set in your .env file!
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


def load_documents():
    """Load all .txt files from the specified directory."""
    documents = []
    print(f"Loading documents from {DATA_DIR}...")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_documents(documents):
    """Split documents into chunks suitable for accurate figure retrieval."""
    # Using the same chunking strategy for consistency
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def index_data(chunks):
    """Create embeddings and store them in Chroma Cloud."""
    
    # 1. Check for Cloud Credentials
    if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
        print("\nFATAL ERROR: One or more Chroma Cloud environment variables are missing.")
        print("Please ensure CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE are set in your .env file.")
        return

    # 2. Initialize the embedding model
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # 3. Create the Chroma store (and upload data) using cloud parameters
    print(f"\nIndexing {len(chunks)} chunks into Chroma Cloud...")
    try:
        # Chroma.from_documents handles the batch embedding and upload process
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            collection_name=COLLECTION_NAME,
            # --- CLOUD CONNECTION PARAMETERS ---
            chroma_cloud_api_key=CHROMA_API_KEY, 
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
            # -----------------------------------
        )
        # Note: db.persist() is removed as the data is saved in the cloud.
        print("Indexing complete! Data is now available in Chroma Cloud.")
        return db
        
    except Exception as e:
        print(f"CRITICAL ERROR during indexing to Chroma Cloud: {e}")
        print("Check your API Key, Tenant, Database, and Collection Name.")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}. Please place your .txt articles inside.")
    else:
        docs = load_documents()
        if docs:
            chunks = split_documents(docs)
            index_data(chunks)
        else:
            print(f"No .txt files found in {DATA_DIR}. Please add your data.")