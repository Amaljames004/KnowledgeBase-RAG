import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Settings
PROJECT_NAME = "DevDocs AI"
VERSION = "1.0.0"
DESCRIPTION = "Intelligent Programming Documentation Search Engine"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_documents")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# Create directories if they don't exist
for directory in [RAW_DOCS_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# Document Processing Settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
SUPPORTED_EXTENSIONS = ['.pdf', '.txt']

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, free, good quality
EMBEDDING_DIMENSION = 384

# Retrieval Settings
TOP_K_RESULTS = 5  # Number of chunks to retrieve
RELEVANCE_THRESHOLD = 0.3  # Minimum similarity score (0-1)

# LLM Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and capable

# Generation Settings
MAX_TOKENS = 800
TEMPERATURE = 0.2  # Lower = more focused

# System Prompt
SYSTEM_PROMPT = """You are DevDocs AI, an expert programming documentation assistant.

Your responsibilities:
- Answer questions based ONLY on the provided documentation chunks
- Provide clear, accurate code examples when relevant
- Be concise but comprehensive
- Always cite the specific documentation sources
- If the answer is not in the provided documentation, clearly state that

Response format:
1. Direct answer to the question
2. Code example (if applicable)
3. Source references"""

# Example Queries for UI
EXAMPLE_QUERIES = [
    "How do I create a FastAPI endpoint with path parameters?",
    "What's the difference between useState and useReducer in React?",
    "Show me how to handle async database queries in Python",
    "How do I configure CORS in Express.js?",
    "Explain Python decorators with an example"
]

# API Settings
API_TITLE = PROJECT_NAME
API_VERSION = VERSION
API_DESCRIPTION = DESCRIPTION