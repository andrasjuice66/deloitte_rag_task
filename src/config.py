"""Configuration for the Gloomhaven rulebook agent system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the agent system."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"
    PDF_PATH = DATA_DIR / "gloomhaven_rulebook.pdf"
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  
    LLM_TEMPERATURE = 0.1
    LLM_MAX_LENGTH = 2048
    
    # RAG configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    
    # Web search configurations
    MAX_SEARCH_RESULTS = 3
    
    # Evaluation
    SYNTHETIC_DATASET_SIZE = 15
    
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    


