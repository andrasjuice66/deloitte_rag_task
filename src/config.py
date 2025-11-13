import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the agent system."""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"
    PDF_PATH = DATA_DIR / "gloomhaven_rulebook.pdf"
    SYNTHETIC_DATASET_PATH = DATA_DIR / "synhetic_dataset.json"
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" # "microsoft/phi-1_5"   

    LLM_MODEL = LLM_MODEL_NAME
    LLM_TEMPERATURE = 0.1
    LLM_MAX_LENGTH = 2048
    USE_LOCAL_LLM = True  
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    
    MAX_SEARCH_RESULTS = 3
    
    SYNTHETIC_DATASET_SIZE = 15
    
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    ENABLE_WEB_SEARCH = True
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(exist_ok=True)

