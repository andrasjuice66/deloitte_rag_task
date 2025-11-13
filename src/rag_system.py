"""RAG (Retrieval Augmented Generation) system for the rulebook."""

from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import Config


class RAGSystem:
    """RAG system for retrieving relevant rules from the Gloomhaven rulebook."""
    
    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        vector_store_path: Optional[Path] = None,
    ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_path: Path to the rulebook PDF
            embedding_model: Name of the embedding model to use
            vector_store_path: Path to save/load the vector store
        """
        self.pdf_path = pdf_path or Config.PDF_PATH
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.vector_store_path = vector_store_path or Config.VECTOR_STORE_DIR
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []
        
    def load_and_process_pdf(self) -> List[Document]:
        """
        Load and process the PDF into chunks.
        
        Returns:
            List of document chunks
        """
        print(f"Loading PDF from {self.pdf_path}...")
        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            # length_function=len,
        )
        
        self.documents = text_splitter.split_documents(documents)
        print(f"Created {len(self.documents)} chunks.")
        
        return self.documents
    
    def create_vectorstore(self):
        """
        Create a vector store from documents.
        """
        docs = self.documents

        
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self.embeddings
        )
        print("Vector store created successfully.")
    
    def save_vectorstore(self):
        """Save the vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("No vector store to save.")
        
        save_path = self.vector_store_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving vector store to {save_path}...")
        self.vectorstore.save_local(str(save_path))
        print("Vector store saved successfully.")
    
    def load_vectorstore(self):
        """Load the vector store from disk."""
        load_path =  self.vector_store_path
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        print(f"Loading vector store from {load_path}...")
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        k = k or Config.TOP_K_RETRIEVAL
        return self.vectorstore.similarity_search(query, k=k)
    
    def setup(self, force_recreate: bool = False):
        """
        Setup the RAG system (load or create vector store).
        
        Args:
            force_recreate: If True, recreate the vector store even if it exists
        """
        Config.ensure_directories()
        
        if force_recreate or not self.vector_store_path.exists():
            self.load_and_process_pdf()
            self.create_vectorstore()
            self.save_vectorstore()
        else:
            try:
                self.load_vectorstore()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Recreating vector store...")
                self.load_and_process_pdf()
                self.create_vectorstore()
                self.save_vectorstore()

