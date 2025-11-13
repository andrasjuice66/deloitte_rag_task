"""Main entry point for the Gloomhaven rulebook agent system."""

from typing import Optional
from pathlib import Path

from .config import Config
from .rag_system import RAGSystem
from .web_search import WebSearchTool
from .agent import GloomhavenAgent
from .synthetic_data import SyntheticDataGenerator
from .evaluator import AgentEvaluator
from .models import AgentResponse


class GloomhavenRulebookSystem:
    """Main system that orchestrates all components."""
    
    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        use_huggingface: bool = True,
        model_name: str = None,
        llm: Optional[any] = None
    ):
        """
        Initialize the Gloomhaven rulebook system.
        
        Args:
            pdf_path: Path to the rulebook PDF
            use_huggingface: Whether to use a local Hugging Face model (default: True)
            model_name: Name of Hugging Face model (defaults to Config.LLM_MODEL)
            llm: Custom LLM instance (overrides use_huggingface)
        """
        self.pdf_path = pdf_path or Config.PDF_PATH
        self.use_huggingface = use_huggingface
        self.model_name = model_name or Config.LLM_MODEL
        self.custom_llm = llm
        
        # Initialize components
        self.rag_system: Optional[RAGSystem] = None
        self.web_search_tool: Optional[WebSearchTool] = None
        self.agent: Optional[GloomhavenAgent] = None
        self.data_generator: Optional[SyntheticDataGenerator] = None
        self.evaluator: Optional[AgentEvaluator] = None
        
    def setup(self, force_recreate_vectorstore: bool = False):
        """
        Setup all system components.
        
        Args:
            force_recreate_vectorstore: Whether to recreate the vector store
        """
        print("Setting up Gloomhaven Rulebook Agent System...")
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Initialize RAG system
        print("\n1. Initializing RAG system...")
        self.rag_system = RAGSystem(pdf_path=self.pdf_path)
        self.rag_system.setup(force_recreate=force_recreate_vectorstore)
        
        # Initialize web search tool (disabled by default for local-only mode)
        if Config.ENABLE_WEB_SEARCH:
            print("\n2. Initializing web search tool...")
            self.web_search_tool = WebSearchTool()
        else:
            print("\n2. Web search disabled (using local models only)")
            self.web_search_tool = None
        
        # Initialize agent
        print("\n3. Initializing agent...")
        self.agent = GloomhavenAgent(
            rag_system=self.rag_system,
            web_search_tool=self.web_search_tool,
            llm=self.custom_llm,
            use_huggingface=self.use_huggingface,
            model_name=self.model_name
        )
        
        # Initialize data generator (note: synthetic data generation may not work well with small models)
        print("\n4. Initializing synthetic data generator...")
        self.data_generator = SyntheticDataGenerator(
            llm=self.custom_llm,
            use_huggingface=self.use_huggingface,
            model_name=self.model_name
        )
        
        # Initialize evaluator
        print("\n5. Initializing evaluator...")
        self.evaluator = AgentEvaluator(agent=self.agent)
        
        print("\n✓ System setup complete!")
    
    def ask_question(self, question: str) -> AgentResponse:
        """
        Ask a question to the agent.
        
        Args:
            question: The question to ask
            
        Returns:
            Structured response from the agent
        """
        if self.agent is None:
            raise ValueError("System not initialized. Call setup() first.")
        
        return self.agent.answer_question(question)
    
    def generate_evaluation_dataset(self, save_path: Optional[Path] = None):
        """
        Generate a synthetic evaluation dataset.
        
        Args:
            save_path: Path to save the dataset (optional)
            
        Returns:
            List of question-answer pairs
        """
        if self.data_generator is None:
            raise ValueError("System not initialized. Call setup() first.")
        
        print("Generating evaluation dataset...")
        
        # Create seed examples
        seed_examples = self.data_generator.create_seed_examples()
        print(f"Created {len(seed_examples)} seed examples.")
        
        # Generate synthetic examples
        dataset = self.data_generator.generate_synthetic_dataset(
            seed_examples=seed_examples,
            num_examples=Config.SYNTHETIC_DATASET_SIZE
        )
        print(f"Generated {len(dataset)} total examples.")
        
        # Save if path provided
        if save_path:
            import json
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(
                    [qa.dict() for qa in dataset],
                    f,
                    indent=2
                )
            print(f"Saved dataset to {save_path}")
        
        return dataset
    
    def evaluate(self, dataset=None, verbose: bool = True):
        """
        Evaluate the agent on a dataset.
        
        Args:
            dataset: Dataset to evaluate on. If None, generates a new one.
            verbose: Whether to print detailed results
            
        Returns:
            Evaluation metrics
        """
        if self.evaluator is None:
            raise ValueError("System not initialized. Call setup() first.")
        
        if dataset is None:
            print("No dataset provided. Generating one...")
            dataset = self.generate_evaluation_dataset()
        
        print(f"\nEvaluating agent on {len(dataset)} questions...")
        metrics = self.evaluator.evaluate_dataset(dataset, verbose=verbose)
        
        if verbose:
            self.evaluator.print_detailed_results(metrics)
        
        return metrics

