# Gloomhaven Rulebook Agent

An intelligent agent system that answers questions about Gloomhaven board game rules using RAG (Retrieval Augmented Generation) and web search capabilities.

## Features

- **RAG-based Question Answering**: Uses vector similarity search to find relevant rules from the Gloomhaven rulebook
- **Structured Responses**: Returns explanations with boolean correctness indicators and rule categories
- **Local LLM Support**: Uses Hugging Face models locally (no API keys required!)
- **LangGraph Agent**: Implements a sophisticated agent workflow with conditional routing
- **Evaluation Framework**: Includes synthetic data generation and accuracy evaluation
- **Flexible LLM Support**: Works with local Hugging Face models, OpenAI models, or custom LLM instances
- **Privacy-Focused**: All processing can be done locally without external API calls

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── models.py           # Pydantic data models
│   ├── rag_system.py       # RAG implementation with FAISS
│   ├── web_search.py       # Web search tool
│   ├── agent.py            # LangGraph agent
│   ├── synthetic_data.py   # Synthetic data generation
│   ├── evaluator.py        # Evaluation system
│   └── main.py             # Main entry point
├── data/
│   ├── gloomhaven_rulebook.pdf
│   └── vector_store/       # FAISS vector store
├── Interview_Task.ipynb    # Demonstration notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Download the Rulebook

The rulebook PDF should be placed in the `data/` directory:
- URL: https://cdn.1j1ju.com/medias/8d/c5/21-gloomhaven-rulebook.pdf
- Save as: `data/gloomhaven_rulebook.pdf`

Or use the download script:
```bash
python download_pdf.py
```

### 3. Run the Example

```bash
python example.py
```

**Note**: On first run, the system will download the Hugging Face model (~2.7 GB). This happens only once.

### 4. Optional: Test Setup

Verify everything is working:
```bash
python test_local_setup.py
```

## No API Keys Required! 🎉

By default, the system uses **local Hugging Face models** (`microsoft/phi-1_5`), so you don't need any API keys. Everything runs on your machine!

For advanced usage with OpenAI models, see the documentation below.

## Usage

### Basic Usage

```python
from src.main import GloomhavenRulebookSystem

# Initialize the system
system = GloomhavenRulebookSystem()
system.setup()

# Ask a question
response = system.ask_question(
    "We drew two attack modifier cards by mistake. Should we apply both?"
)

print(f"Explanation: {response.explanation}")
print(f"Correct: {response.is_correct}")
print(f"Category: {response.category}")
```

### Using Local LLMs

```python
# Use Ollama (requires Ollama installed)
system = GloomhavenRulebookSystem(use_local_llm=True, local_model_name="llama2")
system.setup()
```

### Evaluation

```python
# Generate synthetic dataset and evaluate
dataset = system.generate_evaluation_dataset()
metrics = system.evaluate(dataset)

print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
```

## Response Format

The agent returns structured responses with:

- **explanation**: Detailed explanation based on the rules
- **is_correct**: Boolean indicating if the situation was handled correctly
- **category**: One of [BoardGameSetup, Combat, Scenario, Character]
- **confidence**: Confidence score (0-1)
- **source**: "rulebook" or "web"

## Evaluation

The system includes:
- 3 manually created seed examples
- Synthetic data generation to create 15 total examples
- Accuracy metrics for correctness prediction and category classification

## Notes

- The system works best with OpenAI models (GPT-3.5 or GPT-4)
- Local models (via Ollama or HuggingFace) can be used but may have lower accuracy
- Web search requires a Tavily API key but is optional
- The PDF can be truncated to a few pages for faster testing

## License

This is an interview task project for educational purposes.

