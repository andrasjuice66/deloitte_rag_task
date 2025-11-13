"""Synthetic data generation for evaluation."""

from typing import List, Optional
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from .models import QuestionAnswerPair, AgentResponse, RuleCategoryEnum
from .config import Config


class SyntheticDataGenerator:
    """Generate synthetic question-answer pairs for evaluation."""
    
    def __init__(
        self,
        llm: Optional[any] = None,
        use_huggingface: bool = True,
        model_name: str = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            llm: Language model to use
            use_huggingface: Whether to use a local Hugging Face model
            model_name: Name of Hugging Face model (defaults to Config.LLM_MODEL)
        """
        if llm is not None:
            self.llm = llm
        elif use_huggingface or Config.USE_LOCAL_LLM:
            model_name = model_name or Config.LLM_MODEL
            print(f"Loading Hugging Face model for data generation: {model_name}")
            self.llm = self._create_huggingface_llm(model_name)
        else:
            # Fallback - not recommended per user request
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    def _create_huggingface_llm(self, model_name: str):
        """Create a Hugging Face LLM pipeline."""
        device = 0 if torch.cuda.is_available() else -1
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device_str}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Build pipeline kwargs and avoid passing `device` if model is managed by accelerate (hf_device_map set)
        pipeline_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "max_length": Config.LLM_MAX_LENGTH,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
        }
        if getattr(model, "hf_device_map", None) is None:
            pipeline_kwargs["device"] = device
        
        pipe = pipeline("text-generation", **pipeline_kwargs)
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def create_seed_examples(self) -> List[QuestionAnswerPair]:
        """
        Create 3 seed examples for synthetic data generation.
        
        Returns:
            List of 3 seed question-answer pairs
        """
        seed_examples = [
            QuestionAnswerPair(
                question="We played a scenario where a player drew two attack modifier cards by mistake. We applied both modifiers. Was this correct?",
                situation="During combat, a player accidentally drew two attack modifier cards instead of one and applied both modifiers to their attack.",
                expected_answer=AgentResponse(
                    explanation="According to the Gloomhaven rules, when performing an attack, a player should draw exactly one attack modifier card from their deck. If multiple cards are drawn by mistake, only the first card drawn should be applied, and the other cards should be shuffled back into the deck. Applying both modifiers is incorrect.",
                    is_correct=False,
                    category=RuleCategoryEnum.COMBAT,
                    confidence=1.0,
                    source="rulebook"
                )
            ),
            QuestionAnswerPair(
                question="We set up the scenario and placed all monsters on the board at once, including those in unrevealed rooms. Is this the right way?",
                situation="During scenario setup, all monsters were placed on the board immediately, including those in rooms that haven't been revealed yet.",
                expected_answer=AgentResponse(
                    explanation="This is incorrect. In Gloomhaven, monsters should only be placed when their room is revealed. At the start of a scenario, only place monsters in the starting room or any rooms that are visible from the start. When players open a door and reveal a new room, then the monsters for that room should be placed according to the scenario setup guide.",
                    is_correct=False,
                    category=RuleCategoryEnum.SCENARIO,
                    confidence=1.0,
                    source="rulebook"
                )
            ),
            QuestionAnswerPair(
                question="A character used a lost card ability, and we placed it in the lost pile. Later, they took a long rest and shuffled all their discard pile including the lost card back into their hand. Was this correct?",
                situation="After using a lost card ability, the card was placed in the lost pile. During a long rest, the player shuffled all cards from both the discard pile and lost pile back into their hand.",
                expected_answer=AgentResponse(
                    explanation="This is incorrect. In Gloomhaven, cards placed in the lost pile due to lost actions cannot be recovered through normal rest actions (short or long rest). Lost cards are only returned under specific circumstances, such as certain special abilities or items that specifically allow recovery of lost cards. During a long rest, only cards in the discard pile should be shuffled back into the player's hand.",
                    is_correct=False,
                    category=RuleCategoryEnum.CHARACTER,
                    confidence=1.0,
                    source="rulebook"
                )
            )
        ]
        
        return seed_examples
    
    def generate_synthetic_dataset(
        self,
        seed_examples: List[QuestionAnswerPair],
        num_examples: int = 15
    ) -> List[QuestionAnswerPair]:
        """
        Generate synthetic examples based on seed examples.
        
        Args:
            seed_examples: Seed examples to base generation on
            num_examples: Number of examples to generate
            
        Returns:
            List of synthetic question-answer pairs
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert on the Gloomhaven board game. 
Generate realistic question-answer pairs about game situations where players might be unsure if they played correctly.

Based on the provided examples, create {num_new} new similar question-answer pairs.
Each pair should include:
1. A realistic question from a player about a game situation
2. A description of the situation
3. An expected answer with:
   - A detailed explanation based on the rules
   - Whether the situation was handled correctly (true/false)
   - The category: BoardGameSetup, Combat, Scenario, or Character
   - Confidence score (0.8-1.0 for clear rules)
   - Source (always "rulebook")

Vary the situations and cover different aspects of the game. Include both correct and incorrect scenarios.

Provide the output as a JSON array of objects with this structure:
[
  {{
    "question": "...",
    "situation": "...",
    "expected_answer": {{
      "explanation": "...",
      "is_correct": true/false,
      "category": "...",
      "confidence": 0.9,
      "source": "rulebook"
    }}
  }},
  ...
]

Example format based on these seed examples:
{examples}"""),
            ("human", "Generate {num_new} new diverse question-answer pairs.")
        ])
        
        # Format seed examples
        examples_json = json.dumps(
            [ex.dict() for ex in seed_examples],
            indent=2
        )
        
        num_new = num_examples - len(seed_examples)
        
        messages = prompt.format_messages(
            examples=examples_json,
            num_new=num_new
        )
        
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            response_text = response.content
            
            # Extract JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            elif "[" in response_text:
                # Find the JSON array
                start = response_text.index("[")
                end = response_text.rindex("]") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text
            
            synthetic_data = json.loads(json_str)
            
            # Convert to QuestionAnswerPair objects
            synthetic_pairs = []
            for item in synthetic_data:
                try:
                    pair = QuestionAnswerPair(**item)
                    synthetic_pairs.append(pair)
                except Exception as e:
                    print(f"Error parsing synthetic example: {e}")
                    continue
            
            # Combine seed examples with synthetic examples
            all_examples = seed_examples + synthetic_pairs
            
            return all_examples[:num_examples]
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            print(f"Response: {response.content}")
            # Return just the seed examples if generation fails
            return seed_examples

