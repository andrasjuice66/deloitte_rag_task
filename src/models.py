"""Data models for the Gloomhaven rulebook agent system."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RuleCategoryEnum(str, Enum):
    """Categories of rules in the Gloomhaven rulebook."""
    BOARD_GAME_SETUP = "BoardGameSetup"
    COMBAT = "Combat"
    SCENARIO = "Scenario"
    CHARACTER = "Character"


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    explanation: str = Field(
        description="Detailed explanation based on the rules"
    )
    is_correct: bool = Field(
        description="Whether the user handled the situation correctly"
    )
    category: RuleCategoryEnum = Field(
        description="The relevant aspect of the rulebook"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score of the answer (0-1)"
    )
    source: str = Field(
        default="rulebook",
        description="Source of the answer (rulebook or web)"
    )


class QuestionAnswerPair(BaseModel):
    """Question and answer pair for evaluation."""
    question: str = Field(
        description="User's question about a game situation"
    )
    situation: str = Field(
        description="Description of the game situation"
    )
    expected_answer: AgentResponse = Field(
        description="Expected correct answer"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating the agent on a question."""
    question: str
    predicted: AgentResponse
    expected: AgentResponse
    is_correct_match: bool
    category_match: bool
    overall_correct: bool

