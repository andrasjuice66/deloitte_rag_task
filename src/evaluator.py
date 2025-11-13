"""Evaluation system for the Gloomhaven agent."""

from typing import List
import time
import json
from pathlib import Path

from .models import QuestionAnswerPair, EvaluationResult, AgentResponse
from .agent import GloomhavenAgent
from .config import Config


class AgentEvaluator:
    """Evaluate the agent's performance on question-answer pairs."""
    
    def __init__(self, agent: GloomhavenAgent):
        """
        Initialize the evaluator.
        
        Args:
            agent: The agent to evaluate
        """
        self.agent = agent
    
    def evaluate_single(
        self,
        qa_pair: QuestionAnswerPair
    ) -> EvaluationResult:
        """
        Evaluate the agent on a single question-answer pair.
        
        Args:
            qa_pair: Question-answer pair to evaluate
            
        Returns:
            Evaluation result
        """
        # Get agent's prediction
        predicted = self.agent.answer_question(qa_pair.question)
        expected = qa_pair.expected_answer
        
        # Check if predictions match
        is_correct_match = predicted.is_correct == expected.is_correct
        category_match = predicted.category == expected.category
        overall_correct = is_correct_match and category_match
        
        return EvaluationResult(
            question=qa_pair.question,
            predicted=predicted,
            expected=expected,
            is_correct_match=is_correct_match,
            category_match=category_match,
            overall_correct=overall_correct
        )
    
    def evaluate_dataset(
        self,
        dataset: List[QuestionAnswerPair],
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the agent on a dataset.
        
        Args:
            dataset: List of question-answer pairs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = []
        
        for i, qa_pair in enumerate(dataset):
            if verbose:
                print(f"\nEvaluating question {i+1}/{len(dataset)}...")
                print(f"Q: {qa_pair.question[:100]}...")
            
            try:
                result = self.evaluate_single(qa_pair)
                results.append(result)
                
                if verbose:
                    print(f"Expected is_correct: {result.expected.is_correct}")
                    print(f"Predicted is_correct: {result.predicted.is_correct}")
                    print(f"Match: {result.is_correct_match}")
                
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
                continue
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate metrics
        total = len(results)
        is_correct_matches = sum(1 for r in results if r.is_correct_match)
        category_matches = sum(1 for r in results if r.category_match)
        overall_correct = sum(1 for r in results if r.overall_correct)
        
        metrics = {
            "total_questions": total,
            "is_correct_accuracy": is_correct_matches / total if total > 0 else 0,
            "category_accuracy": category_matches / total if total > 0 else 0,
            "overall_accuracy": overall_correct / total if total > 0 else 0,
            "results": results
        }
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Total questions: {total}")
            print(f"Is Correct Accuracy: {metrics['is_correct_accuracy']:.2%}")
            print(f"Category Accuracy: {metrics['category_accuracy']:.2%}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        
        return metrics
    
    def print_detailed_results(self, metrics: dict):
        """
        Print detailed results of evaluation.
        
        Args:
            metrics: Metrics dictionary from evaluate_dataset
        """
        results = metrics["results"]
        
        print("\n" + "="*50)
        print("DETAILED RESULTS")
        print("="*50)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {result.question[:100]}...")
            print(f"\nExpected:")
            print(f"  - is_correct: {result.expected.is_correct}")
            print(f"  - category: {result.expected.category.value}")
            print(f"\nPredicted:")
            print(f"  - is_correct: {result.predicted.is_correct}")
            print(f"  - category: {result.predicted.category.value}")
            print(f"  - source: {result.predicted.source}")
            print(f"\nMatches:")
            print(f"  - is_correct: {'✓' if result.is_correct_match else '✗'}")
            print(f"  - category: {'✓' if result.category_match else '✗'}")
            print(f"  - overall: {'✓' if result.overall_correct else '✗'}")
    
    @staticmethod
    def load_dataset(file_path: Path = None) -> List[QuestionAnswerPair]:
        """
        Load a dataset from JSON into QuestionAnswerPair objects.
        
        Args:
            file_path: Optional path override; defaults to Config.SYNTHETIC_DATASET_PATH
        
        Returns:
            List of QuestionAnswerPair
        """
        path = file_path or Config.SYNTHETIC_DATASET_PATH
        with open(path, "r") as f:
            data = json.load(f)
        return [QuestionAnswerPair(**item) for item in data]

