"""
test_rag.py — evaluates the RAG pipeline's faithfulness and relevancy
using DeepEval, judged by a local Ollama model (no external API calls).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ragbot.retrieve import retrieve_chunks, generate_answer
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OllamaModel
judge_model = OllamaModel(model="llama3.2", base_url="http://localhost:11434")


def run_rag(question: str):
    chunks = retrieve_chunks(question)
    answer = generate_answer(question, chunks)
    return answer, chunks


def test_sourdough_question():
    question = "How does sourdough bread rise?"
    answer, chunks = run_rag(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=chunks,
    )

    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_model)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    assert_test(test_case, [faithfulness, relevancy])


def test_unrelated_question_says_dont_know():
    """Safety check: an unrelated question should not produce a hallucinated answer."""
    question = "What is the current weather in London?"
    answer, chunks = run_rag(question)

    assert "don't know" in answer.lower() or "not" in answer.lower(), (
        f"Expected a refusal for an unanswerable question, got: {answer}"
    )