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
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

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

    # NOTE: FaithfulnessMetric is excluded from the automated assertion here.
    # llama3.2, used as the local judge model (no external API cost), was
    # found to give unreliable, self-contradictory faithfulness scores
    # (e.g. scoring 0.0 with a reasoning string that logically implied a
    # high score). Manual verification confirmed the actual answer was
    # faithfully grounded in the retrieved context. A production system
    # would use a stronger judge model (e.g. GPT-4-class) for this metric.
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)
    faithfulness_geval = GEval(
        name="Faithfulness",
        criteria="Determine whether the actual output contains only information that is directly supported by the retrieval context. The output should not add facts, numbers, or claims not present in the context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.5,
        model=judge_model,
    )

    #assert_test(test_case, [relevancy])
    assert_test(test_case, [relevancy, faithfulness_geval])

    #faithfulness = FaithfulnessMetric(threshold=0.3, model=judge_model)
    #relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    #assert_test(test_case, [faithfulness, relevancy])


def test_unrelated_question_says_dont_know():
    """Safety check: an unrelated question should not produce a hallucinated answer."""
    question = "What is the current weather in London?"
    answer, chunks = run_rag(question)

    assert "don't know" in answer.lower() or "not" in answer.lower(), (
        f"Expected a refusal for an unanswerable question, got: {answer}"
    )

def test_sourdough_faithfulness_geval():
    question = "How does sourdough bread rise?"
    answer, chunks = run_rag(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=chunks,
    )

    faithfulness_geval = GEval(
        name="Faithfulness",
        criteria="Determine whether the actual output contains only information that is directly supported by the retrieval context. The output should not add facts, numbers, or claims not present in the context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.5,
        model=judge_model,
    )

    assert_test(test_case, [faithfulness_geval])