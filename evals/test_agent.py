"""
test_agent.py — verifies the agent picks the correct tool for each
type of question, and answers directly when no tool is needed.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ragbot.agent import run_agent, looks_like_math


def test_math_question_uses_calculator():
    assert looks_like_math("What is 47 * 23?") is True


def test_geography_question_not_math():
    assert looks_like_math("What is the capital of France?") is False


def test_currency_question_not_flagged_as_math():
    # Should NOT trigger the math heuristic (no operator), but should
    # still work correctly via convert_currency since that tool is
    # always offered regardless of the math heuristic.
    assert looks_like_math("How much is 50 USD in EUR?") is False


def test_agent_answers_math_question():
    answer = run_agent("What is 12 * 8?")
    assert "96" in answer


def test_agent_answers_document_question():
    answer = run_agent("How does sourdough bread rise?")
    assert "yeast" in answer.lower() or "bacteria" in answer.lower()


def test_agent_answers_currency_question():
    answer = run_agent("How much is 10 USD in EUR?")
    assert "EUR" in answer.upper() or "euro" in answer.lower()