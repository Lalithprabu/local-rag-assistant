"""
agent.py — a minimal agent that can decide to use a calculator tool
before answering, using Ollama's function-calling support.
"""

from retrieve import retrieve_chunks
import ollama

MODEL = "llama3.2"


def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    try:
        allowed = "0123456789+-*/(). "
        if not all(ch in allowed for ch in expression):
            return "Error: expression contains disallowed characters."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def search_documents(query: str) -> str:
    """Search the local document store and return matching chunks."""
    chunks = retrieve_chunks(query)
    if not chunks:
        return "No relevant documents found."
    return "\n\n".join(chunks)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate ONLY arithmetic math expressions containing numbers and operators like + - * / ( ). Do NOT use this for general knowledge questions, facts, dates, places, or anything that is not a calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using + - * / ( )",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the local document collection for information relevant to a question. Use this for questions that might be answered by the user's own stored documents, such as topics like sourdough bread or Ollama.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, usually the user's question or a key phrase from it",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

import re


def looks_like_math(question: str) -> bool:
    """Heuristic: only offer the calculator tool if the question actually
    contains digits and a math operator. Small local models sometimes call
    tools even when they shouldn't, so we don't fully trust the model's own
    judgment here — this is a cheap, explicit guard in code."""
    has_digit = bool(re.search(r"\d", question))
    has_operator = bool(re.search(r"[\+\-\*/]", question))
    return has_digit and has_operator


def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    #response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    tools_to_offer = []
    if looks_like_math(question):
        tools_to_offer.append(TOOLS[0])  # calculator
    tools_to_offer.append(TOOLS[1])  # always offer document search
    response = ollama.chat(model=MODEL, messages=messages, tools=tools_to_offer)
    message = response["message"]

    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if name == "calculator":
                print(f"[Agent decided to use calculator with: {args['expression']}]")
                result = calculator(args["expression"])
                print(f"[Calculator returned: {result}]")
            elif name == "search_documents":
                print(f"[Agent decided to search documents with: {args['query']}]")
                result = search_documents(args["query"])
                print(f"[Search returned: {result[:100]}...]")
            else:
                result = f"Unknown tool: {name}"

            messages.append(message)
            messages.append({
                "role": "tool",
                "content": result,
            })
        final = ollama.chat(model=MODEL, messages=messages)
        return final["message"]["content"]

    print("[Agent answered directly, no tool used]")
    return message["content"]


if __name__ == "__main__":
    question = input("Ask something: ")
    print(f"\n{run_agent(question)}\n")