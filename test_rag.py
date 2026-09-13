from agent import query_rag
from langchain_google_genai import ChatGoogleGenerativeAI

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_CSCI2720_Prerequisite():
    assert query_and_validate(
        question="Is CSCI 2670 a prerequisite of CSCI 2720?",
        expected_response="No",
    )


def test_CS_Intended_Major_Requirements():
    assert query_and_validate(
        question="",
        expected_response=""
    )

def test_():
    assert query_and_validate(
        question="",
        expected_response=""
    )

def test_():
    assert query_and_validate(
        question="",
        expected_response=""
    )

def test_():
    assert query_and_validate(
        question="",
        expected_response=""
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )