import ollama
import json
import re
from logger import log_info, log_error


def safe_parse_json(response_text):
    try:
        return json.loads(response_text)
    except:
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None

def calculate_confidence(test_case):
    score = 100

    if len(test_case.get("steps", [])) < 3:
        score -= 30

    if "edge" not in test_case.get("description", "").lower():
        score -= 20

    return max(score, 0)

def call_llm(user_story, model="llama3.1:8b"):
    prompt = f"""
    You are a QA expert.

    Generate EXACTLY 5 test cases in STRICT JSON format.

    Return ONLY valid JSON. No explanation.
    Include:
    - Positive test cases
    - Negative test cases
    - Edge cases
    - Boundary conditions

    Also include:
    "priority": "High/Medium/Low"

    Format:
    [
      {{
        "test_case_id": "TC001",
        "description": "",
        "steps": ["", "", ""],
        "expected_result": "",
        "priority": "High"
    }}
    ]

    User Story:
    {user_story}
    """

    log_info(f"Calling LLM with model: {model}")

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


def generate_test_cases(user_story, retries=3, model="llama3.1:8b"):
    for i in range(retries):
        log_info(f"Attempt {i+1}")

        raw_output = call_llm(user_story, model=model)
        parsed_output = safe_parse_json(raw_output)
        if parsed_output:
            for tc in parsed_output:
                tc["confidence_score"] = calculate_confidence(tc)
            log_info("Parsing successful")
            return raw_output, parsed_output

        log_error("Parsing failed, retrying...")

    log_error("All retries failed")

    return raw_output, None