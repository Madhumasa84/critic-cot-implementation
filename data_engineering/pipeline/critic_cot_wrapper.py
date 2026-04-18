"""
Critic-CoT wrapper that turns notebook logic into a reusable pipeline module.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_engineering.config import settings


CALL_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": settings.OPENROUTER_REFERER,
    "X-Title": settings.OPENROUTER_TITLE,
}

BOXED_PATTERN = re.compile(r"\\boxed\s*\{(.+?)\}", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n]+)",
    re.IGNORECASE,
)
STEP_START_PATTERN = re.compile(r"^\s*step\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
ERROR_STEP_PATTERN = re.compile(r"step\s+(\d+)\s+is\s+incorrect", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")
EQUATION_PATTERN = re.compile(
    r"([0-9\$\.,%\s\(\)\+\-\*\/×÷]+?)=\s*(-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?)"
)

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _runtime_config() -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY") or settings.OPENROUTER_API_KEY
    model = os.getenv("MODEL") or settings.MODEL
    base_url = os.getenv("OPENROUTER_BASE_URL") or settings.OPENROUTER_BASE_URL
    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
    }


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = settings.DEFAULT_TEMPERATURE,
    max_tokens: int = settings.DEFAULT_MAX_TOKENS,
    timeout: int = settings.REQUEST_TIMEOUT_SECONDS,
    max_retries: int = 4,
) -> Dict[str, Any]:
    runtime = _runtime_config()
    api_key = runtime["api_key"]
    model_name = model or runtime["model"]

    if not api_key:
        return {
            "ok": False,
            "content": None,
            "error": (
                "OPENROUTER_API_KEY is not configured. "
                "Set it in the environment, project config.py, or .env file."
            ),
            "model": model_name,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    headers = dict(CALL_HEADERS)
    headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()
    last_error = None
    body = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                runtime["base_url"],
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            response.raise_for_status()
            body = response.json()
            last_error = None
            break
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            should_retry = status_code in {429, 500, 502, 503, 504}
            if should_retry and attempt < max_retries - 1:
                retry_after = None
                if exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else float(2 ** attempt * 5)
                except ValueError:
                    wait_seconds = float(2 ** attempt * 5)
                time.sleep(wait_seconds)
                continue
            return {
                "ok": False,
                "content": None,
                "error": str(exc),
                "model": model_name,
                "latency_ms": latency_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(float(2 ** attempt * 2))
                continue
            return {
                "ok": False,
                "content": None,
                "error": str(exc),
                "model": model_name,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        except ValueError as exc:
            return {
                "ok": False,
                "content": None,
                "error": f"Could not decode API response: {exc}",
                "model": model_name,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }

    if body is None:
        return {
            "ok": False,
            "content": None,
            "error": str(last_error) if last_error else "Unknown API error",
            "model": model_name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    total_tokens = int(usage.get("total_tokens") or 0)
    cost_usd = round(total_tokens * settings.resolve_cost_per_token(model_name), 6)

    return {
        "ok": True,
        "content": message.get("content", ""),
        "error": None,
        "model": body.get("model", model_name),
        "latency_ms": latency_ms,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def _generate_solution_call(question: str, model: Optional[str] = None) -> Dict[str, Any]:
    prompt = f"""Solve this math problem step by step.
Write each reasoning line as "Step X:".
End with the final answer in the format \\boxed{{answer}}.

Problem:
{question}

Solution:"""
    result = call_llm(prompt=prompt, model=model, temperature=0.7)
    result["phase"] = "generate"
    return result


def _critique_solution_call(
    question: str,
    solution: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"""Evaluate the following attempt against the problem.

<problem>
{question}
</problem>

<attempt>
{solution}
</attempt>

Instructions:
- Review the reasoning step by step.
- After each checked step, write either "Conclusion: Step [number] is correct"
  or "Conclusion: Step [number] is incorrect".
- Stop at the first incorrect step.
- If the whole solution is correct, say "All steps are correct."

Critique:"""
    result = call_llm(prompt=prompt, model=model, temperature=0.2)
    result["phase"] = "critique"
    return result


def _refine_solution_call(
    question: str,
    solution: str,
    criticism: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"""Refine the following attempt using the criticism.

<problem>
{question}
</problem>

<attempt>
{solution}
</attempt>

<criticism>
{criticism}
</criticism>

Provide a corrected step-by-step solution.
Write each line as "Step X:" and finish with \\boxed{{answer}}.

Corrected solution:"""
    result = call_llm(prompt=prompt, model=model, temperature=0.5)
    result["phase"] = "refine"
    return result


def generate_solution(question: str, model: Optional[str] = None) -> Optional[str]:
    return _generate_solution_call(question, model=model)["content"]


def critique_solution(question: str, solution: str, model: Optional[str] = None) -> Optional[str]:
    return _critique_solution_call(question, solution, model=model)["content"]


def refine_solution(
    question: str,
    solution: str,
    criticism: str,
    model: Optional[str] = None,
) -> Optional[str]:
    return _refine_solution_call(question, solution, criticism, model=model)["content"]


def _serialize_call(call_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase": call_result.get("phase"),
        "ok": call_result.get("ok", False),
        "error": call_result.get("error"),
        "model": call_result.get("model"),
        "latency_ms": call_result.get("latency_ms", 0.0),
        "prompt_tokens": call_result.get("prompt_tokens", 0),
        "completion_tokens": call_result.get("completion_tokens", 0),
        "total_tokens": call_result.get("total_tokens", 0),
        "cost_usd": call_result.get("cost_usd", 0.0),
    }


def _clean_math_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\$", "$").replace("\\%", "%")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    boxed = BOXED_PATTERN.search(text)
    if boxed:
        return _clean_math_text(boxed.group(1))

    final_answer_matches = FINAL_ANSWER_PATTERN.findall(text)
    if final_answer_matches:
        return _clean_math_text(final_answer_matches[-1].strip().rstrip("."))

    number_matches = NUMBER_PATTERN.findall(text)
    if number_matches:
        return _clean_math_text(number_matches[-1])
    return None


def _to_numeric_value(answer: Optional[str]) -> Optional[float]:
    if answer is None:
        return None

    cleaned = _clean_math_text(answer).lower()
    cleaned = cleaned.replace("percent", "%")

    fraction_match = re.fullmatch(r"-?\d+\s*/\s*-?\d+", cleaned)
    if fraction_match:
        return float(Fraction(cleaned.replace(" ", "")))

    numeric_match = NUMBER_PATTERN.search(cleaned)
    if not numeric_match:
        return None

    token = numeric_match.group(0).strip()
    token = token.rstrip("%")
    try:
        return float(token)
    except ValueError:
        return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None

    numeric_value = _to_numeric_value(answer)
    if numeric_value is not None:
        if math.isclose(numeric_value, round(numeric_value), rel_tol=0.0, abs_tol=1e-9):
            return str(int(round(numeric_value)))
        return f"{numeric_value:.10f}".rstrip("0").rstrip(".")

    normalized = _clean_math_text(answer).lower()
    return normalized or None


def answers_match(predicted: Optional[str], expected: Optional[str]) -> bool:
    normalized_predicted = normalize_answer(predicted)
    normalized_expected = normalize_answer(expected)

    if normalized_predicted is None or normalized_expected is None:
        return False
    if normalized_predicted == normalized_expected:
        return True

    numeric_predicted = _to_numeric_value(predicted)
    numeric_expected = _to_numeric_value(expected)
    if numeric_predicted is not None and numeric_expected is not None:
        return math.isclose(
            numeric_predicted,
            numeric_expected,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    return False


def split_steps(solution_text: Optional[str]) -> List[str]:
    if not solution_text:
        return []

    lines = [line.rstrip() for line in solution_text.splitlines() if line.strip()]
    steps: List[str] = []
    current: List[str] = []

    for line in lines:
        if STEP_START_PATTERN.match(line):
            if current:
                steps.append(" ".join(current).strip())
            current = [line.strip()]
        elif current:
            current.append(line.strip())

    if current:
        steps.append(" ".join(current).strip())

    return steps if steps else [line.strip() for line in lines]


def _safe_eval_expression(expression: str) -> Optional[float]:
    sanitized = expression.strip()
    sanitized = sanitized.replace("×", "*").replace("÷", "/").replace("^", "**")
    sanitized = sanitized.replace("$", "").replace(",", "").replace("%", "/100")

    if re.search(r"[^0-9\.\+\-\*\/\(\)\s]", sanitized):
        return None

    try:
        node = ast.parse(sanitized, mode="eval")
        return float(_eval_ast_node(node.body))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _eval_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8 compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](
            _eval_ast_node(node.left),
            _eval_ast_node(node.right),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_ast_node(node.operand))
    raise ValueError("Unsupported expression")


def verify_solution_steps(solution_text: Optional[str]) -> Dict[str, Any]:
    steps = split_steps(solution_text)
    verified: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for index, step_text in enumerate(steps, start=1):
        for match in EQUATION_PATTERN.finditer(step_text):
            expression = match.group(1).strip()
            claimed_text = match.group(2).strip()
            if not any(symbol in expression for symbol in "+-*/×÷"):
                continue

            actual_value = _safe_eval_expression(expression)
            claimed_value = _to_numeric_value(claimed_text)
            if actual_value is None or claimed_value is None:
                continue

            if math.isclose(actual_value, claimed_value, rel_tol=1e-9, abs_tol=1e-9):
                verified.append(
                    {
                        "step": index,
                        "expression": expression,
                        "claimed": claimed_text,
                        "actual": claimed_value,
                    }
                )
            else:
                errors.append(
                    {
                        "step": index,
                        "expression": expression,
                        "claimed": claimed_text,
                        "actual": round(actual_value, 10),
                    }
                )

    return {
        "has_error": bool(errors),
        "errors": errors,
        "verified": verified,
        "checked_steps": len(verified) + len(errors),
        "first_error_step": errors[0]["step"] if errors else None,
    }


def _verification_summary(verification: Dict[str, Any]) -> str:
    if verification["checked_steps"] == 0:
        return ""

    lines = ["Arithmetic verification summary:"]
    for item in verification["verified"]:
        lines.append(
            f"- Step {item['step']}: {item['expression']} = {item['claimed']} verified."
        )
    for item in verification["errors"]:
        lines.append(
            f"- Step {item['step']}: {item['expression']} = {item['claimed']} is incorrect; "
            f"expected {item['actual']}."
        )
    return "\n".join(lines)


def _extract_error_step(critique_text: str) -> Optional[int]:
    match = ERROR_STEP_PATTERN.search(critique_text or "")
    if match:
        return int(match.group(1))
    return None


def _has_error(critique_text: str) -> bool:
    if not critique_text:
        return False

    text = critique_text.lower()
    if re.search(r"step\s+\d+\s+is\s+incorrect", text):
        return True
    if "all steps are correct" in text or "no errors found" in text or "no error found" in text:
        return False

    negative_phrases = (
        " is incorrect",
        " is wrong",
        "mistake",
        "arithmetic error",
        "calculation error",
        "expected ",
        "should be ",
    )
    return any(phrase in text for phrase in negative_phrases)


def _build_solution_payload(solution_text: Optional[str]) -> Dict[str, Any]:
    steps = split_steps(solution_text)
    extracted = extract_answer(solution_text)
    return {
        "full_text": solution_text or "",
        "steps": steps,
        "extracted_answer": extracted,
        "normalized_answer": normalize_answer(extracted),
    }


def _build_trace(
    question: str,
    expected_answer: Optional[str],
    strategy_name: str,
    strategy_params: Dict[str, Any],
    sample_metadata: Optional[Dict[str, Any]],
    model: Optional[str],
) -> Dict[str, Any]:
    runtime = _runtime_config()
    chosen_model = model or runtime["model"]
    sample_metadata = sample_metadata or {}
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "trace_id": f"{strategy_name}_{int(time.time() * 1000)}_{sample_metadata.get('id', 'sample')}",
        "timestamp": timestamp,
        "question": {
            "id": sample_metadata.get("id"),
            "text": question,
            "expected_answer": expected_answer,
            "source": sample_metadata.get("source"),
            "split": sample_metadata.get("split"),
            "metadata": sample_metadata.get("metadata", {}),
        },
        "strategy": {
            "name": strategy_name,
            "parameters": strategy_params,
        },
        "final_answer": None,
        "is_correct": False,
        "iterations": [],
        "metadata": {
            "model": chosen_model,
            "api_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "total_steps": 0,
            "api_calls": 0,
            "generation_calls": 0,
            "critic_calls": 0,
            "refinement_calls": 0,
            "error": None,
        },
    }


def _record_call(trace: Dict[str, Any], call_result: Dict[str, Any]) -> None:
    metadata = trace["metadata"]
    metadata["api_latency_ms"] += call_result.get("latency_ms", 0.0)
    metadata["prompt_tokens"] += call_result.get("prompt_tokens", 0)
    metadata["completion_tokens"] += call_result.get("completion_tokens", 0)
    metadata["total_tokens"] += call_result.get("total_tokens", 0)
    metadata["cost_usd"] += call_result.get("cost_usd", 0.0)
    metadata["api_calls"] += 1

    phase = call_result.get("phase")
    if phase == "generate":
        metadata["generation_calls"] += 1
    elif phase == "critique":
        metadata["critic_calls"] += 1
    elif phase == "refine":
        metadata["refinement_calls"] += 1


def _analyze_solution(
    trace: Dict[str, Any],
    question: str,
    solution_text: str,
    model: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    verification = verify_solution_steps(solution_text)
    critique_call = _critique_solution_call(question, solution_text, model=model)
    _record_call(trace, critique_call)

    llm_critique = critique_call.get("content") or ""
    verification_summary = _verification_summary(verification)

    if llm_critique and verification_summary:
        full_text = f"{llm_critique}\n\n{verification_summary}"
    elif llm_critique:
        full_text = llm_critique
    else:
        full_text = verification_summary

    has_error = verification["has_error"] or _has_error(llm_critique)
    error_step = verification["first_error_step"] or _extract_error_step(llm_critique)

    critique_payload = {
        "full_text": full_text,
        "llm_text": llm_critique,
        "has_error": has_error,
        "error_step": error_step,
        "tool_verified": verification["checked_steps"] > 0,
        "verification": verification,
    }
    return critique_payload, critique_call


def _finalize_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    selected_solution = None
    for iteration in trace["iterations"]:
        if iteration.get("selected"):
            selected_solution = iteration.get("solution", {}).get("full_text")
            break

    if selected_solution is None and trace["iterations"]:
        selected_solution = trace["iterations"][-1].get("solution", {}).get("full_text")

    extracted = extract_answer(selected_solution)
    trace["final_answer"] = extracted
    trace["metadata"]["total_steps"] = len(split_steps(selected_solution))
    trace["is_correct"] = answers_match(
        extracted,
        trace["question"].get("expected_answer"),
    )
    trace["metadata"]["api_latency_ms"] = round(trace["metadata"]["api_latency_ms"], 2)
    trace["metadata"]["cost_usd"] = round(trace["metadata"]["cost_usd"], 6)
    return trace


def execute_strategy(
    question: str,
    expected_answer: Optional[str],
    strategy_name: str,
    sample_metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    max_iterations: int = settings.DEFAULT_MAX_ITERATIONS,
    num_samples: int = settings.DEFAULT_MAJORITY_SAMPLES,
) -> Dict[str, Any]:
    strategy_name = strategy_name.strip().lower()

    if strategy_name == "baseline":
        return _run_baseline(
            question=question,
            expected_answer=expected_answer,
            sample_metadata=sample_metadata,
            model=model,
        )
    if strategy_name == "iter_refine":
        return _run_iter_refine(
            question=question,
            expected_answer=expected_answer,
            sample_metadata=sample_metadata,
            model=model,
            max_iterations=max_iterations,
        )
    if strategy_name == "filter":
        return _run_filter(
            question=question,
            expected_answer=expected_answer,
            sample_metadata=sample_metadata,
            model=model,
            num_samples=num_samples,
        )
    if strategy_name == "majority":
        return _run_majority(
            question=question,
            expected_answer=expected_answer,
            sample_metadata=sample_metadata,
            model=model,
            num_samples=num_samples,
        )

    raise ValueError(f"Unknown strategy: {strategy_name}")


def _run_baseline(
    question: str,
    expected_answer: Optional[str],
    sample_metadata: Optional[Dict[str, Any]],
    model: Optional[str],
) -> Dict[str, Any]:
    trace = _build_trace(
        question=question,
        expected_answer=expected_answer,
        strategy_name="baseline",
        strategy_params={},
        sample_metadata=sample_metadata,
        model=model,
    )

    generation_call = _generate_solution_call(question, model=model)
    _record_call(trace, generation_call)
    iteration = {
        "round": 0,
        "type": "baseline",
        "llm_calls": [_serialize_call(generation_call)],
        "solution": _build_solution_payload(generation_call.get("content")),
        "selected": True,
    }
    if not generation_call.get("ok"):
        trace["metadata"]["error"] = generation_call.get("error")
    trace["iterations"].append(iteration)
    return _finalize_trace(trace)


def _run_iter_refine(
    question: str,
    expected_answer: Optional[str],
    sample_metadata: Optional[Dict[str, Any]],
    model: Optional[str],
    max_iterations: int,
) -> Dict[str, Any]:
    trace = _build_trace(
        question=question,
        expected_answer=expected_answer,
        strategy_name="iter_refine",
        strategy_params={"max_iterations": max_iterations},
        sample_metadata=sample_metadata,
        model=model,
    )

    generation_call = _generate_solution_call(question, model=model)
    _record_call(trace, generation_call)
    current_iteration = {
        "round": 0,
        "type": "initial_generation",
        "llm_calls": [_serialize_call(generation_call)],
        "solution": _build_solution_payload(generation_call.get("content")),
        "selected": False,
    }
    trace["iterations"].append(current_iteration)

    if not generation_call.get("ok") or not generation_call.get("content"):
        trace["metadata"]["error"] = generation_call.get("error")
        current_iteration["selected"] = True
        return _finalize_trace(trace)

    current_solution = generation_call["content"]

    for round_index in range(max_iterations + 1):
        critique_payload, critique_call = _analyze_solution(trace, question, current_solution, model=model)
        current_iteration["llm_calls"].append(_serialize_call(critique_call))
        current_iteration["critique"] = critique_payload

        if not critique_call.get("ok") and not critique_payload["tool_verified"]:
            trace["metadata"]["error"] = critique_call.get("error")
            current_iteration["selected"] = True
            return _finalize_trace(trace)

        if not critique_payload["has_error"]:
            current_iteration["selected"] = True
            return _finalize_trace(trace)

        if round_index >= max_iterations:
            current_iteration["selected"] = True
            trace["metadata"]["error"] = "Maximum refinement iterations reached."
            return _finalize_trace(trace)

        refinement_call = _refine_solution_call(
            question,
            current_solution,
            critique_payload["full_text"],
            model=model,
        )
        _record_call(trace, refinement_call)

        next_iteration = {
            "round": round_index + 1,
            "type": "refinement",
            "refined_from_round": round_index,
            "llm_calls": [_serialize_call(refinement_call)],
            "solution": _build_solution_payload(refinement_call.get("content")),
            "selected": False,
        }
        trace["iterations"].append(next_iteration)

        if not refinement_call.get("ok") or not refinement_call.get("content"):
            trace["metadata"]["error"] = refinement_call.get("error")
            current_iteration["selected"] = True
            return _finalize_trace(trace)

        current_solution = refinement_call["content"]
        current_iteration = next_iteration

    current_iteration["selected"] = True
    return _finalize_trace(trace)


def _run_filter(
    question: str,
    expected_answer: Optional[str],
    sample_metadata: Optional[Dict[str, Any]],
    model: Optional[str],
    num_samples: int,
) -> Dict[str, Any]:
    trace = _build_trace(
        question=question,
        expected_answer=expected_answer,
        strategy_name="filter",
        strategy_params={"num_samples": num_samples},
        sample_metadata=sample_metadata,
        model=model,
    )

    accepted_index: Optional[int] = None
    fallback_index: Optional[int] = None

    for sample_index in range(num_samples):
        generation_call = _generate_solution_call(question, model=model)
        _record_call(trace, generation_call)

        iteration = {
            "round": sample_index,
            "type": "candidate",
            "candidate_index": sample_index + 1,
            "llm_calls": [_serialize_call(generation_call)],
            "solution": _build_solution_payload(generation_call.get("content")),
            "selected": False,
        }
        trace["iterations"].append(iteration)

        if not generation_call.get("ok") or not generation_call.get("content"):
            if trace["metadata"]["error"] is None:
                trace["metadata"]["error"] = generation_call.get("error")
            continue

        if fallback_index is None:
            fallback_index = sample_index

        critique_payload, critique_call = _analyze_solution(trace, question, generation_call["content"], model=model)
        iteration["llm_calls"].append(_serialize_call(critique_call))
        iteration["critique"] = critique_payload

        if accepted_index is None and not critique_payload["has_error"]:
            accepted_index = sample_index

    selected_index = accepted_index if accepted_index is not None else fallback_index
    if selected_index is not None:
        trace["iterations"][selected_index]["selected"] = True
    return _finalize_trace(trace)


def _run_majority(
    question: str,
    expected_answer: Optional[str],
    sample_metadata: Optional[Dict[str, Any]],
    model: Optional[str],
    num_samples: int,
) -> Dict[str, Any]:
    trace = _build_trace(
        question=question,
        expected_answer=expected_answer,
        strategy_name="majority",
        strategy_params={"num_samples": num_samples},
        sample_metadata=sample_metadata,
        model=model,
    )

    normalized_answers: List[Optional[str]] = []

    for sample_index in range(num_samples):
        generation_call = _generate_solution_call(question, model=model)
        _record_call(trace, generation_call)

        solution_payload = _build_solution_payload(generation_call.get("content"))
        normalized_answers.append(solution_payload.get("normalized_answer"))

        trace["iterations"].append(
            {
                "round": sample_index,
                "type": "majority_sample",
                "candidate_index": sample_index + 1,
                "llm_calls": [_serialize_call(generation_call)],
                "solution": solution_payload,
                "selected": False,
            }
        )

        if not generation_call.get("ok") and trace["metadata"]["error"] is None:
            trace["metadata"]["error"] = generation_call.get("error")

    vote_counter = Counter(answer for answer in normalized_answers if answer is not None)
    winner = vote_counter.most_common(1)[0][0] if vote_counter else None
    trace["strategy"]["vote_tally"] = dict(vote_counter)

    if winner is not None:
        for iteration in trace["iterations"]:
            normalized = iteration["solution"].get("normalized_answer")
            iteration["vote_count"] = vote_counter.get(normalized, 0)
            if not iteration["selected"] and normalized == winner:
                iteration["selected"] = True
                break
    elif trace["iterations"]:
        trace["iterations"][0]["selected"] = True

    return _finalize_trace(trace)


def iterative_refine(
    question: str,
    max_iterations: int = settings.DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> Optional[str]:
    del verbose
    trace = execute_strategy(
        question=question,
        expected_answer=None,
        strategy_name="iter_refine",
        max_iterations=max_iterations,
    )
    for iteration in trace["iterations"]:
        if iteration.get("selected"):
            return iteration["solution"]["full_text"]
    return None


def critic_as_filter(
    question: str,
    num_samples: int = settings.DEFAULT_FILTER_SAMPLES,
    verbose: bool = False,
) -> Optional[str]:
    del verbose
    trace = execute_strategy(
        question=question,
        expected_answer=None,
        strategy_name="filter",
        num_samples=num_samples,
    )
    for iteration in trace["iterations"]:
        if iteration.get("selected"):
            return iteration["solution"]["full_text"]
    return None


def majority_vote(
    question: str,
    num_samples: int = settings.DEFAULT_MAJORITY_SAMPLES,
    verbose: bool = False,
) -> Optional[str]:
    del verbose
    trace = execute_strategy(
        question=question,
        expected_answer=None,
        strategy_name="majority",
        num_samples=num_samples,
    )
    for iteration in trace["iterations"]:
        if iteration.get("selected"):
            return iteration["solution"]["full_text"]
    return None


def run_strategy(question: str, strategy_name: str, **kwargs: Any) -> Optional[str]:
    if strategy_name == "iter_refine":
        return iterative_refine(question, max_iterations=kwargs.get("max_iterations", settings.DEFAULT_MAX_ITERATIONS))
    if strategy_name == "filter":
        return critic_as_filter(question, num_samples=kwargs.get("num_samples", settings.DEFAULT_FILTER_SAMPLES))
    if strategy_name == "majority":
        return majority_vote(question, num_samples=kwargs.get("num_samples", settings.DEFAULT_MAJORITY_SAMPLES))
    if strategy_name == "baseline":
        return generate_solution(question)
    raise ValueError(f"Unknown strategy: {strategy_name}")


STRATEGIES = {
    "baseline": generate_solution,
    "iter_refine": iterative_refine,
    "filter": critic_as_filter,
    "majority": majority_vote,
}
