"""
Rubrics-based evaluation of trained-model rollout trajectories.

Adapted from rubrics/judge.py for the rollout data format where each entry has:
  - prompt: [system, user] messages
  - rejected: the model's rollout trajectory (candidate to evaluate)
  - chosen: the reference/golden trajectory

Full candidate = prompt + rejected, full reference = prompt + chosen.
Reference answer is derived from the last assistant message in the chosen trajectory.

Usage:
    # Evaluate a single checkpoint (all metrics)
    python rubrics/judge_rollouts.py evaluate -d "data/trajectory/rollouts/test_eval_0806/dpo8b"

    # Evaluate all checkpoints under a rollout directory
    python rubrics/judge_rollouts.py evaluate-all -d "data/trajectory/rollouts/test_eval_0806"

    # Run specific metrics on a small subset
    python rubrics/judge_rollouts.py evaluate -d "data/trajectory/rollouts/test_eval_0806/dpo8b" --metrics pass_rate,tool_calling_f1 -n 5

    # Aggregate results across checkpoints
    python rubrics/judge_rollouts.py aggregate

    # Use DeepSeek-V4-Pro as the judge model (requires DEEPSEEK_API_KEY)
    python rubrics/judge_rollouts.py evaluate -d "data/trajectory/rollouts/test_eval_0806/dpo8b" --judge-model deepseek-v4-pro
"""

import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import typer
from dotenv import load_dotenv

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gen_eval import (
    _get_llm_rate_limit_primitives,
    _is_llm_retryable,
    _llm_rate_limit_guard,
    configure_llm_guard,
)

load_dotenv()

if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup  # noqa: F821
else:
    _BaseExceptionGroup = Exception  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_OUTPUT_CHARS = 2000

ALL_METRICS = [
    "tool_calling_f1",
    "step_efficiency",
    "redundancy_score",
    "pass_rate",
    "task_relevance",
    "logical_progression",
    "information_utilization",
    "progress_score",
    "final_answer_quality",
]

ALGORITHMIC_METRICS = {"tool_calling_f1", "step_efficiency", "redundancy_score"}
LLM_JUDGED_METRICS = set(ALL_METRICS) - ALGORITHMIC_METRICS

# ---------------------------------------------------------------------------
# LLM-judge system prompts — one per metric
# ---------------------------------------------------------------------------

PASS_RATE_SYSTEM_PROMPT = """\
You are an expert financial analyst evaluating the correctness of an AI assistant's final answer.

You will be given a financial question, a reference (ground truth) answer, and the candidate's final answer.

Evaluate whether the candidate's final answer is factually correct compared to the reference answer. \
Focus on key facts, figures, and data points. Minor formatting differences are acceptable; \
focus on substantive accuracy.

Scoring rubric (1-5):
1 - Completely wrong or no answer provided
2 - Major errors; only tangentially related to the correct answer
3 - Partially correct; some key facts match but significant information is wrong or missing
4 - Mostly correct; minor numerical or factual errors only
5 - Fully correct; matches reference answer in all key facts and figures

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

TASK_RELEVANCE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI assistant tool-calling trajectories in the financial domain.

You will evaluate whether each tool call in the candidate trajectory is relevant to answering \
the original query. A relevant tool call is one that could logically contribute information \
needed to answer the question.

Scoring rubric (1-5):
1 - Tool calls entirely unrelated to the query
2 - Mostly irrelevant tools called; fundamental misunderstanding of what data is needed
3 - Some relevant tools, but also many irrelevant or misguided calls
4 - Most tools are relevant; only minor irrelevant calls
5 - All tool calls are directly relevant to answering the query

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

LOGICAL_PROGRESSION_SYSTEM_PROMPT = """\
You are an expert evaluator of AI assistant tool-calling trajectories in the financial domain.

You will evaluate whether the sequence of tool calls follows a coherent, logical strategy. \
Each step should build on information from previous steps, and the overall sequence should \
represent a reasonable approach to answering the query.

You will see both a golden (reference) trajectory showing one effective approach, and the \
candidate trajectory to evaluate. The candidate does NOT need to follow the same path as \
the golden trajectory — it just needs to follow a logical strategy.

Scoring rubric (1-5):
1 - Chaotic; no coherent strategy; random or contradictory tool calls
2 - Weak logical flow; major gaps or illogical sequencing
3 - Some logical structure but with unnecessary detours or missed dependencies
4 - Clear logical flow with only minor inefficiencies
5 - Optimal logical progression; each step naturally follows from the previous

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

INFORMATION_UTILIZATION_SYSTEM_PROMPT = """\
You are an expert evaluator of AI assistant tool-calling trajectories in the financial domain.

You will evaluate how effectively the candidate uses the data returned by tool calls. \
Check whether key data from tool outputs is reflected in subsequent reasoning and the \
final answer, and whether any important information was ignored or misinterpreted.

Scoring rubric (1-5):
1 - Ignores tool outputs entirely; final answer does not reference retrieved data
2 - Uses tool outputs poorly; misinterprets data or draws wrong conclusions from it
3 - Uses some information but misses key data points available in tool outputs
4 - Good use of information; minor oversights in utilizing available data
5 - Excellent; fully leverages all relevant tool outputs in reasoning and final answer

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

PROGRESS_SCORE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI assistant tool-calling trajectories in the financial domain.

You will evaluate whether each turn in the candidate trajectory makes meaningful progress \
toward answering the query. Look for signs of stalling, looping, repeated failed attempts, \
or wasted turns that do not advance the solution.

Scoring rubric (1-5):
1 - No progress made; stuck in loops or entirely off-track
2 - Minimal progress; many wasted turns with little forward movement
3 - Moderate progress; some turns are productive but others are wasted
4 - Good progress; nearly every turn advances toward the answer
5 - Every turn makes clear, meaningful progress toward the final answer

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

FINAL_ANSWER_QUALITY_SYSTEM_PROMPT = """\
You are an expert financial analyst evaluating the overall quality of an AI assistant's \
final answer to a financial question.

Evaluate the answer considering accuracy (vs reference answer), completeness (addresses all \
parts of the question), clarity (well-organized and easy to understand), and presentation \
(appropriate use of formatting, tables, numbers).

You will also see the golden (reference) trajectory for context on what a high-quality \
response path looks like.

Scoring rubric (1-5):
1 - No answer or completely unusable response
2 - Poor quality; major factual errors, missing most parts of the question, or incoherent
3 - Acceptable; addresses the question but incomplete, unclear, or has moderate errors
4 - Good quality; well-organized, mostly accurate, minor issues only
5 - Excellent; comprehensive, accurate, clearly presented, and addresses all parts

You MUST respond with valid JSON only, no other text. Use this exact format:
{"score": 4, "justification": "2-3 sentence explanation"}

The "score" field must be an integer from 1 to 5."""

METRIC_SYSTEM_PROMPTS = {
    "pass_rate": PASS_RATE_SYSTEM_PROMPT,
    "task_relevance": TASK_RELEVANCE_SYSTEM_PROMPT,
    "logical_progression": LOGICAL_PROGRESSION_SYSTEM_PROMPT,
    "information_utilization": INFORMATION_UTILIZATION_SYSTEM_PROMPT,
    "progress_score": PROGRESS_SCORE_SYSTEM_PROMPT,
    "final_answer_quality": FINAL_ANSWER_QUALITY_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# LLM-judge user prompt templates — one per metric
# ---------------------------------------------------------------------------

PASS_RATE_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Reference Answer
{reference_answer}

## Candidate Final Answer
{candidate_final_answer}

Evaluate the correctness of the candidate's final answer compared to the reference. Respond with JSON only."""

TASK_RELEVANCE_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Candidate Trajectory
{candidate_trajectory}

Evaluate whether the tool calls in the candidate trajectory are relevant to answering the query. Respond with JSON only."""

LOGICAL_PROGRESSION_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Golden Label Trajectory (Reference)
{golden_trajectory}

## Candidate Trajectory (To Evaluate)
{candidate_trajectory}

Evaluate the logical progression of the candidate trajectory. Respond with JSON only."""

INFORMATION_UTILIZATION_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Candidate Trajectory
{candidate_trajectory}

## Candidate Final Answer
{candidate_final_answer}

Evaluate how effectively the candidate uses data from tool outputs. Respond with JSON only."""

PROGRESS_SCORE_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Candidate Trajectory
{candidate_trajectory}

Evaluate whether each turn makes meaningful progress toward the answer. Respond with JSON only."""

FINAL_ANSWER_QUALITY_USER_TEMPLATE = """\
## Financial Question
{source_query}

## Reference Answer
{reference_answer}

## Golden Label Trajectory (Reference)
{golden_trajectory}

## Candidate Final Answer
{candidate_final_answer}

Evaluate the overall quality of the candidate's final answer. Respond with JSON only."""

METRIC_USER_TEMPLATES = {
    "pass_rate": PASS_RATE_USER_TEMPLATE,
    "task_relevance": TASK_RELEVANCE_USER_TEMPLATE,
    "logical_progression": LOGICAL_PROGRESSION_USER_TEMPLATE,
    "information_utilization": INFORMATION_UTILIZATION_USER_TEMPLATE,
    "progress_score": PROGRESS_SCORE_USER_TEMPLATE,
    "final_answer_quality": FINAL_ANSWER_QUALITY_USER_TEMPLATE,
}

# ---------------------------------------------------------------------------
# Helpers — trajectory formatting & parsing
# ---------------------------------------------------------------------------


def _format_trajectory(messages: list[dict]) -> str:
    """Format a message history into a readable trajectory string."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""

        if role == "system":
            continue

        if role == "user":
            parts.append(f"[User]\n{content}")

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if content:
                parts.append(f"[Assistant]\n{content}")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    parts.append(f"[Tool Call] {name}({args})")

        elif role == "tool":
            output = content
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + "... [truncated]"
            parts.append(f"[Tool Result]\n{output}")

    return "\n\n".join(parts)


def _extract_tool_call_names(messages: list[dict]) -> list[str]:
    """Extract all tool call function names from messages in order."""
    names = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                name = func.get("name", "")
                if name:
                    names.append(name)
    return names


def _extract_tool_calls_with_args(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract all (name, arguments_str) pairs from messages."""
    calls = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                if name:
                    calls.append((name, args))
    return calls


def _count_turns(messages: list[dict]) -> int:
    """Count assistant turns in a message list."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def _extract_final_answer(messages: list[dict]) -> str:
    """Extract the final answer from the last assistant message with content."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return ""


def _parse_eval_response(raw_text: str) -> tuple[dict | None, str | None]:
    """Parse the evaluator's JSON response. Returns (parsed, error)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError as e:
                return None, f"JSON parse failed: {e}"
        else:
            return None, "No JSON object found in response"

    if "score" not in result:
        return None, "Missing 'score' key"
    try:
        score = int(result["score"])
    except (ValueError, TypeError):
        return None, f"Invalid score: {result['score']}"
    if score < 1 or score > 5:
        return None, f"Score out of range: {score}"
    result["score"] = score
    return result, None


# ---------------------------------------------------------------------------
# Rollout data conversion
# ---------------------------------------------------------------------------


def convert_rollout_entry(raw: dict) -> tuple[dict, dict]:
    """Convert a rollout entry into (candidate, golden) dicts compatible with
    the rubrics evaluation functions.

    The rollout format:
      - prompt: [system_msg, user_msg]
      - rejected: model's rollout messages (candidate)
      - chosen: reference/golden messages

    Full candidate = prompt + rejected
    Full golden    = prompt + chosen
    """
    prompt_msgs = raw.get("prompt", [])
    rejected_msgs = raw.get("rejected", [])
    chosen_msgs = raw.get("chosen", [])

    candidate_messages = prompt_msgs + rejected_msgs
    golden_messages = prompt_msgs + chosen_msgs

    # Extract source query from user message in prompt
    source_query = ""
    for m in prompt_msgs:
        if m.get("role") == "user":
            source_query = m.get("content", "")
            break

    # Extract final answers
    candidate_final_answer = _extract_final_answer(rejected_msgs)
    golden_final_answer = _extract_final_answer(chosen_msgs)

    # Extract endpoints from messages
    candidate_endpoints = list(set(_extract_tool_call_names(rejected_msgs)))
    golden_endpoints_raw = raw.get("_endpoints_called", "[]")
    if isinstance(golden_endpoints_raw, list):
        golden_endpoints = golden_endpoints_raw
    else:
        try:
            golden_endpoints = json.loads(golden_endpoints_raw)
        except (json.JSONDecodeError, TypeError):
            golden_endpoints = []

    # Build metadata
    metadata_fields = {
        "task_type": raw.get("_task_type", ""),
        "task_type_bucket": raw.get("_task_type_bucket", ""),
        "difficulty_tier": raw.get("_difficulty_tier", ""),
        "difficulty_score": raw.get("_difficulty_score", 0.0),
    }

    candidate = {
        "id": raw["_id"],
        "model": "qwen3.5-9b-rollout",
        "source_query": source_query,
        "messages": candidate_messages,
        "final_answer": candidate_final_answer,
        "endpoints_called": candidate_endpoints,
        "num_turns": _count_turns(candidate_messages),
        "num_tool_calls": sum(
            len(m.get("tool_calls") or [])
            for m in candidate_messages
            if m.get("role") == "assistant"
        ),
        "has_final_answer": bool(candidate_final_answer),
        **metadata_fields,
    }

    golden = {
        "id": raw["_id"],
        "model": "chosen_reference",
        "source_query": source_query,
        "reference_answer": golden_final_answer,
        "messages": golden_messages,
        "final_answer": golden_final_answer,
        "endpoints_called": golden_endpoints,
        "num_turns": _count_turns(golden_messages),
        **metadata_fields,
    }

    return candidate, golden


def _build_metadata(entry: dict) -> dict:
    """Extract metadata fields for the output record."""
    return {
        "difficulty_tier": entry.get("difficulty_tier", ""),
        "task_type_bucket": entry.get("task_type_bucket", ""),
        "task_type": entry.get("task_type", ""),
        "difficulty_score": entry.get("difficulty_score", 0.0),
    }


# ---------------------------------------------------------------------------
# Algorithmic metrics
# ---------------------------------------------------------------------------


def compute_tool_calling_f1(candidate: dict, golden: dict) -> dict:
    """Compute set-based and bag-based F1 on tool calls."""
    cand_set = set(candidate.get("endpoints_called", []) or [])
    gold_set = set(golden.get("endpoints_called", []) or [])

    if not cand_set and not gold_set:
        set_f1 = 1.0
        set_precision = 1.0
        set_recall = 1.0
    elif not cand_set or not gold_set:
        set_f1 = 0.0
        set_precision = 0.0
        set_recall = 0.0
    else:
        intersection = cand_set & gold_set
        set_precision = len(intersection) / len(cand_set)
        set_recall = len(intersection) / len(gold_set)
        set_f1 = (
            2 * set_precision * set_recall / (set_precision + set_recall)
            if (set_precision + set_recall) > 0
            else 0.0
        )

    # Bag-based F1
    cand_calls = _extract_tool_call_names(candidate.get("messages", []))
    gold_calls = _extract_tool_call_names(golden.get("messages", []))
    cand_bag = Counter(cand_calls)
    gold_bag = Counter(gold_calls)

    if not cand_bag and not gold_bag:
        bag_f1 = 1.0
    elif not cand_bag or not gold_bag:
        bag_f1 = 0.0
    else:
        intersection_count = sum((cand_bag & gold_bag).values())
        bag_precision = intersection_count / sum(cand_bag.values())
        bag_recall = intersection_count / sum(gold_bag.values())
        bag_f1 = (
            2 * bag_precision * bag_recall / (bag_precision + bag_recall)
            if (bag_precision + bag_recall) > 0
            else 0.0
        )

    return {
        "id": candidate["id"],
        "model": candidate.get("model", ""),
        "metric": "tool_calling_f1",
        "score": round(set_f1, 4),
        "max_score": 1.0,
        "details": {
            "set_f1": round(set_f1, 4),
            "set_precision": round(set_precision, 4),
            "set_recall": round(set_recall, 4),
            "bag_f1": round(bag_f1, 4),
            "candidate_endpoints": sorted(cand_set),
            "golden_endpoints": sorted(gold_set),
        },
        "metadata": _build_metadata(candidate),
    }


def compute_step_efficiency(candidate: dict, golden: dict) -> dict:
    """Compute step efficiency as ratio of golden turns to candidate turns."""
    cand_turns = candidate.get("num_turns") or _count_turns(
        candidate.get("messages", [])
    )
    gold_turns = golden.get("num_turns") or _count_turns(golden.get("messages", []))

    if cand_turns == 0:
        score = 0.0
    elif gold_turns == 0:
        score = 1.0 if cand_turns == 0 else 0.0
    else:
        score = min(gold_turns / cand_turns, 1.0)

    return {
        "id": candidate["id"],
        "model": candidate.get("model", ""),
        "metric": "step_efficiency",
        "score": round(score, 4),
        "max_score": 1.0,
        "details": {
            "candidate_turns": cand_turns,
            "golden_turns": gold_turns,
        },
        "metadata": _build_metadata(candidate),
    }


def compute_redundancy_score(candidate: dict) -> dict:
    """Compute redundancy score (1.0 = no redundancy)."""
    calls = _extract_tool_calls_with_args(candidate.get("messages", []))

    if not calls:
        return {
            "id": candidate["id"],
            "model": candidate.get("model", ""),
            "metric": "redundancy_score",
            "score": 1.0,
            "max_score": 1.0,
            "details": {"total_calls": 0, "redundant_calls": 0},
            "metadata": _build_metadata(candidate),
        }

    seen: set[tuple[str, str]] = set()
    redundant = 0
    for name, args_str in calls:
        key = (name, args_str)
        if key in seen:
            redundant += 1
        seen.add(key)

    score = 1.0 - (redundant / len(calls))
    return {
        "id": candidate["id"],
        "model": candidate.get("model", ""),
        "metric": "redundancy_score",
        "score": round(score, 4),
        "max_score": 1.0,
        "details": {
            "total_calls": len(calls),
            "redundant_calls": redundant,
            "unique_calls": len(seen),
        },
        "metadata": _build_metadata(candidate),
    }


# ---------------------------------------------------------------------------
# Judge model helpers (Claude detection, custom_id codec, record builders)
# ---------------------------------------------------------------------------


def _is_claude_model(model: str) -> bool:
    """True if the judge model should route through the Anthropic Batches path."""
    m = model.lower()
    return m.startswith("claude") or m.startswith("anthropic/")


def _bare_anthropic_model(model: str) -> str:
    """Strip an optional 'anthropic/' prefix to get the bare Anthropic model id."""
    return model[len("anthropic/"):] if model.lower().startswith("anthropic/") else model


_CUSTOM_ID_SEP = "__"


def _encode_custom_id(metric_name: str, qid) -> str:
    """Encode (metric, id) into an Anthropic batch custom_id (e.g. 'pass_rate__42')."""
    return f"{metric_name}{_CUSTOM_ID_SEP}{qid}"


def _decode_custom_id(custom_id: str) -> tuple[str, str]:
    """Decode a custom_id back into (metric_name, id_str)."""
    metric_name, _, qid = custom_id.rpartition(_CUSTOM_ID_SEP)
    return metric_name, qid


def _custom_id_ok(custom_id: str) -> bool:
    """Anthropic requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$."""
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", custom_id))


def _make_success_record(candidate: dict, metric_name: str, parsed: dict) -> dict:
    """Build a successful metric result record (shared by per-request & batch paths)."""
    return {
        "id": candidate["id"],
        "model": candidate.get("model", ""),
        "metric": metric_name,
        "score": parsed["score"],
        "max_score": 5,
        "justification": parsed.get("justification", ""),
        "metadata": _build_metadata(candidate),
    }


def _make_error_record(
    candidate: dict,
    metric_name: str,
    justification: str,
    raw_response: str | None = None,
) -> dict:
    """Build an errored metric result record (score=None)."""
    record = {
        "id": candidate["id"],
        "model": candidate.get("model", ""),
        "metric": metric_name,
        "score": None,
        "max_score": 5,
        "justification": justification,
        "metadata": _build_metadata(candidate),
    }
    if raw_response is not None:
        record["raw_response"] = raw_response
    return record


def _build_llm_eval_request(
    metric_name: str, candidate: dict, golden: dict
) -> tuple[str, list[dict], dict | None]:
    """Build the judge prompt for one (metric, candidate) pair.

    Returns (system_prompt, messages, early_exit_record).
      - messages is [{"role": "user", "content": ...}] with NO system-role message;
        the system prompt is returned separately so the Anthropic batch path can pass
        it as a top-level param while the litellm path prepends it as a message.
      - early_exit_record is a fully-formed score=1 result dict when the candidate has
        no final answer and the metric is pass_rate/final_answer_quality, else None.
    """
    system_prompt = METRIC_SYSTEM_PROMPTS[metric_name]
    user_template = METRIC_USER_TEMPLATES[metric_name]

    candidate_final_answer = candidate.get("final_answer") or ""
    if not candidate_final_answer and metric_name in (
        "pass_rate",
        "final_answer_quality",
    ):
        early = {
            "id": candidate["id"],
            "model": candidate.get("model", ""),
            "metric": metric_name,
            "score": 1,
            "max_score": 5,
            "justification": "No final answer provided by the candidate (trajectory ended with a tool call).",
            "metadata": _build_metadata(candidate),
        }
        return system_prompt, [], early

    template_vars = {
        "source_query": candidate.get("source_query", ""),
        "reference_answer": golden.get("reference_answer", ""),
        "candidate_final_answer": candidate_final_answer or "[No answer provided]",
        "candidate_trajectory": _format_trajectory(candidate.get("messages", [])),
        "golden_trajectory": _format_trajectory(golden.get("messages", [])),
    }
    user_content = user_template.format(**template_vars)
    messages = [{"role": "user", "content": user_content}]
    return system_prompt, messages, None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_llm_max_retries = 3

PARSE_RETRY_CORRECTION = (
    "Your response was not valid JSON. Please respond with ONLY a JSON object: "
    '{"score": <integer 1-5>, "justification": "..."}'
)


async def _eval_llm_call(
    model: str, system_prompt: str, messages: list[dict]
) -> str:
    """Call the evaluator LLM and return the raw response text.

    `messages` contains only user/assistant turns; `system_prompt` is prepended
    as a system-role message for the litellm/OpenAI-compatible API.
    """
    import os

    import litellm

    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    call_kwargs: dict = {
        "model": model,
        "messages": full_messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    # DeepSeek (deepseek-v4-pro / deepseek-v4-flash / legacy deepseek-chat,reasoner)
    # is not yet in LiteLLM's built-in model map; route via OpenAI-compatible passthrough.
    if model.lower().startswith("deepseek"):
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set; required for deepseek-* judge models."
            )
        call_kwargs["model"] = f"openai/{model}"
        call_kwargs["api_base"] = "https://api.deepseek.com"
        call_kwargs["api_key"] = api_key

    retries = 0
    while True:
        event, _ = _get_llm_rate_limit_primitives()
        await event.wait()

        try:
            response = await litellm.acompletion(**call_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            if _is_llm_retryable(exc):
                retries += 1
                if retries > _llm_max_retries:
                    raise
                await _llm_rate_limit_guard(exc)
                continue
            raise


# ---------------------------------------------------------------------------
# LLM-judged metric evaluation
# ---------------------------------------------------------------------------


async def _evaluate_llm_metric(
    metric_name: str,
    candidate: dict,
    golden: dict,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generic LLM-judged metric evaluation."""
    async with semaphore:
        system_prompt, messages, early = _build_llm_eval_request(
            metric_name, candidate, golden
        )
        if early is not None:
            return early

        raw_response = await _eval_llm_call(judge_model, system_prompt, messages)
        parsed, parse_error = _parse_eval_response(raw_response)

        # Retry once on parse failure
        if parse_error:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": PARSE_RETRY_CORRECTION,
                }
            )
            raw_response = await _eval_llm_call(judge_model, system_prompt, messages)
            parsed, parse_error = _parse_eval_response(raw_response)

        if parsed and not parse_error:
            return _make_success_record(candidate, metric_name, parsed)
        else:
            return _make_error_record(
                candidate,
                metric_name,
                f"PARSE ERROR: {parse_error}",
                raw_response=raw_response,
            )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


async def evaluate_metric_batch(
    metric_name: str,
    candidates: list[dict],
    golden_by_id: dict[int, dict],
    judge_model: str,
    output_path: Path,
    concurrency: int,
) -> list[dict]:
    """Evaluate one metric across all candidates with resume support."""
    from tqdm import tqdm

    # Load existing results for resume
    all_records: list[dict] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(
                f"  Resumed: {len(all_records)} existing results from {output_path.name}"
            )
        except (json.JSONDecodeError, Exception) as exc:
            print(
                f"  Warning: could not load existing results ({exc}), starting fresh."
            )
            all_records = []

    processed_ids = {r["id"] for r in all_records}
    remaining = [c for c in candidates if c["id"] not in processed_ids]

    if not remaining:
        print(f"  All {len(all_records)} queries already evaluated for {metric_name}.")
        return all_records

    print(f"  {len(remaining)} queries remaining for {metric_name}.")

    is_algorithmic = metric_name in ALGORITHMIC_METRICS
    semaphore = asyncio.Semaphore(concurrency)
    save_lock = asyncio.Lock()

    async def process_one(candidate: dict) -> None:
        golden = golden_by_id.get(candidate["id"])
        if golden is None:
            return

        try:
            if is_algorithmic:
                if metric_name == "tool_calling_f1":
                    result = compute_tool_calling_f1(candidate, golden)
                elif metric_name == "step_efficiency":
                    result = compute_step_efficiency(candidate, golden)
                elif metric_name == "redundancy_score":
                    result = compute_redundancy_score(candidate)
                else:
                    return
            else:
                result = await _evaluate_llm_metric(
                    metric_name, candidate, golden, judge_model, semaphore
                )

            async with save_lock:
                all_records.append(result)
                sorted_records = sorted(all_records, key=lambda r: r["id"])
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_records, f, indent=2, ensure_ascii=False)

        except (asyncio.CancelledError, _BaseExceptionGroup) as exc:
            print(
                f"    [WARN] query {candidate['id']} cancelled: {type(exc).__name__}"
            )
        except Exception as exc:
            print(
                f"    [ERROR] query {candidate['id']}: {type(exc).__name__}: {exc}"
            )

    with tqdm(total=len(remaining), desc=f"  {metric_name}") as pbar:

        async def process_and_update(candidate: dict) -> None:
            await process_one(candidate)
            pbar.update(1)
            pbar.set_postfix(saved=len(all_records))

        tasks = [process_and_update(c) for c in remaining]
        await asyncio.gather(*tasks, return_exceptions=True)

    return all_records


# ---------------------------------------------------------------------------
# Anthropic Message Batches path (Claude judge models)
# ---------------------------------------------------------------------------

BATCH_MAX_TOKENS = 1024


def _load_existing_records(output_path: Path) -> tuple[list[dict], set]:
    """Load existing metric results for resume; return (records, processed_ids)."""
    records: list[dict] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not load {output_path.name} ({exc}); starting fresh.")
            records = []
    return records, {r["id"] for r in records}


def _write_metric_records(output_path: Path, records: list[dict]) -> None:
    """Write metric results sorted by id (same schema as the per-request path)."""
    sorted_records = sorted(records, key=lambda r: r["id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, indent=2, ensure_ascii=False)


async def _submit_and_collect_batch(
    client,
    requests: list[dict],
    judge_model: str,
    poll_interval: int,
    poll_max_seconds: int,
    label: str,
) -> dict[str, object]:
    """Submit one Anthropic message batch, poll to completion, return
    {custom_id: result_object}. Raises on timeout."""
    import asyncio
    import time

    # Submitting a large request list in a single POST can intermittently fail
    # with APIConnectionError/ReadError. Retry the create with backoff.
    create_attempts = 6
    backoff = 10
    batch = None
    for attempt in range(1, create_attempts + 1):
        try:
            batch = await client.messages.batches.create(requests=requests)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= create_attempts:
                raise
            print(
                f"  [{label}] batch create failed (attempt {attempt}/{create_attempts}): "
                f"{type(exc).__name__}: {exc} — retrying in {backoff}s"
            )
            await asyncio.sleep(backoff)
            backoff = min(int(backoff * 1.5), 120)
    print(f"  [{label}] submitted batch {batch.id} with {len(requests)} requests")

    start = time.monotonic()
    interval = poll_interval
    while True:
        info = await client.messages.batches.retrieve(batch.id)
        if info.processing_status == "ended":
            break
        elapsed = time.monotonic() - start
        if elapsed > poll_max_seconds:
            raise TimeoutError(
                f"Batch {batch.id} did not finish within {poll_max_seconds}s "
                f"(status={info.processing_status})"
            )
        counts = info.request_counts
        print(
            f"  [{label}] {info.processing_status}: "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} (elapsed {int(elapsed)}s)"
        )
        await asyncio.sleep(interval)
        interval = min(int(interval * 1.5), 300)

    results: dict[str, object] = {}
    result_stream = await client.messages.batches.results(batch.id)
    async for entry in result_stream:
        results[entry.custom_id] = entry.result
    print(f"  [{label}] batch {batch.id} ended: {len(results)} results retrieved")
    return results


def _extract_text(message) -> str:
    """Extract the first text block from an Anthropic message."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return block.text or ""
    return ""


async def run_llm_metrics_via_batch(
    metric_names: list[str],
    candidates: list[dict],
    golden_by_id: dict,
    judge_model: str,
    output_paths: dict[str, Path],
    poll_interval: int = 30,
    poll_max_seconds: int = 86400,
) -> None:
    """Evaluate all LLM-judged `metric_names` for all candidates in a single
    Anthropic message batch. Preserves the per-request output schema, resume
    support, the no-final-answer early-exit, and the parse-failure retry round.
    """
    from anthropic import AsyncAnthropic

    bare_model = _bare_anthropic_model(judge_model)
    client = AsyncAnthropic()

    # Seed per-metric records from disk (resume) and gather requests.
    records_by_metric: dict[str, list[dict]] = {}
    processed_by_metric: dict[str, set] = {}
    for metric in metric_names:
        recs, ids = _load_existing_records(output_paths[metric])
        records_by_metric[metric] = recs
        processed_by_metric[metric] = ids
        if recs:
            print(f"  Resumed {metric}: {len(recs)} existing results.")

    requests: list[dict] = []
    # custom_id -> (metric, candidate, system_prompt, messages)
    pending: dict[str, tuple] = {}
    early_metrics: set[str] = set()
    skipped_bad_id = 0

    for candidate in candidates:
        golden = golden_by_id.get(candidate["id"])
        if golden is None:
            continue
        for metric in metric_names:
            if candidate["id"] in processed_by_metric[metric]:
                continue
            system_prompt, messages, early = _build_llm_eval_request(
                metric, candidate, golden
            )
            if early is not None:
                records_by_metric[metric].append(early)
                early_metrics.add(metric)
                continue

            custom_id = _encode_custom_id(metric, candidate["id"])
            if not _custom_id_ok(custom_id):
                skipped_bad_id += 1
                continue
            pending[custom_id] = (metric, candidate, system_prompt, messages)
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": bare_model,
                        "max_tokens": BATCH_MAX_TOKENS,
                        "system": system_prompt,
                        "messages": messages,
                    },
                }
            )

    # Persist early-exit records up front (crash-safe).
    for metric in early_metrics:
        _write_metric_records(output_paths[metric], records_by_metric[metric])

    if skipped_bad_id:
        print(f"  [WARN] {skipped_bad_id} requests skipped (invalid custom_id).")

    if not requests:
        print("  Nothing to submit (all resumed or early-exited).")
        for metric in metric_names:
            _write_metric_records(output_paths[metric], records_by_metric[metric])
        return

    # ---- Round 1 ----
    results = await _submit_and_collect_batch(
        client, requests, judge_model, poll_interval, poll_max_seconds, label="round-1"
    )

    # custom_id -> (metric, candidate, system_prompt, messages, raw_text)
    parse_failures: dict[str, tuple] = {}
    for custom_id, (metric, candidate, system_prompt, messages) in pending.items():
        result = results.get(custom_id)
        if result is None:
            records_by_metric[metric].append(
                _make_error_record(candidate, metric, "REQUEST ERROR: missing result")
            )
            continue
        rtype = getattr(result, "type", None)
        if rtype == "succeeded":
            raw = _extract_text(result.message)
            parsed, parse_error = _parse_eval_response(raw)
            if parsed and not parse_error:
                records_by_metric[metric].append(
                    _make_success_record(candidate, metric, parsed)
                )
            else:
                parse_failures[custom_id] = (
                    metric,
                    candidate,
                    system_prompt,
                    messages,
                    raw,
                )
        else:
            records_by_metric[metric].append(
                _make_error_record(candidate, metric, f"REQUEST ERROR: {rtype}")
            )

    # Persist round-1 results before the retry round.
    for metric in metric_names:
        _write_metric_records(output_paths[metric], records_by_metric[metric])

    # ---- Round 2 (parse-failure retry) ----
    if parse_failures:
        print(f"  {len(parse_failures)} parse failures; submitting retry batch.")
        retry_requests: list[dict] = []
        retry_pending: dict[str, tuple] = {}
        for custom_id, (metric, candidate, system_prompt, messages, raw) in (
            parse_failures.items()
        ):
            retry_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": PARSE_RETRY_CORRECTION},
            ]
            retry_pending[custom_id] = (metric, candidate, raw)
            retry_requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": bare_model,
                        "max_tokens": BATCH_MAX_TOKENS,
                        "system": system_prompt,
                        "messages": retry_messages,
                    },
                }
            )

        retry_results = await _submit_and_collect_batch(
            client, retry_requests, judge_model, poll_interval, poll_max_seconds,
            label="round-2",
        )

        for custom_id, (metric, candidate, prev_raw) in retry_pending.items():
            result = retry_results.get(custom_id)
            raw = prev_raw
            parsed = parse_error = None
            if result is not None and getattr(result, "type", None) == "succeeded":
                raw = _extract_text(result.message)
                parsed, parse_error = _parse_eval_response(raw)
            if parsed and not parse_error:
                records_by_metric[metric].append(
                    _make_success_record(candidate, metric, parsed)
                )
            else:
                err = parse_error or "no valid response after retry"
                records_by_metric[metric].append(
                    _make_error_record(
                        candidate, metric, f"PARSE ERROR: {err}", raw_response=raw
                    )
                )

        for metric in metric_names:
            _write_metric_records(output_paths[metric], records_by_metric[metric])

    for metric in metric_names:
        print(f"  {metric}: {len(records_by_metric[metric])} results written.")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_aggregate(model_dir: Path) -> list[dict]:
    """Build per-query aggregate from individual metric JSON files."""
    scores_by_id: dict[int, dict] = {}

    for metric in ALL_METRICS:
        metric_file = model_dir / f"{metric}.json"
        if not metric_file.exists():
            continue
        with open(metric_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            qid = r["id"]
            if qid not in scores_by_id:
                scores_by_id[qid] = {
                    "id": qid,
                    "model": r.get("model", ""),
                    "difficulty_tier": r.get("metadata", {}).get(
                        "difficulty_tier", ""
                    ),
                    "task_type_bucket": r.get("metadata", {}).get(
                        "task_type_bucket", ""
                    ),
                    "scores": {},
                }
            score = r.get("score")
            if score is not None:
                scores_by_id[qid]["scores"][metric] = score

    return sorted(scores_by_id.values(), key=lambda x: x["id"])


def build_summary_csv(results_dir: Path, output_path: Path) -> None:
    """Build a cross-checkpoint summary CSV."""
    rows: list[dict] = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "__pycache__":
            continue

        checkpoint_name = model_dir.name
        aggregate = build_aggregate(model_dir)
        if not aggregate:
            continue

        row = {"checkpoint": checkpoint_name, "num_queries": len(aggregate)}

        # Count how many had no final answer
        no_answer = sum(
            1
            for a in aggregate
            if a.get("scores", {}).get("pass_rate") == 1
            and a.get("scores", {}).get("final_answer_quality") == 1
        )
        row["no_final_answer"] = no_answer

        for metric in ALL_METRICS:
            values = [
                a["scores"][metric]
                for a in aggregate
                if metric in a.get("scores", {}) and a["scores"][metric] is not None
            ]
            row[f"{metric}_avg"] = (
                round(sum(values) / len(values), 4) if values else ""
            )
            row[f"{metric}_n"] = len(values)
        rows.append(row)

    if not rows:
        print("No results found to aggregate.")
        return

    fieldnames = ["checkpoint", "num_queries", "no_final_answer"] + [
        f"{m}_{s}" for m in ALL_METRICS for s in ("avg", "n")
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary CSV written to {output_path}")


def build_detailed_summary_csv(results_dir: Path, output_path: Path) -> None:
    """Build detailed breakdown by checkpoint x difficulty_tier and checkpoint x task_type_bucket."""
    rows: list[dict] = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "__pycache__":
            continue

        checkpoint_name = model_dir.name
        aggregate = build_aggregate(model_dir)
        if not aggregate:
            continue

        # Group by difficulty_tier
        by_tier: dict[str, list[dict]] = {}
        for a in aggregate:
            tier = a.get("difficulty_tier", "unknown") or "unknown"
            by_tier.setdefault(tier, []).append(a)

        for tier, entries in sorted(by_tier.items()):
            row = {
                "checkpoint": checkpoint_name,
                "group_by": "difficulty_tier",
                "group_value": tier,
                "num_queries": len(entries),
            }
            for metric in ALL_METRICS:
                values = [
                    e["scores"][metric]
                    for e in entries
                    if metric in e.get("scores", {})
                    and e["scores"][metric] is not None
                ]
                row[f"{metric}_avg"] = (
                    round(sum(values) / len(values), 4) if values else ""
                )
            rows.append(row)

        # Group by task_type_bucket
        by_bucket: dict[str, list[dict]] = {}
        for a in aggregate:
            bucket = a.get("task_type_bucket", "unknown") or "unknown"
            by_bucket.setdefault(bucket, []).append(a)

        for bucket, entries in sorted(by_bucket.items()):
            row = {
                "checkpoint": checkpoint_name,
                "group_by": "task_type_bucket",
                "group_value": bucket,
                "num_queries": len(entries),
            }
            for metric in ALL_METRICS:
                values = [
                    e["scores"][metric]
                    for e in entries
                    if metric in e.get("scores", {})
                    and e["scores"][metric] is not None
                ]
                row[f"{metric}_avg"] = (
                    round(sum(values) / len(values), 4) if values else ""
                )
            rows.append(row)

    if not rows:
        print("No results found for detailed summary.")
        return

    fieldnames = ["checkpoint", "group_by", "group_value", "num_queries"] + [
        f"{m}_avg" for m in ALL_METRICS
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Detailed summary CSV written to {output_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rollout_checkpoint(
    checkpoint_dir: str, model_name: str | None = None
) -> tuple[list[dict], dict[int, dict]]:
    """Load a rollout checkpoint and convert to (candidates, golden_by_id).

    Returns:
        candidates: list of candidate dicts (prompt + rejected)
        golden_by_id: dict mapping id -> golden dict (prompt + chosen)
    """
    rollout_path = Path(checkpoint_dir) / "rollout_results.json"
    if not rollout_path.exists():
        raise FileNotFoundError(f"No rollout_results.json found in {checkpoint_dir}")

    with open(rollout_path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    candidates = []
    golden_by_id = {}

    for raw in raw_entries:
        if raw.get("status") != "success":
            continue

        candidate, golden = convert_rollout_entry(raw)

        # Override model name if provided
        if model_name:
            candidate["model"] = model_name

        candidates.append(candidate)
        golden_by_id[candidate["id"]] = golden

    return candidates, golden_by_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Rubrics-based evaluation of trained-model rollout trajectories."
)


@app.command()
def evaluate(
    checkpoint_dir: str = typer.Option(
        ...,
        "--checkpoint-dir",
        "-d",
        help="Path to a checkpoint directory containing rollout_results.json.",
    ),
    output_dir: str = typer.Option(
        "rubrics_results_trained",
        "--output-dir",
        "-o",
        help="Base output directory for results.",
    ),
    judge_model: str = typer.Option(
        "claude-opus-4-6",
        "--judge-model",
        "-m",
        help="Model to use as LLM judge.",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        help="Override model name in output records.",
    ),
    metrics: str = typer.Option(
        "all",
        "--metrics",
        help="Comma-separated list of metrics to run, or 'all'.",
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Max concurrent LLM judge calls."
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all)."
    ),
    llm_rate_limit_sleep: int = typer.Option(
        60, "--llm-rate-limit-sleep", help="Seconds to pause on 429."
    ),
    llm_server_error_sleep: int = typer.Option(
        30, "--llm-server-error-sleep", help="Seconds to pause on 500."
    ),
    llm_max_retries: int = typer.Option(
        3, "--llm-max-retries", help="Max retries per LLM call."
    ),
    batch: bool = typer.Option(
        True,
        "--batch/--no-batch",
        help="Use the Anthropic Message Batches API when the judge is a Claude model.",
    ),
    batch_poll_interval: int = typer.Option(
        30, "--batch-poll-interval", help="Seconds between batch status polls."
    ),
) -> None:
    """Evaluate a single checkpoint's rollout trajectories."""
    global _llm_max_retries
    _llm_max_retries = llm_max_retries
    configure_llm_guard(
        rate_limit_sleep=llm_rate_limit_sleep,
        server_error_sleep=llm_server_error_sleep,
        max_retries=llm_max_retries,
    )

    # Parse metrics
    if metrics.strip().lower() == "all":
        selected_metrics = list(ALL_METRICS)
    else:
        selected_metrics = [m.strip() for m in metrics.split(",")]
        invalid = [m for m in selected_metrics if m not in ALL_METRICS]
        if invalid:
            print(f"ERROR: Unknown metrics: {invalid}")
            print(f"Available: {ALL_METRICS}")
            raise typer.Exit(1)

    # Derive checkpoint label for output directory
    ckpt_path = Path(checkpoint_dir).resolve()
    ckpt_label = ckpt_path.name  # e.g. "checkpoint-320"

    print(f"Checkpoint: {ckpt_label}")
    print(f"Metrics: {selected_metrics}")

    # Load data
    print(f"Loading rollout data from {checkpoint_dir}...")
    candidates, golden_by_id = load_rollout_checkpoint(checkpoint_dir, model_name)
    print(f"  {len(candidates)} candidate trajectories loaded.")

    # Stats
    has_answer = sum(1 for c in candidates if c.get("has_final_answer"))
    print(f"  {has_answer}/{len(candidates)} have a final answer, "
          f"{len(candidates) - has_answer} ended mid-trajectory (no final answer).")

    # Apply query limit
    if num_queries > 0:
        candidates = candidates[:num_queries]
        print(f"  Limited to first {num_queries} queries.")

    # Prepare output directory
    ckpt_out = Path(output_dir) / ckpt_label
    ckpt_out.mkdir(parents=True, exist_ok=True)

    use_batch = batch and _is_claude_model(judge_model)
    if use_batch:
        print(
            f"Judge is a Claude model — using the Anthropic Message Batches API "
            f"(poll every {batch_poll_interval}s). Pass --no-batch for per-request mode."
        )

    # Run evaluation
    async def _run() -> None:
        algo_metrics = [m for m in selected_metrics if m in ALGORITHMIC_METRICS]
        llm_metrics = [m for m in selected_metrics if m in LLM_JUDGED_METRICS]

        # Algorithmic metrics: synchronous per-metric compute (unchanged).
        for metric in algo_metrics:
            print(f"\n--- Evaluating: {metric} ---")
            await evaluate_metric_batch(
                metric_name=metric,
                candidates=candidates,
                golden_by_id=golden_by_id,
                judge_model=judge_model,
                output_path=ckpt_out / f"{metric}.json",
                concurrency=concurrency,
            )

        # LLM-judged metrics: one Anthropic batch for Claude judges, else per-request.
        if llm_metrics:
            if use_batch:
                print(f"\n--- Evaluating (batch): {llm_metrics} ---")
                await run_llm_metrics_via_batch(
                    metric_names=llm_metrics,
                    candidates=candidates,
                    golden_by_id=golden_by_id,
                    judge_model=judge_model,
                    output_paths={m: ckpt_out / f"{m}.json" for m in llm_metrics},
                    poll_interval=batch_poll_interval,
                )
            else:
                for metric in llm_metrics:
                    print(f"\n--- Evaluating: {metric} ---")
                    await evaluate_metric_batch(
                        metric_name=metric,
                        candidates=candidates,
                        golden_by_id=golden_by_id,
                        judge_model=judge_model,
                        output_path=ckpt_out / f"{metric}.json",
                        concurrency=concurrency,
                    )

        # Build aggregate
        agg = build_aggregate(ckpt_out)
        agg_path = ckpt_out / "aggregate.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"\nAggregate written to {agg_path}")

        # Print summary
        print(f"\n{'Metric':<30} {'Avg':>8} {'N':>6}")
        print("-" * 46)
        for metric in selected_metrics:
            values = [
                a["scores"][metric]
                for a in agg
                if metric in a.get("scores", {}) and a["scores"][metric] is not None
            ]
            if values:
                avg = sum(values) / len(values)
                max_score = 5 if metric in LLM_JUDGED_METRICS else 1.0
                print(f"{metric:<30} {avg:>8.4f}/{max_score} {len(values):>5}")

    asyncio.run(_run())


@app.command()
def evaluate_all(
    rollout_dir: str = typer.Option(
        ...,
        "--rollout-dir",
        "-d",
        help="Path to rollout directory containing checkpoint-* subdirectories.",
    ),
    output_dir: str = typer.Option(
        "rubrics_results_trained",
        "--output-dir",
        "-o",
        help="Base output directory for results.",
    ),
    judge_model: str = typer.Option(
        "claude-opus-4-6",
        "--judge-model",
        "-m",
        help="Model to use as LLM judge.",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        help="Override model name in output records.",
    ),
    metrics: str = typer.Option(
        "all",
        "--metrics",
        help="Comma-separated list of metrics to run, or 'all'.",
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Max concurrent LLM judge calls."
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all)."
    ),
    llm_rate_limit_sleep: int = typer.Option(
        60, "--llm-rate-limit-sleep", help="Seconds to pause on 429."
    ),
    llm_server_error_sleep: int = typer.Option(
        30, "--llm-server-error-sleep", help="Seconds to pause on 500."
    ),
    llm_max_retries: int = typer.Option(
        3, "--llm-max-retries", help="Max retries per LLM call."
    ),
) -> None:
    """Evaluate all checkpoints under a rollout directory."""
    rdir = Path(rollout_dir)
    checkpoints = sorted(
        [d for d in rdir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    )
    if not checkpoints:
        print(f"No checkpoint-* directories found in {rollout_dir}")
        raise typer.Exit(1)

    print(f"Found {len(checkpoints)} checkpoints: {[c.name for c in checkpoints]}")

    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {ckpt.name}")
        print(f"{'='*60}")

        # Build args for the evaluate command and invoke it
        evaluate(
            checkpoint_dir=str(ckpt),
            output_dir=output_dir,
            judge_model=judge_model,
            model_name=model_name,
            metrics=metrics,
            concurrency=concurrency,
            num_queries=num_queries,
            llm_rate_limit_sleep=llm_rate_limit_sleep,
            llm_server_error_sleep=llm_server_error_sleep,
            llm_max_retries=llm_max_retries,
        )

    # Final cross-checkpoint aggregate
    print(f"\n{'='*60}")
    print("  Aggregating across checkpoints")
    print(f"{'='*60}")
    aggregate(results_dir=output_dir)


@app.command()
def aggregate(
    results_dir: str = typer.Option(
        "rubrics_results_trained",
        "--results-dir",
        "-r",
        help="Directory containing per-checkpoint result subdirectories.",
    ),
) -> None:
    """Aggregate all per-metric results into cross-checkpoint summary CSVs."""
    rdir = Path(results_dir)
    if not rdir.exists():
        print(f"ERROR: {rdir} does not exist.")
        raise typer.Exit(1)

    # Build per-checkpoint aggregates
    for model_dir in sorted(rdir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "__pycache__":
            continue
        agg = build_aggregate(model_dir)
        agg_path = model_dir / "aggregate.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"  {model_dir.name}: {len(agg)} queries aggregated")

    # Summary CSV
    build_summary_csv(rdir, rdir / "summary.csv")

    # Detailed breakdown CSV
    build_detailed_summary_csv(rdir, rdir / "summary_detailed.csv")


if __name__ == "__main__":
    app()
