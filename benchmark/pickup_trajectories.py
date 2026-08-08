"""
LLM-as-Judge: pick the best answer from 3 model trajectories per query.

Usage:
    python benchmark/pickup_trajectories.py --judge-model claude-opus-4-6
    python benchmark/pickup_trajectories.py --judge-model claude-opus-4-6 --num-queries 5
"""

import asyncio
import json
import random
import re
import sys
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

TRAJECTORY_FILES = [
    "selected_800_claude-opus-4-6_trajectories.json",
    "selected_800_gemini_gemini-3.1-pro-preview_trajectories.json",
    "selected_800_gpt-5.4_trajectories.json",
]

MODEL_NAMES = [
    "claude-opus-4-6",
    "gemini/gemini-3.1-pro-preview",
    "gpt-5.4",
]

SYSTEM_PROMPT = """\
You are an expert financial analyst acting as an impartial judge.

You will be given a financial question, a reference (ground truth) answer, and \
three candidate answers labeled A, B, and C. Each candidate answer includes the \
full conversation trajectory showing the assistant's reasoning and tool calls.

Your task: decide which candidate answer is the best overall, considering:
- **Correctness**: Does the final answer match the reference answer in key facts and figures?
- **Completeness**: Does it address all parts of the question?
- **Reasoning quality**: Is the step-by-step reasoning sound?
- **Tool usage**: Did the assistant call the right tools with correct arguments?
- **Clarity**: Is the final answer well-organized and concise?

You MUST respond with valid JSON only, no other text. Use this exact format:
{"winner": "A", "justification": "2-3 sentence explanation"}

The "winner" field must be exactly "A", "B", or "C"."""

USER_PROMPT_TEMPLATE = """\
## Financial Question
{source_query}

## Reference Answer
{reference_answer}

## Candidate Answer A
{answer_a}

## Candidate Answer B
{answer_b}

## Candidate Answer C
{answer_c}

Pick the best answer. Respond with JSON only: {{"winner": "A"|"B"|"C", "justification": "..."}}"""

MAX_TOOL_OUTPUT_CHARS = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_trajectory(messages: list[dict]) -> str:
    """Format a message history into a readable trajectory string.

    Truncates long tool outputs to keep the prompt manageable.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""

        if role == "system":
            continue  # skip system prompt — same for all models

        if role == "user":
            parts.append(f"[User]\n{content}")

        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
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


def shuffle_answers(
    query_id: int, model_answers: dict[str, str]
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Randomize assignment of model answers to positions A/B/C.

    Uses a deterministic seed for reproducibility.

    Returns:
        ordered: list of (label, model_name, answer_text)
        position_map: {"A": model_name, "B": model_name, "C": model_name}
    """
    rng = random.Random(query_id)
    models = list(model_answers.keys())
    rng.shuffle(models)
    labels = ["A", "B", "C"]
    ordered = [(labels[i], models[i], model_answers[models[i]]) for i in range(3)]
    position_map = {labels[i]: models[i] for i in range(3)}
    return ordered, position_map


def build_judge_prompt(
    source_query: str,
    reference_answer: str,
    ordered_answers: list[tuple[str, str, str]],
) -> list[dict]:
    """Build the messages list for the judge LLM call."""
    answer_map = {label: text for label, _, text in ordered_answers}
    user_content = USER_PROMPT_TEMPLATE.format(
        source_query=source_query,
        reference_answer=reference_answer,
        answer_a=answer_map["A"],
        answer_b=answer_map["B"],
        answer_c=answer_map["C"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_judge_response(raw_text: str) -> tuple[dict | None, str | None]:
    """Parse and validate the judge's JSON response.

    Returns (parsed_dict, error_message). error_message is None on success.
    """
    text = raw_text.strip()
    # Strip markdown code fences
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

    if "winner" not in result:
        return None, "Missing 'winner' key"
    if result["winner"] not in ("A", "B", "C"):
        return None, f"Invalid winner: {result['winner']}"

    return result, None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_llm_max_retries = 3


async def judge_llm_call(model: str, messages: list[dict]) -> str:
    """Call the judge LLM and return the raw response text."""
    import litellm

    retries = 0
    while True:
        event, _ = _get_llm_rate_limit_primitives()
        await event.wait()

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
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
# Core judging logic
# ---------------------------------------------------------------------------


async def judge_single_query(
    query_id: int,
    trajectories: dict[str, dict],
    judge_model: str,
    semaphore: asyncio.Semaphore,
    all_records: list[dict],
    save_lock: asyncio.Lock,
    output_path: Path,
    pbar,
) -> None:
    """Judge a single query: shuffle, prompt, call LLM, parse, save."""
    try:
        async with semaphore:
            # Get the first trajectory for shared metadata
            first = next(iter(trajectories.values()))
            source_query = first["source_query"]
            reference_answer = first.get("reference_answer", "")

            # Build formatted trajectories per model
            model_answers: dict[str, str] = {}
            for model_name, traj in trajectories.items():
                final_answer = traj.get("final_answer", "") or ""
                messages = traj.get("messages", [])
                formatted = _format_trajectory(messages)
                if final_answer:
                    formatted += f"\n\n[Final Answer]\n{final_answer}"
                else:
                    formatted = "[No answer provided — the model failed to generate a response]"
                model_answers[model_name] = formatted

            ordered, position_map = shuffle_answers(query_id, model_answers)
            messages = build_judge_prompt(source_query, reference_answer, ordered)

            raw_response = await judge_llm_call(judge_model, messages)
            parsed, parse_error = parse_judge_response(raw_response)

            # Retry once on parse failure
            if parse_error:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your response was not valid JSON. Please respond with ONLY a JSON object: "
                        '{"winner": "A"|"B"|"C", "justification": "..."}',
                    }
                )
                raw_response = await judge_llm_call(judge_model, messages)
                parsed, parse_error = parse_judge_response(raw_response)

            record = {
                "id": query_id,
                "source_query": source_query,
                "reference_answer": reference_answer,
                "difficulty_tier": first.get("difficulty_tier", ""),
                "task_type_bucket": first.get("task_type_bucket", ""),
                "position_map": position_map,
                "judge_model": judge_model,
            }

            if parsed and not parse_error:
                winner_label = parsed["winner"]
                record["winner"] = winner_label
                record["winner_model"] = position_map[winner_label]
                record["justification"] = parsed.get("justification", "")
            else:
                record["winner"] = None
                record["winner_model"] = None
                record["justification"] = f"PARSE ERROR: {parse_error}"
                record["judge_raw_response"] = raw_response

            async with save_lock:
                all_records.append(record)
                sorted_records = sorted(all_records, key=lambda r: r["id"])
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_records, f, indent=2, ensure_ascii=False)

            pbar.update(1)
            pbar.set_postfix(saved=len(all_records))

    except (asyncio.CancelledError, _BaseExceptionGroup) as exc:
        print(f"  [WARN] query {query_id} cancelled: {type(exc).__name__}")
        pbar.update(1)
    except Exception as exc:
        print(f"  [ERROR] query {query_id}: {type(exc).__name__}: {exc}")
        pbar.update(1)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_batch(
    trajectory_dir: str,
    output_dir: str,
    judge_model: str,
    num_queries: int,
    concurrency: int,
) -> None:
    """Load trajectories, run judging, save results."""
    from collections import Counter

    from tqdm import tqdm

    traj_path = Path(trajectory_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all 3 trajectory files and index by query ID
    all_trajectories: dict[int, dict[str, dict]] = {}  # id -> {model -> entry}
    for filename, model_name in zip(TRAJECTORY_FILES, MODEL_NAMES):
        filepath = traj_path / filename
        if not filepath.exists():
            print(f"ERROR: {filepath} not found")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            qid = entry["id"]
            if qid not in all_trajectories:
                all_trajectories[qid] = {}
            all_trajectories[qid][model_name] = entry
        print(f"Loaded {len(entries)} entries from {filename}")

    # Filter to queries that have all 3 model answers
    complete_ids = sorted(
        qid
        for qid, models in all_trajectories.items()
        if len(models) == len(MODEL_NAMES)
    )
    print(f"{len(complete_ids)} queries with all {len(MODEL_NAMES)} model answers")

    if num_queries > 0:
        complete_ids = complete_ids[:num_queries]

    # Resume
    model_slug = judge_model.replace("/", "_")
    output_file = out_path / f"judge_results_{model_slug}.json"
    all_records: list[dict] = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(f"Resumed: {len(all_records)} existing judgments from {output_file}")
        except (json.JSONDecodeError, Exception) as exc:
            print(f"Warning: could not load existing results ({exc}), starting fresh.")
            all_records = []

    processed_ids = {r["id"] for r in all_records}
    remaining_ids = [qid for qid in complete_ids if qid not in processed_ids]
    if not remaining_ids:
        print("All queries already judged.")
    else:
        print(f"{len(remaining_ids)} queries remaining.")

        semaphore = asyncio.Semaphore(concurrency)
        save_lock = asyncio.Lock()

        with tqdm(total=len(remaining_ids), desc="Judging") as pbar:
            tasks = [
                judge_single_query(
                    qid,
                    all_trajectories[qid],
                    judge_model,
                    semaphore,
                    all_records,
                    save_lock,
                    output_file,
                    pbar,
                )
                for qid in remaining_ids
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    # Print summary
    print(f"\nResults saved to {output_file}")
    print(f"Total judged: {len(all_records)}")

    wins = Counter(r["winner_model"] for r in all_records if r.get("winner_model"))
    parse_errors = sum(1 for r in all_records if r.get("winner") is None)

    print(f"\n{'Model':<40} Wins")
    print("-" * 50)
    for model in MODEL_NAMES:
        print(f"{model:<40} {wins.get(model, 0)}")
    if parse_errors:
        print(f"\nParse errors: {parse_errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="LLM-as-Judge: pick the best answer from 3 model trajectories.")


@app.command()
def judge(
    trajectory_dir: str = typer.Option(
        "data/trajectory/testset_trajectory",
        "--trajectory-dir",
        "-t",
        help="Directory containing the 3 trajectory JSON files.",
    ),
    output_dir: str = typer.Option(
        "data/testset/pickup_results",
        "--output-dir",
        "-o",
        help="Directory to save judgment results.",
    ),
    judge_model: str = typer.Option(
        "claude-opus-4-6",
        "--judge-model",
        "-m",
        help="Model to use as judge.",
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all)."
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Max concurrent judge calls."
    ),
    llm_rate_limit_sleep: int = typer.Option(
        60,
        "--llm-rate-limit-sleep",
        help="Seconds to pause all LLM calls on 429.",
    ),
    llm_server_error_sleep: int = typer.Option(
        30,
        "--llm-server-error-sleep",
        help="Seconds to pause all LLM calls on 500/overload.",
    ),
    llm_max_retries: int = typer.Option(
        3,
        "--llm-max-retries",
        help="Max retries per LLM call on rate limit/server errors.",
    ),
) -> None:
    """Run LLM-as-Judge on all queries."""
    global _llm_max_retries
    _llm_max_retries = llm_max_retries
    configure_llm_guard(
        rate_limit_sleep=llm_rate_limit_sleep,
        server_error_sleep=llm_server_error_sleep,
        max_retries=llm_max_retries,
    )
    asyncio.run(
        process_batch(
            trajectory_dir=trajectory_dir,
            output_dir=output_dir,
            judge_model=judge_model,
            num_queries=num_queries,
            concurrency=concurrency,
        )
    )


if __name__ == "__main__":
    app()
