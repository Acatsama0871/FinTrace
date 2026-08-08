"""
Generate tool-calling trajectories with natural prompting (no think/answer tags).

Usage:
    python benchmark/generate_traj.py --model gpt-5.4
    python benchmark/generate_traj.py --model claude-opus-4-6
    python benchmark/generate_traj.py --model gemini/gemini-3.1-pro-preview
"""

import asyncio
import json
import sys
from typing import Any

import typer  # type: ignore[import-untyped]
from dotenv import load_dotenv

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gen_eval import (
    FMP_MCP_URL,
    LiteLLMClient,
    MCPConnection,
    configure_llm_guard,
    execute_tool_calls_parallel,
    get_logfire,
    is_client_request_error,
)

load_dotenv()

if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup  # noqa: F821
else:
    _BaseExceptionGroup = Exception  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial assistant operating in an interactive environment with access to external tools.\n"
    "Your goal is to answer user queries accurately and efficiently.\n"
)

USER_PROMPT_TEMPLATE = "{query}"

# ---------------------------------------------------------------------------
# Core trajectory generation
# ---------------------------------------------------------------------------


async def generate_trajectory(
    query: str,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LiteLLMClient,
    semaphore: "asyncio.Semaphore",
    max_turns: int = 10,
    max_tool_calls: int = 30,
) -> dict:
    """Run an iterative tool-calling conversation for a single query.

    Returns dict with: messages, final_answer, endpoints_called, num_turns, num_tool_calls.
    """
    logfire = get_logfire()
    user_content = USER_PROMPT_TEMPLATE.format(query=query)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    endpoints_called: list[str] = []
    total_tool_calls = 0
    last_had_tools = False

    with logfire.span("generate_trajectory", query=query[:100], max_turns=max_turns):
        for turn in range(max_turns):
            response = await llm_client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            messages.append(response["raw_message"])

            if not response["tool_calls"]:
                last_had_tools = False
                break

            last_had_tools = True
            total_tool_calls += len(response["tool_calls"])
            endpoints_called.extend(tc["name"] for tc in response["tool_calls"])

            tool_outputs = await execute_tool_calls_parallel(
                conn, response["tool_calls"], semaphore
            )
            messages.extend(tool_outputs)

            if total_tool_calls >= max_tool_calls:
                print(
                    f"  [MAX TOOL CALLS] Reached {total_tool_calls}/{max_tool_calls}, "
                    "forcing stop."
                )
                break

        # Force final answer if last turn had tool calls
        if last_had_tools:
            response = await llm_client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="none",
            )
            messages.append(response["raw_message"])

    # Extract final answer from last assistant message
    final_answer = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("content"):
            final_answer = msg["content"]
            break

    return {
        "messages": messages,
        "final_answer": final_answer,
        "endpoints_called": list(dict.fromkeys(endpoints_called)),
        "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "num_tool_calls": total_tool_calls,
    }


# ---------------------------------------------------------------------------
# Record building & saving
# ---------------------------------------------------------------------------


def _build_record(
    query_item: dict,
    gen_result: dict,
    model_name: str,
) -> dict:
    """Build an output record from query metadata and generation result."""
    return {
        "id": query_item["id"],
        "source_query": query_item["source_query"],
        "task_type": query_item.get("task_type", ""),
        "task_type_bucket": query_item.get("task_type_bucket", ""),
        "resource": query_item.get("resource", ""),
        "data_source": query_item.get("data_source", ""),
        "difficulty_tier": query_item.get("difficulty_tier", ""),
        "difficulty_score": query_item.get("difficulty_score", 0),
        "reference_answer": query_item.get("reference_answer", ""),
        "model": model_name,
        "messages": gen_result["messages"],
        "final_answer": gen_result["final_answer"],
        "endpoints_called": gen_result["endpoints_called"],
        "num_turns": gen_result["num_turns"],
        "num_tool_calls": gen_result["num_tool_calls"],
    }


async def _save_record(
    record: dict,
    all_records: list[dict],
    save_lock: asyncio.Lock,
    output_path: Any,
) -> None:
    """Append record to all_records and write the sorted list to disk."""
    async with save_lock:
        all_records.append(record)
        sorted_records = sorted(all_records, key=lambda r: r["id"])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def process_single_query(
    query_item: dict,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LiteLLMClient,
    mcp_semaphore: asyncio.Semaphore,
    concurrency_semaphore: asyncio.Semaphore,
    max_turns: int,
    max_tool_calls: int,
    max_retries: int,
    retry_delay: float,
    client_error_sleep: float,
    all_records: list[dict],
    save_lock: asyncio.Lock,
    output_path: Any,
    pbar: Any,
    model_name: str,
) -> None:
    """Process a single query with retry logic and incremental saving."""
    import traceback

    logfire = get_logfire()
    idx = query_item["id"]

    try:
        async with concurrency_semaphore:
            query = query_item["source_query"]

            with logfire.span("process_query", query_id=idx, query=query[:100]):
                for attempt in range(1, max_retries + 1):
                    try:
                        gen_result = await generate_trajectory(
                            query,
                            conn,
                            tools,
                            llm_client,
                            mcp_semaphore,
                            max_turns=max_turns,
                            max_tool_calls=max_tool_calls,
                        )
                        record = _build_record(query_item, gen_result, model_name)
                        await _save_record(
                            record, all_records, save_lock, output_path
                        )
                        pbar.update(1)
                        pbar.set_postfix(saved=len(all_records))
                        return

                    except Exception as exc:
                        print(
                            f"[ERROR] query {idx} attempt {attempt}/{max_retries}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()

                        if is_client_request_error(exc):
                            print(
                                f"  [CLIENT ERROR] query {idx}: "
                                f"sleeping {client_error_sleep}s..."
                            )
                            await asyncio.sleep(client_error_sleep)
                            try:
                                gen_result = await generate_trajectory(
                                    query,
                                    conn,
                                    tools,
                                    llm_client,
                                    mcp_semaphore,
                                    max_turns=max_turns,
                                    max_tool_calls=max_tool_calls,
                                )
                                record = _build_record(
                                    query_item, gen_result, model_name
                                )
                                await _save_record(
                                    record, all_records, save_lock, output_path
                                )
                                pbar.update(1)
                                pbar.set_postfix(saved=len(all_records))
                            except Exception as retry_exc:
                                print(
                                    f"  [SKIP] query {idx} still failed: "
                                    f"{type(retry_exc).__name__}: {retry_exc}"
                                )
                            return

                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                        else:
                            print(
                                f"  query {idx} failed after {max_retries} "
                                "retries — skipping"
                            )
                            pbar.update(1)

    except (asyncio.CancelledError, _BaseExceptionGroup) as exc:  # type: ignore
        print(f"  [WARN] query {idx} cancelled: {type(exc).__name__}")
        pbar.update(1)


async def process_testset(
    input_path: str,
    output_dir: str,
    model_name: str,
    num_queries: int,
    max_turns: int,
    max_tool_calls: int,
    concurrency: int,
    mcp_concurrency: int,
    max_retries: int,
    retry_delay: int,
    client_error_sleep: int,
) -> None:
    """Load queries, generate trajectories, and save results."""
    import os
    from pathlib import Path

    from tqdm import tqdm

    logfire = get_logfire()

    fmp_api_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_api_key:
        raise ValueError("Set the FMP_API_KEY environment variable before running.")

    mcp_url = f"{FMP_MCP_URL}?apikey={fmp_api_key}"

    llm_client = LiteLLMClient(model=model_name)
    print(f"Using model: {model_name}")

    with open(input_path, "r", encoding="utf-8") as f:
        query_items: list[dict] = json.load(f)
    if num_queries > 0:
        query_items = query_items[:num_queries]
    total = len(query_items)
    print(f"Loaded {total} queries from {input_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_stem = Path(input_path).stem
    model_slug = model_name.replace("/", "_")
    output_name = f"{input_stem}_{model_slug}_trajectories.json"
    output_path = out / output_name

    # Resume: load existing records
    all_records: list[dict] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(
                f"Resumed: loaded {len(all_records)} existing records from {output_path}"
            )
        except (json.JSONDecodeError, Exception) as exc:
            print(
                f"Warning: could not load existing records ({exc}), starting fresh."
            )
            all_records = []

    processed_ids = {r["id"] for r in all_records}
    remaining = [item for item in query_items if item["id"] not in processed_ids]
    if not remaining:
        print("All queries already processed.")
        return
    print(f"{len(remaining)} queries remaining.")

    concurrency_semaphore = asyncio.Semaphore(concurrency)
    mcp_semaphore = asyncio.Semaphore(mcp_concurrency)
    save_lock = asyncio.Lock()

    conn = MCPConnection(
        mcp_url=mcp_url,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_sleep=client_error_sleep,
    )

    try:
        tools = await conn.start()

        with logfire.span(
            "process_testset",
            model=model_name,
            total=total,
            remaining=len(remaining),
        ):
            with tqdm(total=len(remaining), desc="Generating trajectories") as pbar:
                tasks = [
                    process_single_query(
                        item,
                        conn,
                        tools,
                        llm_client,
                        mcp_semaphore,
                        concurrency_semaphore,
                        max_turns,
                        max_tool_calls,
                        max_retries,
                        retry_delay,
                        client_error_sleep,
                        all_records,
                        save_lock,
                        output_path,
                        pbar,
                        model_name,
                    )
                    for item in remaining
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        await conn.close()

    print(f"\nDone. {len(all_records)}/{total} trajectories saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Generate tool-calling trajectories (natural prompting, no think/answer tags)."
)


@app.command()
def generate(
    input: str = typer.Option(
        "testset/selected_800.json",
        "--input",
        "-i",
        help="Path to the JSON test set file.",
    ),
    output: str = typer.Option(
        "data/trajectory/testset_trajectory",
        "--output",
        "-o",
        help="Directory to save trajectory results.",
    ),
    model: str = typer.Option(
        "gpt-5.4",
        "--model",
        "-m",
        help="Model name (e.g. gpt-5.4, claude-opus-4-6, gemini-3.1-pro-preview).",
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all)."
    ),
    max_turns: int = typer.Option(
        10, "--max-turns", help="Max agentic loop turns per query."
    ),
    max_tool_calls: int = typer.Option(
        30,
        "--max-tool-calls",
        help="Max total tool calls per query.",
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", help="Max concurrent query tasks."
    ),
    mcp_concurrency: int = typer.Option(
        50,
        "--mcp-concurrency",
        help="Max concurrent in-flight MCP calls.",
    ),
    max_retries: int = typer.Option(
        2, "--max-retries", help="Max retries per query on failure."
    ),
    retry_delay: int = typer.Option(
        10, "--retry-delay", help="Seconds between retries."
    ),
    client_error_sleep: int = typer.Option(
        120,
        "--client-error-sleep",
        help="Seconds to sleep on client-side errors before final retry.",
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
    """Generate tool-calling trajectories for a test set using a given LLM."""
    configure_llm_guard(
        rate_limit_sleep=llm_rate_limit_sleep,
        server_error_sleep=llm_server_error_sleep,
        max_retries=llm_max_retries,
    )
    asyncio.run(
        process_testset(
            input_path=input,
            output_dir=output,
            model_name=model,
            num_queries=num_queries,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            concurrency=concurrency,
            mcp_concurrency=mcp_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
            client_error_sleep=client_error_sleep,
        )
    )


if __name__ == "__main__":
    app()
