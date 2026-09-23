"""Bounded live Qwen compatibility/latency probe with synthetic prompts only.

The report contains timings, token usage, termination and route shape, never
reasoning text. Run against the actual served endpoint while it is otherwise
idle; compare modes before changing Peter's conversation budget.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from urllib import error, request


TOOL = {"type": "function", "function": {
    "name": "use_tools", "description": "Use tools for current research or coding tasks, not greetings.",
    "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                   "required": ["reason"], "additionalProperties": False}}}
CASES = {
    "greeting": "yo Peter, how's it going?",
    "factual": "What is binary search? Answer in a couple of sentences.",
    "research": "Find the current stable Rust release using a source and cite it.",
    "coding": "Create and test a Rust command-line program that computes e to 100 decimal digits and give me its files.",
}


def measure(url: str, model: str, api_key: str, mode: str, case: str,
            timeout: float = 35.0) -> dict:
    payload = {
        "model": model, "messages": [
            {"role": "system", "content": "You are Peter, a casual but capable club engineering bot. Reply naturally to conversation; call use_tools for research or coding that must be executed. Never claim unexecuted work."},
            {"role": "user", "content": CASES[case]},
        ],
        "tools": [TOOL], "tool_choice": "auto", "parallel_tool_calls": False,
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": 768 if mode == "none" else 1536,
        "temperature": 0.7 if mode == "none" else 1.0,
        "chat_template_kwargs": {"enable_thinking": mode != "none"},
    }
    if mode != "none":
        payload["reasoning_effort"] = mode
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    start = time.monotonic()
    first = None
    first_answer = None
    finish = None
    text = []
    tool_names = []
    tool_arguments: dict[int, str] = {}
    usage = {}
    malformed = 0
    response = request.Request(url.rstrip("/") + "/chat/completions",
                               data=json.dumps(payload).encode(), headers=headers)
    try:
        with request.urlopen(response, timeout=timeout) as stream:
            for raw in stream:
                if not raw.startswith(b"data:"):
                    continue
                part = raw[5:].strip()
                if part == b"[DONE]":
                    break
                try:
                    chunk = json.loads(part)
                except ValueError:
                    malformed += 1
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    name = (call.get("function") or {}).get("name")
                    if name:
                        tool_names.append(name)
                    arguments = (call.get("function") or {}).get("arguments")
                    if arguments:
                        tool_arguments[index] = tool_arguments.get(index, "") + arguments
                if first_answer is None and (delta.get("content") or delta.get("tool_calls")):
                    first_answer = time.monotonic() - start
                if first is None and (delta.get("content") or delta.get("reasoning")
                                      or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first = time.monotonic() - start
    except (error.HTTPError, error.URLError, TimeoutError) as exc:
        return {"case": case, "mode": mode, "error_type": type(exc).__name__,
                "status": getattr(exc, "code", None), "seconds": round(time.monotonic() - start, 3)}
    valid_tools = 0
    for arguments in tool_arguments.values():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("reason"), str):
            valid_tools += 1
    return {"case": case, "mode": mode, "first_token_s": round(first, 3) if first else None,
            "first_answer_s": round(first_answer, 3) if first_answer else None,
            "seconds": round(time.monotonic() - start, 3), "finish_reason": finish,
            "answer_chars": len("".join(text)), "tool_names": list(dict.fromkeys(tool_names)),
            "valid_tool_arguments": valid_tools, "malformed_chunks": malformed, "usage": usage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("none", "low", "medium"), action="append")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5:
        parser.error("repeat must be 1 to 5")
    key = os.environ.get("MODEL_API_KEY", "")
    results = []
    for mode in args.mode or ("none", "low", "medium"):
        for case in args.case or CASES:
            for _ in range(args.repeat):
                row = measure(args.base_url, args.model, key, mode, case)
                print(json.dumps(row, sort_keys=True), flush=True)
                results.append(row)
    for case in args.case or CASES:
        for mode in args.mode or ("none", "low", "medium"):
            timings = [row["seconds"] for row in results if row["case"] == case
                       and row["mode"] == mode and "error_type" not in row]
            if timings:
                print(json.dumps({"summary_case": case, "mode": mode,
                                  "median_s": round(statistics.median(timings), 3),
                                  "max_s": round(max(timings), 3)}), flush=True)


if __name__ == "__main__":
    main()
