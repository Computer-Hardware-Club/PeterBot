"""Bounded live Qwen compatibility/latency probe with synthetic prompts only.

The report contains timings, token usage, termination and route shape, never
reasoning text or prompt text. Run against the actual served endpoint while it
is otherwise idle; compare modes before changing Peter's conversation budget.

Each row records:
  first_content_s  first visible answer token
  first_tool_s     first tool-call delta (the useful signal for handoff cases)
  first_useful_s   whichever of the two arrived first
  seconds          total wall time, retries included
  retries          transport/HTTP failures retried once (the conversation path may do the same)
  route/route_expected/route_valid
                   'answer' vs 'handoff' vs 'none'; the expected route encodes the
                   contract (greetings must answer, research/coding must call use_tools
                   with well-formed arguments), so a loaded server still proves routing.
  finish_reason, answer_chars, tool_names, valid_tool_arguments, malformed_chunks, usage

If --metrics-url is given, server load (running/waiting requests, KV usage) is
sampled before each row so a loaded sample can never be mistaken for warm-idle
latency. Label loaded runs explicitly in the report: they prove compatibility only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from urllib import error, request


TOOL = {"type": "function", "function": {
    "name": "use_tools", "description": "Call when the request needs research, current facts, or executed code.",
    "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                   "required": ["reason"], "additionalProperties": False}}}
CASES = {
    "greeting": ("yo Peter, how's it going?", "answer"),
    "factual": ("What is binary search? Answer in a couple of sentences.", "answer"),
    "research": ("Find the current stable Rust release using a source and cite it.", "handoff"),
    "coding": ("Create and test a Rust command-line program that computes e to 100 decimal digits and give me its files.", "handoff"),
}
# thinking shape per mode, mirroring the conversation tiers: effort is only ever sent
# with thinking enabled (vLLM rejects reasoning_effort with enable_thinking=false).
MODES = {
    "none": {"thinking": False, "effort": None, "budget": 768, "thinking_budget": None, "temperature": 0.7},
    "low": {"thinking": True, "effort": "low", "budget": 2048, "thinking_budget": 1024, "temperature": 1.0},
    "medium": {"thinking": True, "effort": "medium", "budget": 2048, "thinking_budget": 1024, "temperature": 1.0},
}


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear percentile over repeated wall times; one sample stays one sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def load_gauges(metrics_url: str) -> dict:
    try:
        with request.urlopen(metrics_url, timeout=5.0) as response:
            body = response.read().decode("utf-8", "replace")
    except (error.HTTPError, error.URLError, TimeoutError):
        return {}
    wanted = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
              "vllm:kv_cache_usage_perc": "kv_usage"}
    gauges = {}
    for line in body.splitlines():
        for metric, name in wanted.items():
            if line.startswith(metric + "{"):
                try:
                    gauges[name] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return gauges


def attempt(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """One streaming request; returns timing/route facts or an error shape."""
    start = time.monotonic()
    first_content = first_tool = None
    finish = None
    text: list[str] = []
    tool_names: list[str] = []
    tool_arguments: dict[int, str] = {}
    usage: dict = {}
    malformed = 0
    response = request.Request(url, data=json.dumps(payload).encode(), headers=headers)
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
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    if first_content is None:
                        first_content = time.monotonic() - start
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    name = (call.get("function") or {}).get("name")
                    if name:
                        tool_names.append(name)
                    arguments = (call.get("function") or {}).get("arguments")
                    if arguments:
                        tool_arguments[index] = tool_arguments.get(index, "") + arguments
                    if first_tool is None:
                        first_tool = time.monotonic() - start
    except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        return {"error_type": type(exc).__name__, "status": getattr(exc, "code", None),
                "seconds": round(time.monotonic() - start, 3)}
    valid_tools = 0
    for arguments in tool_arguments.values():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("reason"), str) and parsed["reason"]:
            valid_tools += 1
    # A handoff only counts the way the gateway counts it: use_tools, finished, and
    # with parseable arguments. A truncated tool call is not a valid route.
    if tool_names:
        route = "handoff" if tool_names == ["use_tools"] and finish in ("tool_calls", "stop") and valid_tools else "bad_tool_call"
    elif text:
        route = "answer" if finish in (None, "stop") else "truncated_answer"
    else:
        route = "none"
    useful = [t for t in (first_content, first_tool) if t is not None]
    return {"error_type": None, "first_content_s": round(first_content, 3) if first_content else None,
            "first_tool_s": round(first_tool, 3) if first_tool else None,
            "first_useful_s": round(min(useful), 3) if useful else None,
            "seconds": round(time.monotonic() - start, 3), "finish_reason": finish,
            "route": route, "answer_chars": len("".join(text)),
            "tool_names": list(dict.fromkeys(tool_names)),
            "valid_tool_arguments": valid_tools, "malformed_chunks": malformed, "usage": usage}


def measure(url: str, model: str, api_key: str, mode: str, case: str, timeout: float,
            metrics_url: str) -> dict:
    shape, (prompt, expected) = MODES[mode], CASES[case]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Peter, a casual but capable club engineering bot. Reply naturally to conversation; call use_tools for research or coding that must be executed. Never claim unexecuted work."},
            {"role": "user", "content": prompt},
        ],
        "tools": [TOOL], "tool_choice": "auto", "parallel_tool_calls": False,
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": shape["budget"], "temperature": shape["temperature"],
        "chat_template_kwargs": {"enable_thinking": shape["thinking"]},
    }
    if shape["thinking"]:
        payload["reasoning_effort"] = shape["effort"]
        payload["chat_template_kwargs"]["thinking_budget"] = shape["thinking_budget"]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    endpoint = url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    load = load_gauges(metrics_url) if metrics_url else {}
    retries = 0
    result = attempt(endpoint, payload, headers, timeout)
    if result["error_type"]:
        retries = 1
        result = attempt(endpoint, payload, headers, timeout)
    row = {"case": case, "mode": mode, "route_expected": expected, **result,
           "retries": retries, "total_s": round(time.monotonic() - started, 3)}
    if result["error_type"] is None:
        row["route_valid"] = row["route"] == expected
    row.update({"load_" + key: value for key, value in load.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=tuple(MODES), action="append")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--metrics-url", default="",
                        help="vLLM /metrics URL for server-load labels (optional)")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5:
        parser.error("repeat must be 1 to 5")
    key = os.environ.get("MODEL_API_KEY", "")
    modes = args.mode or tuple(MODES)
    cases = args.case or tuple(CASES)
    results = []
    for mode in modes:
        for case in cases:
            for _ in range(args.repeat):
                row = measure(args.base_url, args.model, key, mode, case, args.timeout, args.metrics_url)
                print(json.dumps(row, sort_keys=True), flush=True)
                results.append(row)
    for case in cases:
        for mode in modes:
            rows = [row for row in results if row["case"] == case and row["mode"] == mode]
            answered = [row for row in rows if row.get("error_type") is None]
            useful = [row["first_useful_s"] for row in answered if row["first_useful_s"] is not None]
            timings = [row["total_s"] for row in answered]
            if not rows:
                continue
            print(json.dumps({"summary_case": case, "mode": mode, "ok": len(answered), "of": len(rows),
                              "route_valid": sum(1 for row in answered if row.get("route_valid")),
                              "first_useful_p50_s": percentile(useful, 0.5),
                              "first_useful_p95_s": percentile(useful, 0.95),
                              "p50_s": percentile(timings, 0.5),
                              "p95_s": percentile(timings, 0.95),
                              "max_s": round(max(timings), 3) if timings else None,
                              "errors": sorted({str(row.get("error_type")) for row in rows if row.get("error_type")}),
                              "loaded": any((row.get("load_running") or 0) > 0 or (row.get("load_waiting") or 0) > 0
                                            for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
