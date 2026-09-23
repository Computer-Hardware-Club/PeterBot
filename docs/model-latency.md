# Conversation latency and tier budgets (PETER-05)

Deployed model: `Qwen3.8-Flash-Next` on the p910-network vLLM host
(`docs/model-backend-baseline-2026-09-22.md`). Thinking is billed against the same
completion budget as the answer, and live notes show thinking-only turns ending
blank with `finish_reason=stop` — so a blanket 4096-token thinking allowance is both
slow for greetings and still not safe for hard questions. `peterbot/conversation.py`
instead bounds the turn with three tiers. The tier bounds the **generation shape
only**: the model normally decides whether to call `use_tools`. One output
postcondition prevents a clean text reply from claiming an explicitly requested
attachment or sandbox execution; that request is handed to the worker instead.

## Tiers

| tier | trigger (shape, not topic) | thinking | budget tokens | thinking cap | effort |
| --- | --- | --- | --- | --- | --- |
| casual | short greeting/banter, ≤2 context turns | off | 512 | — | — |
| normal | explain/compare/how-does style questions, long or multi-`?` messages | on | 2048 | 1024 | low |
| deep | current facts, research, code/files/club-state verbs, attachments, explicit depth requests | on | 4096 (follows `inference.max_tokens`, clamped 4096–8192) | 2048 | low |

- Thinking stays **on** for normal/deep: earlier live notes report unreliable tool
  routing without thinking, and a handoff must remain reachable from any tier.
- `reasoning_effort` is only ever sent together with `enable_thinking: true`;
  the served vLLM build rejects the combination otherwise. A capped thinking budget
  therefore always pairs with an effort value.
- A non-thinking rescue attempt (blank nudge or continue-prompt) follows any blank,
  truncated, or malformed first attempt.

## Wall-clock budget

One budget covers the whole turn: `budget_seconds` (scheduler remaining deadline)
or `inference.timeout_seconds`.

- Thinking attempt: `min(max(remaining − 130 s reserve, remaining/2), 300 s)`.
  The 130 s reserve guarantees the rescue can still answer after the model thinks
  for three minutes.
- Rescue attempt: ≤120 s, and never started with <10 s left.
- `max_tokens` is additionally capped at `remaining × 20 tok/s` (served host does
  ~40 tok/s aggregate; 20 is the conservative half) so an oversized call never
  starts near the deadline.
- Nothing runs with <5 s left; the turn then returns the safe blank line instead of
  burning the gateway deadline.

## Failure semantics (no false sandbox handoff)

- `use_tools` authorizes a handoff **only** with one call, well-formed
  `{"reason": …}` arguments, and a completion finish marker (`tool_calls`/`stop`).
- `finish_reason=length`/`content_filter`, a missing finish marker, malformed or
  invented tool arguments ⇒ retry without thinking, never a handoff, never member-visible.
- A clean text promise cannot satisfy an explicit request to attach source/files
  or compile and test in Peter's sandbox. That answer hands off to real work;
  malformed tool output still does not authorize a handoff.
- Transport failure on every attempt ⇒ `ValueError(MODEL_UNAVAILABLE_REPLY)`;
  two blank completions ⇒ the canned retry-line. A ≥40-char truncated answer is
  kept as a last resort rather than replaced by a canned line.
- Reasoning text is never returned to a member.

## Gateway/worker integration

`reply_or_use_tools(..., has_attachments=..., budget_seconds=...)` are additive
kwargs. The trusted gateway passes fresh `club_context`, `style_instruction`,
durable scoped context, and the request's remaining conversation time to the
model. `/ask` uses what remains after its history fetch. Attached files go
directly to the isolated work path, so they never enter a casual model turn.
The staged P910 configuration sets ordinary conversation requests to 90 seconds
and substantial work to 600 seconds; live timing remains a release gate.

## Live probe

`deploy/probe_model_compat.py` runs synthetic prompts only and reports timings,
usage, retries, and route validity (never prompt or reasoning text). `--metrics-url`
samples `vllm:num_requests_running/waiting` and KV usage around each row so loaded
samples cannot masquerade as idle ones.

### Compatibility (server lightly loaded — running=1: compatibility only, not warm-idle latency)

2026-09-23 ~04:20 UTC, host 100.73.210.66:8000, one repeat per row, retries 0,
malformed chunks 0.

| case | mode | route (expected) | first useful s | total s | completion/reasoning tok | finish | valid args |
| --- | --- | --- | --- | --- | --- | --- | --- |
| greeting | none | answer (answer) ✓ | 1.30 | 1.97 | 18/0 | stop | — |
| greeting | low | answer (answer) ✓ | 2.39 | 3.71 | 77/37 | stop | — |
| research | none | handoff (handoff) ✓ | 1.36 | 3.75 | 68/0 | tool_calls | 1 |
| research | low | handoff (handoff) ✓ | 2.97 | 5.08 | 114/42 | tool_calls | 1 |

### Idle warm p50/p95 — DEFERRED

Accepted control surface, verified against the served build: `enable_thinking`
false/true, `reasoning_effort=low` **with** thinking, and
`chat_template_kwargs.thinking_budget`; tool deltas arrive with
`finish_reason=tool_calls` and parseable `{"reason":…}` args under both thinking
modes. The non-thinking research sample also routed correctly, but one sample does
not overturn the earlier live no-thinking failures, so deep keeps thinking per the
PETER-05 brief.

Local agents hold the server during this pass; per coordination notes, warm-idle
p50/p95 is left for Codex after all agents finish, with
`--repeat 5 --metrics-url http://<host>:8000/metrics` while the dashboard reads
0 running / 0 waiting.
