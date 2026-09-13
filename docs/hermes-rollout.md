# Hermes-backed Peter

Status: implementation in progress. The deployed bot remains on the previous bounded harness until the staged integration passes verification.

## Decision

Integrate upstream Hermes Agent as a Python dependency pinned to the official v2026.9.11 release commit. Do not fork or use a Git submodule. Peter owns Discord identity, authorization, scoped memory, the job queue, delivery, and the trusted execution supervisor. Hermes owns the model/tool loop and sandbox-local file/terminal tools.

## Boundaries

- The Discord gateway resolves user and role IDs from Discord, including fresh membership checks before tool actions. Conversation text cannot grant authority.
- Public club memory is officer-writable. Personal memory is bound to the originating guild/user. Changes are versioned and audited. Memory never grants permissions.
- Each task runs in its own unprivileged, read-only-root Docker sandbox with bounded tmpfs, CPU, RAM, process count and runtime. No host mounts, host credentials, Discord token or Docker socket enter the sandbox.
- The trusted runner alone controls Docker. It accepts a fixed job schema, not arbitrary Docker arguments. Its control network is separate from the worker network.
- Workers use an internal network. A per-task revocable token grants only that task's model proxy and explicitly allowed tool calls. Network isolation also blocks worker-to-host access. The initial worker has no unrestricted internet, package installation, native Hermes global memory, cron, cross-session search, messaging, or subagents.
- Gemini/OpenAI/etc. fallbacks are not enabled. The existing Qwen endpoint remains the model, with thinking enabled and a larger response budget.
- Member access initially keeps the legacy conversational path. Officer pilot uses isolated Discord task threads, durable queue state and explicit continuation/cancellation.

## Delivery sequence

1. Establish baseline tests, pin Hermes, implement/test policy, memory, worker and runner.
2. Implement Discord queue integration, task-scoped tool/model proxy and operational controls.
3. Push tested commits. Build images on p910, save previous deployment configuration, stage without a second live Discord connection.
4. Run real-model tool, memory and code/artifact smoke tests plus network/authorization rejection tests.
5. Switch the live deployment, check health and gateway connection, preserve rollback instructions.

## Acceptance

No member can write club memory, impersonate another actor, retrieve another member's private state, choose arbitrary runner arguments, or reach p910 host services from a worker. Thinking-enabled Hermes can call tools, execute code in its workspace, and return a real artifact. Role revocation prevents further task tool use. Queue and memory survive gateway restart. User cancellation terminates the corresponding sandbox. Existing bot commands remain available.

## Later capability gates

A separate VM for untrusted execution remains the preferred stronger host boundary; a restricted Docker pilot is the first milestone and shares the host kernel. General browser automation, broader network/package access, private officer knowledge, outbound club actions, scheduled agent tasks, and delegation require explicit scoped implementations and verification. Native upstream features are not automatically exposed merely because Hermes supports them.
