# Served model baseline — September 22, 2026

Read-only inspection of the live inference server at `100.73.210.66` and its capacity dashboard:

| Setting | Observed value |
| --- | --- |
| Served model name | `Qwen3.8-Flash-Next` |
| Container image | `qwen38-flash-next-vllm:concurrency-20260907` |
| vLLM build | `0.1.dev20073+g8e685d198` |
| Tool parser | `qwen3_xml` |
| Reasoning parser | `qwen3` |
| Automatic tool choice | Enabled |
| Tensor parallel size | 2 |
| Maximum concurrent sequences | 4 |
| Maximum model length | 262,144 tokens |
| Maximum batched tokens | 7,168 |
| KV cache | `fp8_e4m3`, capacity about 975,288 tokens |

The exact model checkpoint and weight quantization were not established by these checks. The dashboard showed about 41 generation tokens per second during one observed active sample; that is aggregate instantaneous throughput, not a user response benchmark.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) documents `enable_thinking` and `reasoning_effort` controls. The current [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) uses `qwen3_coder` for tool calls, while this deployment uses `qwen3_xml`. That difference is a compatibility question to test against the served model; it is not proof that the current parser is broken. Do not change the running model or server from this observation alone.
