# Reproduce a speed report

Build the `tintero` target for CUDA or HIP using the [README instructions](../README.md#quick-start). Use the Inkling 2 Mini GGUF you want to measure. The benchmark launches the engine directly and needs Python 3 and a GPU build.

Run this from the repository root. Substitute your engine path, GGUF, GPU, backend and quant. Use the revision the binary was built from. The command assumes the binary was built from the current checkout.

```sh
printf '10,11,12,13,14,15,16,17\n' > /tmp/tintero-bench-tokens.txt
IK2_KV=fp4-global python3 tools/ik2_bench.py --report \
  --engine ./build/tintero --model ./models/Inkling-2-Mini-IQ4_NL.gguf \
  --gpu 'RTX 3090 24 GB' --backend CUDA --model-quant 'Inkling 2 Mini IQ4_NL' \
  --revision "$(git rev-parse HEAD)" --tokens /tmp/tintero-bench-tokens.txt \
  --depths 512,4096 --ctx 8192 --chunk 128 --gen 129 --repeats 3 > speed-report.md
```

Open a **Speed report** issue and paste the table from `speed-report.md` along with the command you ran. Include your driver and CUDA or ROCm version, KV format and any other engine settings you changed. If you use a different token file, include its token IDs so others can repeat the run.

## What the table measures

- The benchmark cycles through the supplied raw token IDs without applying a chat template. It alternates the first token across requests and checks that the engine reused zero prompt tokens. The benchmark runs one warmup request, then measures one request at a time with one cache slot. Context capacity is the number of positions reported by the engine. Prompt length is the number of positions filled by the benchmark input.
- Prefill starts after the engine checks the prefix cache and ends when it has processed the prompt and prepared for decoding. Divide the prompt length by that interval to get prefill throughput.
- Decode runs from the end of prefill until the generation loop completes. Generated includes the first token predicted from the prompt. Decode throughput counts the remaining tokens, so a run with 129 generated tokens counts 128 for decode throughput.
- Both intervals use engine wall-clock time. Model loading and warmup occur outside the reported measurements. Each row represents one repeat. Compare rows at the same prompt length, context capacity, token limit and engine settings.

A nonzero exit means the run failed. Use a complete run for the report. The tool rejects cached prefixes, incomplete generations and invalid timings. Without `--report`, the benchmark emits its usual JSONL output.
