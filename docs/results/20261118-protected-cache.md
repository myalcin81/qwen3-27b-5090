# Protected prefix-cache acceptance on aibench

Accepted for the tested workload on 2026-11-18. This report covers PR #7's
`KV_CACHE_POLICY=protected`, tested at source commit
`eea2096b46feba286f701c0dc94e43c85968160e`. The default remains `lru`;
merging the code does not enable the policy in an existing deployment.

## Configuration and correctness

- RTX 5090; existing local Qwen model/template and DSpine configuration.
- Context limit 262,144 tokens; GPU KV pool 674,816 tokens.
- Host KV 68 GiB; state cache 32 GiB. Both baseline and candidate used the same budgets.
- CPU/sanitizer suite: 479 passed, 16 skipped; CPU and CUDA CI passed.
- Tested `build-w3/libq27.so` SHA-256:
  `01e130ec2d7659cd2ecde46780200d38f165f87a3eb47b474d7b787344fce33a`.
- GPU host-KV byte/token parity, parked/high-page allocation, state-pressure,
  prefix-cache and parking fixtures passed. No HTTP, engine, OOM or host-allocation
  errors were reported in the acceptance runs.

## Actual host-capacity pressure (TT-19)

The earlier TT-17 trial produced zero host evictions and did not test host overflow.
TT-19 used a fresh engine per arm, four warm sessions of approximately 146–152k
tokens, two turns with seed 1, then 80 distinct cold-prefix requests containing
6,149,012 prompt tokens (concurrency 2, at most 16 output tokens). Finally, it
probed the original four warm-session prompts.

Every TT-19 arm reached the host budget of 73,014,312,960 allocated bytes,
64–73 host evictions, and all 172 state slots allocated. Minimum available RAM
across the suite was 13.26–14.13 GiB, above the 12 GiB guard; no restarts/errors.
The matching token/context limits were verified at every start.

| Measurement | LRU before | Protected | LRU after |
| --- | --- | --- | --- |
| Four warm-probe TTFTs, seconds | 17.5 / 34.8 / 52.4 / 70.2 | **0.42 / 0.53 / 0.76 / 0.98** | 17.9 / 35.2 / 52.7 / 70.3 |
| Cached tokens per warm probe | 640–4,736 | **150,912–153,024** | 448–4,736 |
| Cold-stream duration, seconds | 588 | 590 | 590 |

The reused warm prefixes survived the cold scan with protection. These are
end-to-end first-token measurements, not isolated copy times. The result supports
protection for a reused hot set that fits its bounded protected capacity; it does
not imply every workload or an oversized hot set will achieve these latencies.

## Normal-load regression repeat

The initial single comparison showed a possible approximately 4% decode loss,
so it was not accepted. Three interleaved main/protected repetitions then used
the same profile, seed 1, four agents for six turns and one agent for six turns.

| Measurement | Main / LRU | Protected |
| --- | --- | --- |
| Four-agent median decode per run, tok/s | 105.95 / 109.15 / 103.20 | 107.35 / 112.45 / 110.05 |
| Four-agent warm TTFT medians, seconds | 8.7 / 9.6 / 7.8 | 8.5 / 8.1 / 8.5 |
| Mean warm turn duration, seconds | 28.16 | 27.35 |
| One-agent median decode, tok/s | 231.8 in all three runs | 231.8 in all three runs |
| One-agent turn duration, seconds | 9.11 | 9.12 |

The repeat passed the agreed no-repeated-regression-above-3% criterion, with
zero errors/restarts. This establishes the regression check for this profile;
the principal demonstrated benefit is warm-prefix retention under overflow.

## Evidence and deployment boundary

The external test agent recorded TT-19 at 10:57 and the normal-load repeat at
11:29 (Europe/Istanbul) in
`/home/researcher/mkyt-study/GERI-BILDIRIM.md`. Detailed artifacts remain under
`/home/researcher/evals/tt19/` and the corresponding repeat-run directory.
Private prompt bodies are not included in this repository.

This acceptance covers only protected-prefix retention. Incremental reference
accounting, tokenizer caching and asynchronous KV demotion are separate candidates
with separate decisions. Deployment requires enabling `KV_CACHE_POLICY=protected`
and a normal engine restart; this report does not claim that deployment occurred.
