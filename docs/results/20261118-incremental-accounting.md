# Incremental checkpoint accounting review

PR #5 remains pending its integrated GPU check. Lack of an inference speedup is
not a rejection criterion: bounded bookkeeping cost and useful correctness
diagnostics are benefits in their own right. Neither implies a proven reduction
in production failures.

## Change and tradeoff

`PagePool` tracks checkpoint references separately from total references. Its
checkpoint-only page count changes when checkpoint, active, parked or temporary
owners change, including page relocation. Admission/stat queries no longer
rebuild this count by scanning all retained checkpoint page lists. Other cache
operations, such as prefix lookup and eviction selection, still perform scans.

The cost is one additional 32-bit counter per device page, a total counter, and
updates at ownership changes. `Q27_CHECK_CACHE_REFS=1` compares these counters
against an independent full scan; it is a diagnostic option, disabled by default.
GPU/host budgets, output math and eviction policy are unchanged.

## Current integration evidence

Candidate `73716dabade39a6c1abbe2d0e52fc27b680505ae` merges main
`3bd5980dfed3996f2afe2c22a9aa5d5bc932e349` into the original accounting change.
Both the accounting oracle and the protected-prefix policy are retained.

- Full CPU suite with address/undefined-behavior sanitizers: **481 passed,
  16 skipped**. The randomized ownership test checks 12,000 mutations for each
  of the low-page and high-page allocation orders.
- [CPU/updater CI](https://github.com/myalcin81/qwen3-27b-5090/actions/runs/44313746403)
  and [CUDA CI](https://github.com/myalcin81/qwen3-27b-5090/actions/runs/44313746219)
  passed. A separate native Release build on aibench also completed.
- The earlier standalone candidate `8149f5a` passed seven GPU fixtures with
  byte/token parity and zero ownership-oracle mismatches, as recorded by the
  external test agent. Those results do not substitute for the integrated run.
- The integrated test launch at 16:28 TRT and five-minute retry ending at
  16:34 TRT found active production requests. Both stopped at preflight;
  **no GPU fixture or A/B arm ran, and production was not stopped**. Its health
  endpoint remained successful. This is a scheduling blocker, not a failed
  engine-correctness test.

## Why the earlier performance verdict was revised

The earlier three-run comparison reported mean four-agent decode of
108.87 versus 104.8 tok/s, and one-agent decode of 239.93 versus 227.7 tok/s.
These observations remain evidence, but do not establish a causal regression.
The user's acceptance criteria explicitly include scalability and diagnostics.

Independent inspection also found limitations in the earlier `ajan4.py` driver:

- Its warm throughput numerator includes responses completed before the warm
  timing window starts. Raw initial runs contain one such response in A0/A1,
  and two in B5A. This invalidates that aggregate throughput calculation; it
  does not itself explain the separately calculated per-agent decode rates.
- The one-agent phase reuses the preceding four-agent phase's engine/cache.
- It does not reject HTTP errors, SSE error events, or missing terminal events.
  Some initial TT-17 pressure records have zero output while reporting no error.
  This does not automatically invalidate the later TT-19 cache-retention result.

The prepared independent driver uses identical frozen growing histories,
temperature 0, fixed seeds, 1,024 output-token caps and three request waves with
barriers. Every A/B/A arm and concurrency level starts a fresh engine. It checks
HTTP status, SSE errors, terminal completion, nonempty output and usage; only
request/output hashes and measurements are stored, not private prompt text.
Its self-checks cover incomplete/error/zero-output streams, HTTP 400 rejection,
and output hashes independent of streaming chunk boundaries.

## Remaining acceptance

Run the seven integrated GPU fixtures with `Q27_CHECK_CACHE_REFS=1` and
`KV_CACHE_POLICY=protected`, then the controlled one-/four-agent comparison with
the oracle disabled. Preserve state32/host68 GiB, the 674,816-token device pool
and 262,144-token context limit. Check exact fixture parity, errors, RAM headroom
and any repeatable material regression, rather than requiring a speed increase.

TT-20 in `/home/researcher/mkyt-study/GERI-BILDIRIM.md` coordinates this work.
The ready runner and strict replay driver are under `/home/researcher/evals/`
as `codex-run-5a-independent-20261118.py` and
`codex-fixed-trace-20261118.py`; the native build is in
`codex-5a-build-20261118/`. Acquire a new idle test window before running them.
The runner restores the original production engine settings in `finally` if
it has stopped that engine. No production deployment is claimed here.
