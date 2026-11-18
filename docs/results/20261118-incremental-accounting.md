# Incremental checkpoint accounting review

PR #5 is accepted after integrated GPU validation on 2026-11-18. The acceptance
reason is bounded bookkeeping cost and useful correctness diagnostics; an
inference speedup is not required or established. Neither benefit implies a
proven reduction in production failures.

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
- The initial integrated test launch at 16:28 TRT and five-minute retry ending at
  16:34 TRT found active production requests. Both stopped at preflight;
  **no GPU fixture or A/B arm ran, and production was not stopped**. Its health
  endpoint remained successful. This was a scheduling blocker, not a failed
  engine-correctness test. The later run below completed in an idle window.

## Integrated GPU acceptance (TT-20)

The external agent launched the prepared independent runner at 16:43 TRT and
reported completion at 16:53. Codex then independently checked the actual
`result.json`, per-request hashes, fixture logs and source/library identities.
The successful directory is
`/home/researcher/evals/codex-5a-independent-20261118-164359/`;
the older `-r1` directory still contains the blocked preflight attempt.

All seven integrated fixtures passed with `Q27_CHECK_CACHE_REFS=1` and
`KV_CACHE_POLICY=protected`: host-KV restore, parked/high-page restore, state
pressure with host KV off/on, parked prefill, prefix cache, and parking. They
verify exact KV/GDN bytes and greedy tokens across pressure/restoration, with
no ownership-oracle mismatch. The candidate library SHA-256 was
`2d8eb81cf93630c511c95e0942d0a19cf6bff49f278cf8b14f666f744208e470`.

The six fresh-engine A/B/A arms used the same private frozen growing histories,
temperature 0, fixed seeds, three waves and at most 1,024 output tokens. A was
`eea2096` (same runtime tree as main `3bd5980`); B was `73716da`. State32/host68
GiB, context262144 and GPU pool674816 were identical. The oracle was disabled
for timing. All 45 requests completed without HTTP/SSE/terminal errors, and
request hashes and semantic output hashes matched across all three arms at each
concurrency level.

| Measurement | A before | B: incremental | A after |
| --- | ---: | ---: | ---: |
| Four-agent warm decode median, tok/s | 191.06 | 195.04 | 212.67 |
| Four-agent warm TTFT median, s | 3.17 | 3.30 | 3.30 |
| Four-agent warm turn median, s | 9.55 | 8.96 | 9.22 |
| Four-agent warm aggregate throughput, tok/s | 400.71 | 377.13 | 375.97 |
| One-agent warm decode median, tok/s | 638.21 | 638.32 | 638.26 |
| One-agent warm TTFT median, s | 1.290 | 1.290 | 1.287 |

The four-agent timing variation is substantial: B's decode rate falls between
the two baseline observations. This short run establishes neither a general
speedup nor statistical equivalence, and does not demonstrate a repeatable
regression. The one-agent rates and output hashes are stable. Acceptance rests
on correctness plus the scalability/diagnostics benefit, not a speed claim.

Minimum available RAM was 98.05 GiB: this short normal-load run did not fill the
lazily allocated 68 GiB host cache. It must not be described as a full-host
pressure test; forced pressure is covered separately by the small GPU fixtures.
The runner recorded successful restoration of the original production engine
settings and a healthy endpoint.

The bookkeeping microbenchmark also returned equal answers for 2,000 queries
over 419,680 checkpoint references: full scan183.368 ms, incremental0.008 ms.
This measures only CPU bookkeeping, not inference throughput.

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

## Evidence and deployment boundary

TT-20 in `/home/researcher/mkyt-study/GERI-BILDIRIM.md` records this work.
The runner and strict replay driver are under `/home/researcher/evals/`
as `codex-run-5a-independent-20261118.py` and
`codex-fixed-trace-20261118.py`; the native build is in
`codex-5a-build-20261118/`. The replay driver SHA-256 was
`94045d9dcc1d0b1c55a2fd2255988c2983656392a3985fd6a49184c9799b586d`.
Acquire a new idle test window before repeating them.
The runner restores the original production engine settings in `finally` if
it has stopped that engine. This report accepts the accounting change for merge;
it does not claim that the accounting library has been deployed to production.
