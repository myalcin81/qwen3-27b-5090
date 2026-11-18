# Opt-in tokenizer encoding cache

The cache is available with `TOKENIZER_CACHE_MIB=32`, but defaults to **0** in
both direct server startup and Compose. It retains exact text encodings within
an estimated CPU-memory budget and preserves ordinary versus escaped-text token
semantics. It neither caches arbitrary token prefixes nor reserves GPU/pinned
memory. With the default disabled, encoding bypasses cache lookup and storage.

Repeated identical text has a measurable tokenizer-only benefit; existing
synthetic measurements reduced approximately 98 ms to 1 ms on a repeat. Growing
plain conversations usually encode a different complete prompt every turn, so
they receive no such benefit. This is a workload-specific option, not a general
inference-speed or reliability claim.

## Traffic replay (TT-21, 2026-11-18)

The external test agent replayed the last 3,000 reconstructed Anthropic requests
in chronological order from 2026-11-14 22:44:39 through 2026-11-18 03:03:01,
using the local model/template and candidate `2eeb427`. Each arm made one pass:
cache0 first, then cache32; there was no artificial second warm pass. Codex read
the resulting JSON and inspected the replay code.

| Measurement | Cache disabled | Cache 32 MiB |
| --- | ---: | ---: |
| Render/tokenize median, ms | 37.4 | 36.0 |
| Render/tokenize p90, ms | 73.0 | 70.6 |
| Maximum, ms | 99.5 | 107.7 |
| Total process CPU time, s | 120.6 | 115.8 |

Cache32 recorded 58 hits and 2,942 misses: approximately **1.9% call hits** and
**1.5% text-byte hits**, with estimated peak retention 32 MiB and 2,896 evictions.
The replay reported 59 repeated token sequences; 58 reused cached encodings.
The template path made one encode call per reconstructed request. All 3,000
baseline/candidate token-sequence hashes matched.

Limitations matter: packed records flatten content blocks and reconstruct tool
schemas from tool names, so this is not an exact replay of the original HTTP
payloads. It measures repetition in reconstructed traffic. The script compares
Python tuple hashes of token IDs, not each ID directly or a cryptographic hash.
Its single fixed-order timing comparison cannot establish a causal CPU speedup.
Exact ID parity, escaped-token boundaries, caller mutation, dropout behavior,
eviction limits and concurrent access are separately covered by automated tests.

The low observed hit rate does not justify enabling 32 MiB for every server.
The implementation remains useful when exact retries/repeated prompts dominate,
and can be enabled explicitly without changing the default path. Segment-level
caching would need a separate design preserving BPE and special-token boundaries;
it is not part of this change.

## Final integration validation

The default-off candidate was integrated with main `539231d` (protected-prefix
retention and incremental checkpoint accounting), yielding runtime `45b6d1d`.
The full CPU suite with address/undefined-behavior sanitizers passed **489 tests,
16 skipped**, including a check that unset configuration bypasses cache lookup
and storage for both ordinary and escaped-client encoding.

Codex additionally compared 100 reconstructed local traffic requests containing
4,073,226 input tokens on the actual model tokenizer. Explicit0, unset and
explicit32 produced **byte-identical packed token-ID arrays for every request**;
the enabled mode also returned identical IDs on immediate repeats. Explicit0
and unset retained zero entries/bytes and made zero cache lookups. Enabled32
stayed below its 33,554,432-byte budget (33,256,114 bytes observed). These
deliberate repeats test correctness, not a real-traffic hit rate. The test is
CPU-only and does not restart or load a GPU engine.

The exact-comparison artifact is
`/home/researcher/evals/codex-5b-exact-parity-20261118.json`; the tested
`frontend/tokenizer.py` SHA-256 is
`3af719d01cbfe56e377456ecef9e82460127b023f26191baa35eec431373f6df`.

Evidence remains on aibench in `/home/researcher/evals/tt21/sonuc-3000.json`,
`/home/researcher/tools/tt21.py` and TT-21 of
`/home/researcher/mkyt-study/GERI-BILDIRIM.md`. Private traffic is not included.
No production cache setting is enabled by this report or by merging the code.
