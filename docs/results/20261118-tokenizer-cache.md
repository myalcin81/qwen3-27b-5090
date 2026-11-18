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
**1.5% text-byte hits**, with estimated peak retention32 MiB and 2,896 evictions.
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

The low observed hit rate does not justify enabling32 MiB for every server.
The implementation remains useful when exact retries/repeated prompts dominate,
and can be enabled explicitly without changing the default path. Segment-level
caching would need a separate design preserving BPE and special-token boundaries;
it is not part of this change.

Evidence remains on aibench in `/home/researcher/evals/tt21/sonuc-3000.json`,
`/home/researcher/tools/tt21.py` and TT-21 of
`/home/researcher/mkyt-study/GERI-BILDIRIM.md`. Private traffic is not included.
No production cache setting is enabled by this report or by merging the code.
