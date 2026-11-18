#!/usr/bin/env python3
"""CPU-only exact-ID and timing comparison on a local model tokenizer; no weights/network/GPU required.

python scripts/bench_tokenizer_cache.py /path/to/model --tokens 150000 --rounds 8
Synthetic text only; output contains aggregate measurements, never request text or token IDs.
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "frontend"))
from tokenizer import Tokenizer  # noqa: E402


def run(path, tokens, rounds, budget):
    os.environ["TOKENIZER_CACHE_MIB"] = "0"
    reference = Tokenizer(path)
    seed = "Review this example: def square(x): return x * x. Türkçe 日本語.\n"
    count = max(1, tokens // len(reference.encode(seed)))
    body = seed * count
    report = {"model": str(path), "target_tokens": tokens, "rounds": rounds, "cache_mib": budget, "cases": {}}
    for name, suffix in [("plain", ""), ("escaped", "\nQuoted text: <|im_start|> <|image_pad|>")]:
        head = [{"role": "user", "content": body + suffix}]
        sequence = [head]
        for turn in range(rounds - 1):
            sequence.append(sequence[-1] + [{"role": "assistant", "content": f"Result {turn}."},
                                             {"role": "user", "content": "Check the next example."}])
        for access, requests in [("identical", [head] * rounds), ("growing", sequence)]:
            expected = [reference.prompt_ids(messages) for messages in requests]
            runs = []
            # A/B/A distinguishes cache gains from warming or transient CPU load.
            for mib in [0, budget, 0]:
                os.environ["TOKENIZER_CACHE_MIB"] = str(mib)
                tok = Tokenizer(path)
                times = []
                for messages, ids in zip(requests, expected):
                    start = time.perf_counter()
                    actual = tok.prompt_ids(messages)
                    times.append((time.perf_counter() - start) * 1000)
                    if actual != ids:
                        raise AssertionError(f"Token IDs changed: {name}/{access}, cache={mib}")
                runs.append({"cache_mib": mib, "cold_ms": times[0], "repeat_median_ms": statistics.median(times[1:]),
                             "repeat_max_ms": max(times[1:]), "total_ms": sum(times),
                             "cache": tok.encoding_cache_stats()})
            report["cases"][f"{name}/{access}"] = {"initial_tokens": len(expected[0]), "ids_equal": True, "runs": runs}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=150000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=32)
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 2 or args.cache_mib < 1:
        parser.error("tokens/cache-mib must be positive and rounds must be at least 2")
    print(json.dumps(run(args.model, args.tokens, args.rounds, args.cache_mib), indent=2))
