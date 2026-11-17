"""GPU acceptance: two sessions retain their retry and continuation checkpoints in five state slots.

Run only in a coordinated idle GPU window, with already installed weights. No downloads or service changes.
This small deterministic fixture exercises state pressure and parked prefill metadata before the long replay.
"""

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine/python"), str(ROOT / "reference")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="gdnfp4")
    ap.add_argument("--draft", choices=["dspine", "dflash2"], default="dspine")
    ap.add_argument("--kv-dtype", choices=["nvfp4", "fp8", "bf16"], default="nvfp4")
    ap.add_argument("--host-gib", choices=["0", "1"], default="1")
    ap.add_argument("--park-prefill", action="store_true")
    args = ap.parse_args()
    fixture = dict(STATE_HOST_CACHE="1", KV_HOST_CACHE=args.host_gib, KV_POOL_TOKENS="8192",
                   CHECKPOINT_INTERVAL="512", MAX_SEQS="2" if args.park_prefill else "1", MAX_TOKENS_FIT="0")
    os.environ.update(fixture)
    print("fixture:", fixture, "prefill_chunk=256", flush=True)
    print("runtime:", {k: os.environ.get(k, "default") for k in
                       ("Q27_SKINNY_EXACT", "Q27_ATTN_TC", "Q27_SPEC_PATHS", "Q27_PAGES_HIGH")}, flush=True)
    from q27lib import Engine
    from qwen27b.paths import draft_snapshot, engine_target
    from host_kv_test import fingerprint

    prompts = [json.loads(line)["ids"] for line in
               (ROOT / "bench/golden/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    base = max(prompts, key=len)
    assert len(base) >= 2304
    ids = [list(base[:2048]), list(base[:2048])]
    # One common boundary, then two distinct chains; nonce values are ordinary valid tokens.
    ids[0][128], ids[1][128] = 1000, 1001
    tail = base[2048:2304]
    for spec in (False, True):
        eng = Engine(engine_target(args.weights), max_tokens=4096, prefill_chunk=256,
                     draft=draft_snapshot(args.draft), kv_dtype=args.kv_dtype)

        def generate(rest, count=16):
            if spec:
                out = [eng.spec_begin(rest, temperature=0, top_k=1, max_new=count, ignore_eos=True)]
                while len(out) < count:
                    out.extend(eng.spec_step())
            else:
                out = [eng.forward_sample(rest, temperature=0, top_k=1)]
                while len(out) < count:
                    out.append(eng.forward_sample([out[-1]], temperature=0, top_k=1))
            assert len(out) == count
            return out

        try:
            # 1 GiB holds five 189.9 MiB state slots (draft loaded in both modes), never silently run a larger test.
            while eng.cache_grow(1):
                pass
            assert eng.cache_stats()["slots_allocated"] == 5, eng.cache_stats()
            records = []
            for i, prompt in enumerate(ids):
                cached = eng.prefix_begin(prompt, at=(128, 1792), spec=spec)
                assert cached == (0 if i == 0 else 128), ("shared boundary", i, cached)
                fed = cached
                if args.park_prefill:
                    fed += eng.prefill(prompt[fed:], budget=512, spec=spec)
                    assert 0 < fed < 1792
                    area = eng.seq_park()
                    eng.seq_unpark(area)  # future boundary 1792 must travel with the parked sequence
                out = generate(prompt[fed:])
                eng.checkpoint()
                pos = eng.pos
                assert pos == len(prompt) + len(out) - 1
                records.append((out, pos, fingerprint(eng, pos, args.kv_dtype)))
            assert eng.cache_stats()["evicted_state"] > 0, ("state pressure was not exercised", eng.cache_stats())
            for _ in range(2):
                for prompt, (expected, pos, expected_bytes) in zip(ids, records):
                    cached = eng.prefix_begin(prompt, at=(128, 1792), spec=spec)
                    assert cached == 1792, ("lost retry boundary", cached, eng.cache_stats())
                    got = generate(prompt[cached:])
                    assert got == expected, ("retry tokens changed", spec, got, expected)
                    eng.checkpoint()
                    # A next-turn hit restores the partial-page continuation, with byte-identical KV/GDN.
                    next_turn = prompt + expected[:-1] + tail
                    cached = eng.prefix_begin(next_turn, spec=spec)
                    assert cached == pos, ("lost continuation", cached, pos, eng.cache_stats())
                    assert fingerprint(eng, pos, args.kv_dtype) == expected_bytes, "restored KV/GDN bytes differ"
                    assert eng.cache_stats()["checkpoints"] <= 5
            print("PASS", "spec" if spec else "decode", eng.cache_stats(), flush=True)
        finally:
            eng.close()
    print("All state-pressure retry/continuation cases passed.", flush=True)


if __name__ == "__main__":
    main()
