"""Forced GPU eviction: raw KV/GDN byte parity and greedy continuation after host restore.

Run only in a coordinated, idle GPU window with already installed weights. No downloads or service changes.
The small GPU pool forces demotion; --park also exercises parking-area relocation while a sequence is held.
This is a correctness test, not the representative TTFT/decode acceptance benchmark.
"""

import argparse
import ctypes
import hashlib
import json
import math
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
    ap.add_argument("--park", action="store_true", help="hold a parked sequence through pressure and restore")
    args = ap.parse_args()
    # Explicit fixture settings, printed below. Runtime math policies are inherited without modification.
    os.environ.update(STATE_HOST_CACHE="4", KV_HOST_CACHE="1", KV_POOL_TOKENS="4096", CHECKPOINT_INTERVAL="0",
                      MAX_SEQS="2" if args.park else "1", MAX_TOKENS_FIT="0")
    print("fixture:", {k: os.environ[k] for k in ("STATE_HOST_CACHE", "KV_HOST_CACHE", "KV_POOL_TOKENS",
                                                "CHECKPOINT_INTERVAL", "MAX_SEQS", "MAX_TOKENS_FIT")}, flush=True)
    print("runtime:", {k: os.environ.get(k, "default") for k in
                       ("Q27_SKINNY_EXACT", "Q27_ATTN_TC", "Q27_SPEC_PATHS", "Q27_PAGES_HIGH")}, flush=True)
    from q27lib import Engine, lib
    from qwen27b.paths import draft_snapshot, engine_target

    prompts = [json.loads(line)["ids"] for line in (ROOT / "bench/golden/prompts.jsonl").read_text().splitlines()]
    base = max(prompts, key=len)[:2048]
    tail = max(prompts, key=len)[2048:2304]
    assert len(base) == 2048 and len(tail) == 256
    eng = Engine(engine_target(args.weights), max_tokens=4096, draft=draft_snapshot(args.draft), kv_dtype=args.kv_dtype)
    nonce = 1000

    def unique_prompt():
        nonlocal nonce
        p = list(base)
        p[0] = nonce  # a valid ordinary token, making each pressure prompt a distinct checkpoint key
        nonce += 1
        return p

    def run(ids, spec, count):
        if spec:
            out = [eng.spec_begin(ids, temperature=0, top_k=1, max_new=count, ignore_eos=True)]
            while len(out) < count:
                out.extend(eng.spec_step())
        else:
            out = [eng.forward_sample(ids, temperature=0, top_k=1)]
            while len(out) < count:
                out.append(eng.forward_sample([out[-1]], temperature=0, top_k=1))
        assert len(out) == count
        return out

    def fingerprint(n):
        # Every valid KV row in all 16 attention layers, including NVFP4 scales, and all 48 FP32 GDN states.
        row_bytes = {"nvfp4": 144, "fp8": 256, "bf16": 512}[args.kv_dtype]
        kv = ctypes.create_string_buffer(2 * n * 4 * row_bytes)
        state = ctypes.create_string_buffer(48 * 128 * 128 * 4)
        hashes = []
        for layer in range(16):
            eng._check(lib().q27_debug_kv(eng.h, layer, 0, n, kv), "debug_kv")
            hashes.append(hashlib.sha256(kv.raw).hexdigest())
        for layer in range(48):
            eng._check(lib().q27_debug_ssm(eng.h, layer, state), "debug_ssm")
            hashes.append(hashlib.sha256(state.raw).hexdigest())
        return hashes

    try:
        for spec in (False, True):
            for partial in (False, True):
                label = f"{'spec' if spec else 'decode'}/{'partial' if partial else 'full'}"
                eng.reset()
                prompt = unique_prompt()
                generated = run(prompt, spec, 18 if partial else 1)
                eng.checkpoint()
                pos = eng.pos
                assert pos == len(prompt) + len(generated) - 1
                assert bool(pos % 64) == partial
                expected_bytes = fingerprint(pos)
                request = prompt + generated[:-1] + tail
                eng.reset()

                parked, parked_bytes, parked_pos = None, None, 0
                if args.park:
                    run(unique_prompt()[:256], spec, 1)
                    parked_pos = eng.pos
                    parked_bytes = fingerprint(parked_pos)
                    parked = eng.seq_park()

                before = eng.cache_stats()
                # More distinct pages than the entire device pool, even when earlier test checkpoints exist.
                pressure_count = math.ceil(before["pool_pages"] * 64 / len(base)) + 2
                for _ in range(pressure_count):
                    eng.reset()
                    run(unique_prompt(), spec, 1)
                    eng.checkpoint()
                st = eng.cache_stats()
                assert st["kv_demoted_pages"] > before["kv_demoted_pages"], (label, "no demotion", st)
                assert 0 < st["kv_host_pages"] and st["kv_host_bytes"] <= st["kv_host_budget_bytes"]
                restored_before = st["kv_restored_pages"]
                cached = eng.prefix_begin(request, spec=spec)
                assert cached == pos, (label, "checkpoint lost under GPU pressure", cached, pos, st)
                assert eng.cache_stats()["kv_restored_pages"] > restored_before, (label, "no host restore")
                assert fingerprint(pos) == expected_bytes, (label, "KV/GDN bytes differ after restore")
                got = run(request[pos:], spec, 32)

                # Replay exactly the same prefill/decode chunk boundaries without consulting the cache.
                eng.reset()
                assert run(prompt, spec, len(generated)) == generated
                ref = run(request[pos:], spec, 32)
                assert got == ref, (label, "greedy continuation differs", got, ref)
                if parked is not None:
                    eng.reset()
                    eng.seq_unpark(parked)
                    assert fingerprint(parked_pos) == parked_bytes, (label, "parked state/KV changed")
                    eng.reset()
                print("PASS", label, eng.cache_stats(), flush=True)
        print("All forced-eviction byte/token parity cases passed.", flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    main()
