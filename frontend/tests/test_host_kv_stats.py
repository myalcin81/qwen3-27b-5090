"""The host KV counters use a new entry point; an older engine still receives exactly twelve int64s."""

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "engine/python"))
import q27lib


def engine():
    e = object.__new__(q27lib.Engine)
    e.h = None
    return e


def old_stats(handle, out):
    assert len(out) == 12
    for i in range(12):
        out[i] = i
    return 0


def test_old_engine_keeps_safe_stats_buffer(monkeypatch):
    monkeypatch.setattr(q27lib, "lib", lambda: SimpleNamespace(q27_cache_stats=old_stats))
    st = engine().cache_stats(inventory=True)
    assert st["slots_allocated"] == 11
    assert st["kv_host_bytes"] == st["kv_host_pages"] == st["evicted_state"] == 0
    assert "prefix_unique_tokens" not in st  # unavailable measurements must not masquerade as zero


def test_new_engine_adds_host_counters(monkeypatch):
    def host_stats(handle, out):
        assert len(out) == 9
        for i in range(9):
            out[i] = 100 + i
        return 0
    monkeypatch.setattr(q27lib, "lib", lambda: SimpleNamespace(q27_cache_stats=old_stats, q27_kv_host_stats=host_stats))
    st = engine().cache_stats()
    assert st["slots_allocated"] == 11
    assert st["kv_host_pages"] == 100 and st["kv_restored_pages"] == 104
    assert st["evicted_state"] == 105 and st["kv_host_alloc_failures"] == 108


def test_inventory_uses_separate_buffer(monkeypatch):
    calls = []
    def inventory(handle, out):
        calls.append(True)
        assert len(out) == 10
        for i in range(10):
            out[i] = 200 + i
        return 0
    monkeypatch.setattr(q27lib, "lib", lambda: SimpleNamespace(q27_cache_stats=old_stats,
                                                              q27_cache_inventory=inventory))
    assert "prefix_unique_tokens" not in engine().cache_stats()
    assert not calls  # admission queries must not scan retained token histories
    st = engine().cache_stats(inventory=True)
    assert len(calls) == 1
    assert st["prefix_unique_tokens"] == 200 and st["prefix_duplicate_full_pages"] == 203
    assert st["kv_host_unreferenced_pages"] == 207 and st["kv_host_used_bytes"] == 208
    assert st["prefix_leaves"] == 209 and st["slots_allocated"] == 11
