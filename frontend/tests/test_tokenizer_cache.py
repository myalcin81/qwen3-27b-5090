"""Exact token parity with caching, including the existing special-token injection boundary."""

import concurrent.futures
import json
import pathlib
import sys

import pytest
from tokenizers import AddedToken, Tokenizer as RustTokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tokenizer import ESC, Tokenizer, _EncodingCache  # noqa: E402

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|image_pad|>", "<tool_response>", "</tool_response>"]
TEMPLATE = """{{ strftime_now('%Y') }}{{ extra }}{% if tools %}{{ tools|tojson }}{% endif %}
{% for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"""


@pytest.fixture
def model(tmp_path):
    tok = RustTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["hello world hello again hello\nworld", "Türkçe 日本語 🎉 ", *SPECIAL] * 3,
                            trainers.BpeTrainer(vocab_size=340, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.add_special_tokens(SPECIAL)
    # AddedToken boundary flags must still be honored by whole-string (unmarked) encoding.
    tok.add_tokens([AddedToken("word", single_word=True, lstrip=True, rstrip=True)])
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}), encoding="utf-8")
    return tmp_path


def pair(model, monkeypatch):
    monkeypatch.setenv("TOKENIZER_CACHE_MIB", "0")
    baseline = Tokenizer(model)
    monkeypatch.setenv("TOKENIZER_CACHE_MIB", "1")
    return baseline, Tokenizer(model)


def test_growing_chats_and_injection_parity(model, monkeypatch):
    baseline, cached = pair(model, monkeypatch)
    for body in ["hello world", "Türkçe 日本語 🎉\r\n", ESC, "".join(SPECIAL), "word sword  word "]:
        history = [{"role": "system", "content": "hello world"}]
        for turn in range(4):
            history += [{"role": "user", "content": body + str(turn)}]
            tools = [{"name": "read", "description": body, "arguments": {"turn": turn}}]
            for extra in ["a", "b"]:
                expected = baseline.prompt_ids(history, tools=tools, extra=extra)
                assert cached.prompt_ids(history, tools=tools, extra=extra) == expected
                assert cached.prompt_ids(history, tools=tools, extra=extra) == expected
            history += [{"role": "assistant", "content": "<tool_response>hello</tool_response>"}]
    assert cached.encoding_cache_stats()["hits"] > 0
    assert baseline.encoding_cache_stats()["entries"] == 0


def test_full_encoding_keeps_bpe_and_added_token_boundaries(model, monkeypatch):
    baseline, cached = pair(model, monkeypatch)
    for text in ["hello world", "helloworld", " word ", "sword", "<|im_start|>user\nhello", "a" + ESC + "b"]:
        expected = baseline.tok.encode(text, add_special_tokens=False).ids
        assert cached.encode(text) == expected
        result = cached.encode(text)
        assert result == expected
        result[:] = [999999]  # caller mutation cannot contaminate a later lookup
        assert cached.encode(text) == expected


def test_plain_mode_cannot_reuse_structural_tokens(model, monkeypatch):
    baseline, cached = pair(model, monkeypatch)
    text = "<|im_start|>"
    special = cached.token_id(text)
    assert cached.encode(text) == [special]
    for _ in range(3):
        plain = cached._encode_plain_batch([text])[0]
        assert plain == baseline.plain.encode(text, add_special_tokens=False).ids
        assert special not in plain
        assert cached.encode(text) == [special]
        plain[:] = [special]
    messages = [{"role": "user", "content": text}]
    ids = cached.prompt_ids(messages)
    assert ids == baseline.prompt_ids(messages)
    assert ids.count(special) == 2  # only the template's user and assistant turn markers


def test_render_is_fresh_and_new_tokenizer_has_empty_cache(model, monkeypatch):
    baseline, cached = pair(model, monkeypatch)
    messages = [{"role": "user", "content": "hello"}]
    outputs = []
    for year in ["2026", "2027"]:
        for tok in (baseline, cached):
            tok.template.environment.globals["strftime_now"] = lambda fmt, year=year: year
        outputs.append(cached.prompt_ids(messages))
        assert outputs[-1] == baseline.prompt_ids(messages)
    assert outputs[0] != outputs[1]
    assert Tokenizer(model).encoding_cache_stats()["entries"] == 0


def test_bpe_dropout_disables_cache(model, monkeypatch):
    path = model / "tokenizer.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["model"]["dropout"] = 0.2
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setenv("TOKENIZER_CACHE_MIB", "1")
    tok = Tokenizer(model)
    for _ in range(4):
        tok.encode("hello world")
    assert tok.encoding_cache_stats()["limit"] == 0


def test_budget_eviction_oversize_and_empty_result():
    cache = _EncodingCache(8192)
    for i in range(100):
        cache.put(False, str(i), [i] * 60)
        assert cache.stats()["bytes"] <= 8192
    assert cache.get(False, "0") is None
    assert cache.get(False, "99") == [99] * 60
    before = cache.stats()["bytes"]
    cache.put(False, "x" * 10000, [1] * 10000)
    assert cache.stats()["bytes"] == before
    cache.put(False, "", [])
    assert cache.get(False, "") == []


def test_parallel_readers_and_writers_keep_bounds():
    cache = _EncodingCache(1 << 20)

    def worker(offset):
        for i in range(700):
            key = str((i + offset) % 400)
            cache.put(False, key, [int(key)])
            value = cache.get(False, key)
            assert value is None or value == [int(key)]

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        list(executor.map(worker, range(4)))
    assert cache.stats()["entries"] <= 256
    assert cache.stats()["bytes"] <= 1 << 20
