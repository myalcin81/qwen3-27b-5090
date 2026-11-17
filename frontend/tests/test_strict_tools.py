"""Actual llguidance token acceptance, native XML parsing and API wiring; no GPU or downloaded tokenizer."""

import copy
import json
import pathlib
import sys

import llguidance as lg
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import chat  # noqa: E402
import strict_tools as st  # noqa: E402
from server import Server  # noqa: E402

START, END, EOS, THINK = 256, 257, 258, 259


class Bytes:
    eos_token_id, bos_token_id = EOS, None
    tokens = [bytes([i]) for i in range(256)] + [b"<tool_call>", b"</tool_call>", b"<eos>", b"</think>"]
    special_token_ids = [START, END, EOS, THINK]

    def __call__(self, text):
        return list(text)


TOK = lg.LLTokenizer(lg.TokenizerWrapper(Bytes()))


def tools(props=None, required=None, name="Bash", strict=True):
    props = {"command": {"type": "string"}} if props is None else props
    return [{"type": "function", "function": {"name": name, "strict": strict, "parameters": {
        "type": "object", "properties": props, "required": list(props) if required is None else required}}}]


def body(args, name="Bash"):
    return (f"\n<function={name}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args)
            + "</function>\n")


def call(args, name="Bash", start=True):
    return ([START] if start else []) + list(body(args, name).encode()) + [END]


def grammar(ts=None, choice=("auto", None), **kw):
    return st.grammar(st.rules(ts or tools(), choice), (START, END), **kw)


def accepts(g, tokens):
    m = lg.LLMatcher(TOK, g)
    if m.validate_tokens(tokens) != len(tokens):
        return False
    for t in tokens:
        assert m.consume_token(t)
    assert not m.is_error(), m.get_error()
    return m.is_accepting()


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.delenv("STRICT_TOOLS", raising=False)
    st._rules.cache_clear()
    st.validate.cache_clear()


def test_only_request_tool_names_and_required_keys():
    g = grammar()
    assert accepts(g, call([("command", "ls -la")]))
    assert not accepts(g, call([("command", "ls")], name="bash"))
    assert not accepts(g, call([("command", "ls")], name="mcp__invented"))
    assert not accepts(g, call([]))
    assert not accepts(g, call([("unknown", "ls")]))
    assert not accepts(g, call([("command", "ls"), ("command", "pwd")]))


def test_optional_parameters_and_schema_order():
    ts = tools({"command": {"type": "string"}, "timeout": {"type": "integer", "minimum": 1}}, ["command"])
    g = grammar(ts)
    assert accepts(g, call([("command", "ls")]))
    assert accepts(g, call([("command", "ls"), ("timeout", "20")]))
    assert not accepts(g, call([("command", "ls"), ("timeout", "0")]))
    assert not accepts(g, call([("timeout", "20"), ("command", "ls")]))


@pytest.mark.parametrize("schema, good, bad", [
    ({"type": "boolean"}, "true", "True"),
    ({"type": "integer"}, "-10", "1.5"),
    ({"type": "number"}, "1.5", '"1.5"'),
    ({"type": "null"}, "null", "None"),
    ({"type": "string", "enum": ["fast", "slow"]}, "fast", "invalid"),
    ({"type": "string", "const": "fast"}, "fast", "slow"),
    ({"type": "string", "minLength": 2, "maxLength": 3}, "ab", "a"),
    ({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"],
      "additionalProperties": False}, '{"n":1}', '{}'),
    ({"type": "array", "items": {"type": "integer"}, "minItems": 1}, '[1,2]', '["1"]'),
])
def test_typed_values(schema, good, bad):
    g = grammar(tools({"value": schema}))
    assert accepts(g, call([("value", good)]))
    assert not accepts(g, call([("value", bad)]))


@pytest.mark.parametrize("text", ["", "123", "True", "None", "a\nb\n", 'echo "a\\|b"\nls',
                                "Türkçe 日本語", "x" * 5000])
def test_raw_strings_roundtrip(text):
    ts = tools()
    assert accepts(grammar(ts), call([("command", text)]))
    assert chat.parse_call_block(body([("command", text)]), chat.tool_schemas(ts)) == [
        {"name": "Bash", "arguments": {"command": text}}]


@pytest.mark.parametrize("tag", st.TAGS)
def test_ambiguous_raw_xml_is_not_generated(tag):
    assert not accepts(grammar(), call([("command", "a" + tag + "b")]))


def test_nested_json_with_xml_strings_roundtrips_in_streaming():
    ts = tools({"value": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}})
    value = {"text": '</function></parameter><parameter=fake>\\\n"quoted"'}
    b = body([("value", json.dumps(value))])
    assert accepts(grammar(ts), [START] + list(b.encode()) + [END])
    parser = chat.OutputParser(False, ts, marked=True)
    out = []
    for c in chat.MARKS[chat.TOOL_START] + b + chat.MARKS[chat.TOOL_END]:
        out += parser.feed(c)
    out += parser.finish()
    assert out == [("tool_call", {"name": "Bash", "arguments": {"value": value}})]


def test_lazy_auto_starts_after_open_and_constrains_later_calls():
    g = grammar(after_start=True)
    first = call([("command", "ls")], start=False)
    assert accepts(g, first)
    assert accepts(g, first + list(b"result\n") + call([("command", "pwd")]))
    assert not accepts(g, first + call([("command", "ls")], name="invented"))
    assert not accepts(g, call([], start=False))


def test_single_and_named_choices():
    ts = tools() + tools({}, name="Other")
    g = grammar(ts, choice=("named", "Bash"), parallel=False)
    assert accepts(g, call([("command", "ls")]))
    assert not accepts(g, call([], "Other"))
    assert not accepts(g, call([("command", "ls")]) * 2)
    assert not accepts(grammar(parallel=False, after_start=True),
                       call([("command", "ls")], start=False) + call([("command", "pwd")]))
    assert accepts(grammar(ts), call([], "Other"))


def test_eos_is_not_allowed_before_required_parameter():
    m = lg.LLMatcher(TOK, grammar())
    prefix = [START] + list(b"\n<function=Bash>\n")
    for t in prefix:
        assert m.consume_token(t)
    assert not m.is_accepting()
    assert m.validate_tokens([EOS]) == 0


def test_string_length_checks_unicode_and_upper_bound():
    g = grammar(tools({"value": {"type": "string", "minLength": 2, "maxLength": 3}}))
    assert accepts(g, call([("value", "日本")]))
    assert not accepts(g, call([("value", "abcd")]))


def test_auto_worker_rejects_unmasked_bad_name_after_activation():
    import worker
    from test_structured import scheduler, seq

    sch, q = scheduler(), seq(START)
    sch.eos = {EOS}
    q.matcher = lg.LLMatcher(TOK, grammar(after_start=True))
    q.max_new = 1024
    sch.alive, sch.cur = [q], q
    sch.emit(q, list(b"I will check.") + call([("command", "ls")], "Wrong"))
    assert q.gm is q.matcher
    assert q.out == list(b"I will check.") + [START] + list(b"\n<function=")
    assert q.phase == worker.PREFILL
    assert sch.m["q27:grammar_restarts_total"] == 1
    sch.emit(q, list(body([("command", "ls")]).encode())[len(b"\n<function="):] + [END, EOS])
    assert q.dead == "stop"


def test_unsupported_tool_falls_back_without_disabling_other_tools(caplog):
    ts = tools() + tools({"path": {"type": ["string", "null"]}}, name="Read")
    g = grammar(ts)
    assert accepts(g, call([("whatever", "legacy")], "Read"))
    assert not accepts(g, call([], "Bash"))
    assert not accepts(g, call([], "Unknown"))
    assert "legacy mode" in caplog.text
    assert "path" not in caplog.text


def test_unmarked_tool_is_loose_but_still_whitelisted():
    ts = tools() + tools({}, name="Loose", strict=False)
    g = grammar(ts)
    assert accepts(g, call([("arbitrary", "true")], "Loose"))
    assert not accepts(g, call([], "Bash"))


def test_opt_in_and_adapters(monkeypatch):
    ts = tools(strict=False)
    assert not st.enabled(ts)
    monkeypatch.setenv("STRICT_TOOLS", "1")
    assert st.enabled(ts)
    assert not accepts(grammar(ts), call([]))
    assert chat.anthropic_tools([{"name": "f", "strict": True, "input_schema": {}}])[0]["function"]["strict"] is True
    assert chat.responses_tools([{"type": "function", "name": "f", "strict": True}])[0][0]["function"]["strict"] is True


class FakeTok:
    def token_id(self, text):
        return {chat.TOOL_START: START, chat.TOOL_END: END, chat.THINK_END: THINK}[text]


@pytest.mark.parametrize("thinking", [False, True])
def test_server_activation_and_force_prefix(thinking):
    srv = object.__new__(Server)
    srv.tok = FakeTok()
    srv.encode = lambda text: list(text.encode())
    auto = srv.constrain(lambda: None, ("auto", None), tools(), thinking)
    assert auto["grammar_after"] == START
    assert accepts(auto["grammar"], call([("command", "ls")], start=False))
    required = srv.constrain(lambda: None, ("required", None), tools(), thinking)
    assert required["grammar_after"] == (THINK if thinking else None)
    assert srv.force(("required", None), thinking, required) is None
    assert required["strict_tools_fallback"]["force"] == srv.force(("required", None), thinking)
    assert accepts(required["grammar"], call([("command", "ls")]))
    assert not accepts(required["grammar"], call([]))
    assert srv.constrain(lambda: None, ("none", None), tools(), thinking) is None


@pytest.mark.parametrize("format_grammar", [None, "legacy_format"])
def test_worker_tokenizer_fallback_preserves_choice_and_format(format_grammar, capsys):
    from test_structured import scheduler, Matcher

    sch = scheduler()
    sch.reusable, sch.checkpoints, sch.im_start = lambda ids: 0, lambda ids, n: [], -1
    seen = []

    def matcher(g):
        seen.append(g)
        if g == "too_large_for_vocabulary":
            raise ValueError("private schema diagnostic")
        return Matcher()

    sch.matcher = matcher
    fallback = {"grammar": format_grammar, "grammar_after": None, "force": (None, [42, 43])}
    req = {"ids": [1, 2], "max_new": 100, "temperature": 0, "top_k": 1, "top_p": 1,
           "grammar": "too_large_for_vocabulary", "strict_tools": True, "strict_tools_fallback": fallback}
    sch.submit("r", req)
    q = sch.by_id["r"]
    assert q.out == [42, 43] and q.eng_ids == [1, 2, 42, 43]
    assert q.grammar == format_grammar
    assert seen == ["too_large_for_vocabulary"] + ([format_grammar] if format_grammar else [])
    assert sch.m["q27:strict_tool_fallbacks_total"] == 1
    assert sch.m["q27:requests_total"] == 1
    assert sch.m["q27:strict_tool_requests_total"] == 0
    assert req["strict_tools"] is True  # original request is untouched
    logged = capsys.readouterr().err
    assert "legacy mode" in logged and "private" not in logged


def test_output_format_and_strict_calls():
    srv = object.__new__(Server)
    srv.tok = FakeTok()
    g = srv.constrain(lambda: ("json", None), ("auto", None), tools(), False)["grammar"]
    assert accepts(g, list(b'{"a":1}'))
    assert accepts(g, call([("command", "ls")]))
    assert not accepts(g, call([]))
    assert not accepts(g, call([("command", "ls")], "Invented"))


@pytest.mark.parametrize("change", ["too_many", "too_big", "duplicate", "bad_name"])
def test_request_limits_fall_back_and_log(change, caplog):
    srv = object.__new__(Server)
    srv.tok = FakeTok()
    ts = tools()
    if change == "too_many":
        ts *= st.MAX_TOOLS + 1
    elif change == "too_big":
        ts[0]["function"]["parameters"]["description"] = "x" * (st.MAX_SCHEMA_BYTES + 1)
    elif change == "duplicate":
        ts += copy.deepcopy(ts)
    else:
        ts[0]["function"]["name"] = "bad\nname"
    assert srv.constrain(lambda: None, ("auto", None), ts, False) is None
    assert "request uses legacy mode" in caplog.text
