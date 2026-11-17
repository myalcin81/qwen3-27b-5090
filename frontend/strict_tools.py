"""Opt-in qwen3_coder tool grammars. Unsupported schemas retain legacy argument generation, with a warning.

This constrains syntax, not tool authorization or the truth of arguments. Parameter order follows the schema;
raw strings keep the model's native XML representation, all other supported values use JSON subgrammars.
"""

import functools
import json
import logging
import os
import re

import llguidance

from structured import JSON_OPTIONS

LOG = logging.getLogger(__name__)
MAX_TOOLS, MAX_PROPERTIES, MAX_SCHEMA_BYTES = 128, 64, 262144
MAX_GRAMMAR_BYTES = 1048576
NAME = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")
META = {"title", "description", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly"}
TAGS = ("</parameter>", "</function>", "<parameter=", "<function=")
WS = r"/[ \t\r\n]{0,32}/"


class Unsupported(ValueError):
    pass


def enabled(tools):
    return bool(tools) and (os.environ.get("STRICT_TOOLS") == "1" or any(
        isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("strict") is True
        for t in tools))


@functools.lru_cache(maxsize=32)
def validate(lark):
    if len(lark.encode()) > MAX_GRAMMAR_BYTES:
        raise Unsupported("grammar size limit")
    g = llguidance.LLMatcher.grammar_from_lark(lark)
    failed, warnings = llguidance.LLMatcher.validate_grammar_with_warnings(g)
    if failed or any(not w.startswith("no tokenizer - can't validate") for w in warnings):
        # Do not log user schema values or compiler diagnostics containing those values.
        raise Unsupported("grammar compiler rejected or did not enforce the schema")
    return g


def _bounded(value, depth=0, count=None):
    count = [0] if count is None else count
    count[0] += 1
    if depth > 32 or count[0] > 8192:
        raise Unsupported("schema complexity limit")
    for v in value.values() if isinstance(value, dict) else value if isinstance(value, list) else ():
        _bounded(v, depth + 1, count)


def _raw_string(schema, rule):
    if set(schema) - META - {"type", "enum", "const", "minLength", "maxLength"}:
        raise Unsupported("raw string schema keyword")
    lo, hi = schema.get("minLength", 0), schema.get("maxLength")
    if (type(lo) is not int or lo < 0 or lo > 65536 or
            (hi is not None and (type(hi) is not int or hi < lo or hi > 65536))):
        raise Unsupported("raw string length limit")
    values = schema.get("enum")
    if "const" in schema:
        values = [schema["const"]] if values is None or schema["const"] in values else []
    if values is not None:
        if not isinstance(values, list) or not values or any(
                not isinstance(v, str) or any(t in v for t in TAGS) for v in values):
            raise Unsupported("raw string enum")
        values = [v for v in values if len(v) >= lo and (hi is None or len(v) <= hi)]
        if not values:
            raise Unsupported("empty raw string enum")
        return rule + ": (" + " | ".join(json.dumps(v) for v in values) + ') "\\n</parameter>"\n'
    quant = "*" if lo == 0 and hi is None else "{" + str(lo) + "," + (str(hi) if hi is not None else "") + "}"
    # Exclude the XML delimiters from raw text. JSON objects/arrays can contain them (the parser skips JSON strings).
    forbidden = "|".join(re.escape(t).replace("/", r"\/") for t in TAGS)
    return (f'{rule}[suffix="\\n</parameter>"]: /(?s:.){quant}/ & ~/(?s:.*)({forbidden})(?s:.*)/\n')


def _tool(schema, prefix):
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise Unsupported("parameters must be an object schema")
    if set(schema) - META - {"type", "properties", "required", "additionalProperties", "$defs", "definitions", "$schema"}:
        raise Unsupported("object schema keyword")
    props, required = schema.get("properties", {}), schema.get("required", [])
    if not isinstance(props, dict) or len(props) > MAX_PROPERTIES:
        raise Unsupported("property count limit")
    if not isinstance(required, list) or any(not isinstance(k, str) or k not in props for k in required):
        raise Unsupported("required property is not declared")
    seq, rules = [], []
    for i, (name, prop) in enumerate(props.items()):
        if not NAME.fullmatch(name) or not isinstance(prop, dict):
            raise Unsupported("parameter name or schema")
        p, v = f"{prefix}_p{i}", f"{prefix}_v{i}"
        seq.append(p + ("" if name in required else "?"))
        header = json.dumps(f"<parameter={name}>\n")
        if prop.get("type") == "string":
            rules.append(_raw_string(prop, v))
            rules.append(f'{p}: {header} {v} "\\n"\n')
        else:
            # String unions and references to strings use a different wire representation in the legacy parser.
            # Decline them rather than validating JSON quotes and then returning those quotes as argument data.
            if prop.get("type") not in ("integer", "number", "boolean", "null", "object", "array"):
                raise Unsupported("parameter type requires legacy string conversion")
            js = dict(prop)
            for defs in ("$defs", "definitions"):
                if defs in schema:
                    js.setdefault(defs, schema[defs])
            if "x-guidance" in js:
                raise Unsupported("custom JSON grammar options")
            js["x-guidance"] = JSON_OPTIONS
            rules.append(v + ": %json " + json.dumps(js, ensure_ascii=False, allow_nan=False) + "\n")
            rules.append(f'{p}: {header} {v} "\\n</parameter>\\n"\n')
    return " ".join(seq), "".join(rules)


@functools.lru_cache(maxsize=64)
def _rules(key, force_all, selected):
    definitions = json.loads(key)
    names, bodies, rules = set(), [], []
    for i, fn in enumerate(definitions):
        name = fn.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise Unsupported("invalid or duplicate tool name")
        names.add(name)
        if selected is not None and name != selected:
            continue
        prefix = f"q27_tool_{i}"
        schema = fn.get("parameters") or {"type": "object"}
        strict = force_all or fn.get("strict") is True
        body, extra = "", ""
        if strict:
            try:
                _bounded(schema)
                body, extra = _tool(schema, prefix)
                validate(f'start: {body}\n' + extra)
            except (ValueError, RuntimeError, RecursionError, TypeError):
                LOG.warning("strict_tools: tool index %d arguments use legacy mode (unsupported schema or limit)", i)
                strict = False
        header = json.dumps(f"<function={name}>\n")
        if strict:
            rules.append(f'{prefix}: {header} {body} "</function>"\n' + extra)
        else:
            # The name stays whitelisted, but legacy arguments are unconstrained until the function closes.
            rules.append(f'{prefix}: {header} {prefix}_legacy\n'
                         f'{prefix}_legacy[suffix="</function>"]: /(.|\\n)*/\n')
        bodies.append(prefix)
    if not bodies:
        raise Unsupported("no selected tool")
    return "q27_function: " + " | ".join(bodies) + "\n" + "".join(rules)


def rules(tools, choice):
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
        raise Unsupported("tool count limit")
    definitions = []
    for t in tools:
        if not isinstance(t, dict) or not isinstance(t.get("function"), dict):
            raise Unsupported("function definitions required")
        fn = t["function"]
        definitions.append({k: fn[k] for k in ("name", "parameters", "strict") if k in fn})
    key = json.dumps(definitions, ensure_ascii=False, allow_nan=False)
    if len(key.encode()) > MAX_SCHEMA_BYTES:
        raise Unsupported("schema size limit")
    return _rules(key, os.environ.get("STRICT_TOOLS") == "1", choice[1] if choice[0] == "named" else None)


def call_rules(definitions, tokens, parallel=True):
    a, b = tokens
    repeat = f" ({WS} q27_call)*" if parallel else ""
    return (f"calls: q27_call{repeat} {WS}\n"
            f"q27_call: <[{a}]> {WS} q27_function {WS} <[{b}]>\n" + definitions)


def grammar(definitions, tokens, *, after_start=False, parallel=True, lead_ws=False):
    rules_text = call_rules(definitions, tokens, parallel)
    if after_start:
        # The opening special token activates the matcher and is not consumed by it. Subsequent calls remain
        # constrained, including when normal text occurs between calls. Ordinary regex cannot consume specials.
        end = tokens[1]
        tail = " (/(.|\\n)*/ q27_call)*" if parallel else ""
        start = f"{WS} q27_function {WS} <[{end}]>{tail} /(.|\\n)*/"
    else:
        start = (WS + " " if lead_ws else "") + "calls"
    return validate("start: " + start + "\n" + rules_text)
