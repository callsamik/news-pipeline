"""LLM response validation — all-or-nothing cardinality and schema."""

from __future__ import annotations

from news_pipeline.llm.validate import (
    extract_json_object,
    strip_think,
    validate_summaries_payload,
)


def test_strip_think_removes_closed_blocks():
    raw = '<think>reasoning</think>\n{"summaries":["a"]}'
    assert strip_think(raw).startswith("{")


def test_extract_json_from_fenced_block():
    raw = 'Here:\n```json\n{"summaries":["one","two"]}\n```\n'
    assert extract_json_object(raw) == {"summaries": ["one", "two"]}


def test_validate_success_exact_cardinality():
    raw = '{"summaries":["First summary here.","Second summary here."]}'
    result = validate_summaries_payload(raw, expected_count=2)
    assert result.ok is True
    assert result.summaries == ["First summary here.", "Second summary here."]
    assert result.error is None


def test_validate_rejects_length_mismatch():
    raw = '{"summaries":["only one"]}'
    result = validate_summaries_payload(raw, expected_count=2)
    assert result.ok is False
    assert result.summaries == []
    assert "length mismatch" in (result.error or "")


def test_validate_rejects_empty_string_entry():
    raw = '{"summaries":["ok", "  "]}'
    result = validate_summaries_payload(raw, expected_count=2)
    assert result.ok is False
    assert "non-empty" in (result.error or "")


def test_validate_rejects_non_string_entries():
    raw = '{"summaries":[{"summary":"x"}]}'
    result = validate_summaries_payload(raw, expected_count=1)
    assert result.ok is False
    assert "strings" in (result.error or "")


def test_validate_rejects_malformed_json():
    result = validate_summaries_payload("not json at all", expected_count=1)
    assert result.ok is False
    assert result.summaries == []


def test_validate_rejects_empty_response():
    result = validate_summaries_payload("   ", expected_count=1)
    assert result.ok is False
    assert result.error == "empty response"


def test_validate_rejects_missing_summaries_key():
    result = validate_summaries_payload('{"analyses":[]}', expected_count=1)
    assert result.ok is False
