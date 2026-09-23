"""Core library must not hard-require LangGraph (optional [graph] adapter only)."""

from __future__ import annotations

import pathlib

import news_pipeline


def test_core_modules_do_not_import_langgraph():
    root = pathlib.Path(news_pipeline.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "graph.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "langgraph" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_public_api_does_not_export_graph():
    assert callable(news_pipeline.run_pipeline)
    assert "graph" not in news_pipeline.__all__
    assert "invoke_pipeline" not in news_pipeline.__all__
    assert "build_pipeline_graph" not in news_pipeline.__all__
