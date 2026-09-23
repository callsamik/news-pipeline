"""Core library must not hard-require LangGraph (optional [graph] adapter only).

Acceptance: with langgraph unavailable, ``import news_pipeline`` and
``import news_pipeline.pipeline`` succeed; ``news_pipeline.graph`` raises a clear
InstallError pointing at ``news-pipeline[graph]``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

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


def test_core_imports_when_langgraph_unavailable():
    """No-graph acceptance — does not rely on langgraph being absent from the env.

    Blocks langgraph via ``sys.meta_path`` in a fresh subprocess so the test
    still fails closed when ``[dev]`` has langgraph installed.
    """
    src_root = pathlib.Path(news_pipeline.__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _BlockLangGraph(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):  # noqa: ANN001
                if fullname == "langgraph" or fullname.startswith("langgraph."):
                    raise ImportError("langgraph blocked for no-graph acceptance")
                return None

        sys.meta_path.insert(0, _BlockLangGraph())
        for key in list(sys.modules):
            if key == "langgraph" or key.startswith("langgraph."):
                del sys.modules[key]
            if key == "news_pipeline" or key.startswith("news_pipeline."):
                del sys.modules[key]

        import news_pipeline
        import news_pipeline.pipeline

        assert callable(news_pipeline.run_pipeline)
        assert callable(news_pipeline.pipeline.run_pipeline)

        try:
            import news_pipeline.graph  # noqa: F401
        except ImportError as exc:
            msg = str(exc)
            assert "news-pipeline[graph]" in msg
        else:
            raise SystemExit("news_pipeline.graph must fail without langgraph")
        """
    )
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(src_root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
