import news_pipeline
import pathlib


def test_library_tree_has_no_ain_imports():
    root = pathlib.Path(news_pipeline.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import ain" in text or "from ain" in text:
            offenders.append(str(path))
    assert offenders == []
