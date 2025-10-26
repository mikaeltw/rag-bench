def test_import() -> None:
    import rag_bench

    assert isinstance(rag_bench.__version__, str)
