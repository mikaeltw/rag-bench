def test_import():
    import rag_bench

    assert isinstance(rag_bench.__version__, str)
