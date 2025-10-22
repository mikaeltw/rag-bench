def is_installed():
    try:
        import langchain_aws  # noqa: F401

        return True
    except Exception:
        return False
