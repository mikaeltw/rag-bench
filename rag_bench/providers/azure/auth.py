def is_installed():
    try:
        import langchain_openai
        return True
    except Exception:
        return False
