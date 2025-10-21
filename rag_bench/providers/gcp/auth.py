def is_installed():
    try:
        import langchain_google_vertexai
        return True
    except Exception:
        return False
