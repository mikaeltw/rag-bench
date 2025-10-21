def is_installed():
    try:
        import langchain_aws
        return True
    except Exception:
        return False
