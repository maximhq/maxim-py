current_version = "unknown"

try:
    import importlib.metadata

    current_version = importlib.metadata.version("maxim-py")
except Exception:
    pass
