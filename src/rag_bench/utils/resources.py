from importlib.resources import files
from pathlib import Path


def get_resource_path(relative: str) -> str:
    base = files("rag_bench").joinpath("resources")
    return str(Path(base).joinpath(*relative.split("/")))
