from importlib.resources import files
from importlib.resources.abc import Traversable


def get_resource_path(relative: str) -> str:
    base: Traversable = files("rag_bench").joinpath("resources")
    return str(base.joinpath(*relative.split("/")))
