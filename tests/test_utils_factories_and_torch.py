from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_bench.utils import factories, repro, torch_utils

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_preferred_device_respects_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: True)
    monkeypatch.setattr(factories, "cuda_available", lambda: True)
    assert factories._preferred_device() == "cpu"


def test_preferred_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: True)
    assert factories._preferred_device() == "cuda"


def test_preferred_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: False)
    assert factories._preferred_device() == "cpu"


def test_make_hf_embeddings_sets_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbeddings:
        __slots__ = ("model_name", "model_kwargs", "encode_kwargs")

        def __init__(self, *, model_name: str, model_kwargs: dict[str, Any], encode_kwargs: dict[str, Any]) -> None:
            self.model_name = model_name
            self.model_kwargs = model_kwargs
            self.encode_kwargs = encode_kwargs

    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=DummyEmbeddings))
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: False)

    emb = factories.make_hf_embeddings("mini", encode_kwargs={"normalize": True})
    assert isinstance(emb, DummyEmbeddings)
    assert emb.model_name == "mini"
    assert emb.model_kwargs["device"] == "cpu"
    assert emb.encode_kwargs == {"normalize": True}


def test_set_seeds_sets_numpy_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    dummy_np = cast(Any, types.ModuleType("numpy"))
    dummy_np.random = SimpleNamespace(seed=lambda value: calls.append(value))
    monkeypatch.setitem(sys.modules, "numpy", dummy_np)

    repro.set_seeds(77)
    assert calls == [77]


def test_set_seeds_handles_missing_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    repro.set_seeds(13)


def test_make_run_id_returns_hex() -> None:
    token = repro.make_run_id()
    assert len(token) == 10 and all(c in "0123456789abcdef" for c in token)


@pytest.mark.gpu
def test_cuda_available_with_stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_torch = cast(Any, types.ModuleType("torch"))
    dummy_torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    assert torch_utils.cuda_available() is True


@pytest.mark.gpu
def test_cuda_available_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("torch", None)
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("torch"):
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert torch_utils.cuda_available() is False


@pytest.mark.gpu
def test_device_str_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: False)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)
    assert torch_utils.device_str() == "cuda"


@pytest.mark.gpu
def test_device_str_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:  # pragma: no cover - trivial helper
        raise RuntimeError("boom")

    monkeypatch.setattr(torch_utils, "wants_cpu", boom)
    assert torch_utils.device_str() == "cpu"


@pytest.mark.gpu
def test_to_device_moves_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_torch = cast(Any, types.ModuleType("torch"))
    dummy_torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(torch_utils, "device_str", lambda: "cuda")

    class TensorLike:
        def __init__(self) -> None:
            self.moved: list[str] = []

        def to(self, device: str) -> str:
            self.moved.append(device)
            return f"tensor@{device}"

    tensor = TensorLike()
    result = torch_utils.to_device(tensor)
    assert result == "tensor@cuda"
    assert tensor.moved == ["cuda"]


@pytest.mark.gpu
def test_to_device_handles_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("torch", None)
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("torch"):
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sentinel = object()
    assert torch_utils.to_device(sentinel) is sentinel


@pytest.mark.gpu
def test_new_tensor_uses_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTorch(types.ModuleType):
        __slots__ = ("calls",)

        def __init__(self) -> None:
            super().__init__("torch")
            self.calls: list[tuple[object, object | None, object | None]] = []
            self.cuda = SimpleNamespace(is_available=lambda: True)

        def as_tensor(
            self,
            data: object,
            *,
            dtype: object | None = None,
            device: object | None = None,
        ) -> tuple[object, object | None, object | None]:
            self.calls.append((data, dtype, device))
            return (data, dtype, device)

    dummy_torch = DummyTorch()
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(torch_utils, "device_str", lambda: "cuda")

    result = torch_utils.new_tensor([1, 2, 3], dtype=None)
    assert result == ([1, 2, 3], None, "cuda")
    assert dummy_torch.calls == [([1, 2, 3], None, "cuda")]
