import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.gpu
def test_cuda_tensor_math() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.ones((256, 256), device="cuda")
    y = torch.ones((256, 256), device="cuda")
    z = (x @ y).sum().item()
    assert z > 0
