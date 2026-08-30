"""Comprehensive tests for InfoNCELoss."""

import pytest
import torch

from torch.nn import InfoNCELoss, info_nce_loss


def get_device():
    """Get available device for testing."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TestInfoNCELossBasic:
    """Basic functionality tests."""

    def test_forward_pass(self):
        """Test basic forward pass works."""
        query = torch.randn(32, 128)
        positive_key = torch.randn(32, 128)
        loss = info_nce_loss(query, positive_key)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_module_forward(self):
        """Test module API forward pass."""
        loss_fn = InfoNCELoss(temperature=0.07)
        query = torch.randn(32, 128)
        positive_key = torch.randn(32, 128)
        loss = loss_fn(query, positive_key)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_explicit_negatives(self):
        """Test with explicit negative keys."""
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        negative_keys = torch.randn(128, 64)
        loss = info_nce_loss(query, positive_key, negative_keys)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_module_with_negatives(self):
        """Test module API with explicit negatives."""
        loss_fn = InfoNCELoss()
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        negative_keys = torch.randn(128, 64)
        loss = loss_fn(query, positive_key, negative_keys)
        assert loss.shape == ()


class TestInfoNCELossGradient:
    """Gradient computation tests."""

    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        query = torch.randn(8, 32, requires_grad=True)
        positive_key = torch.randn(8, 32, requires_grad=True)
        loss = info_nce_loss(query, positive_key)
        loss.backward()
        assert query.grad is not None
        assert positive_key.grad is not None
        assert not torch.isnan(query.grad).any()
        assert not torch.isnan(positive_key.grad).any()

    def test_gradient_with_negatives(self):
        """Test gradients with explicit negatives."""
        query = torch.randn(8, 32, requires_grad=True)
        positive_key = torch.randn(8, 32, requires_grad=True)
        negative_keys = torch.randn(64, 32, requires_grad=True)
        loss = info_nce_loss(query, positive_key, negative_keys)
        loss.backward()
        assert query.grad is not None
        assert positive_key.grad is not None
        assert negative_keys.grad is not None


class TestInfoNCELossReduction:
    """Test different reduction modes."""

    def test_reduction_none(self):
        """Test reduction='none' returns per-sample losses."""
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        loss = info_nce_loss(query, positive_key, reduction="none")
        assert loss.shape == (16,)
        assert (loss >= 0).all()

    def test_reduction_mean(self):
        """Test reduction='mean' returns scalar mean."""
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        loss_none = info_nce_loss(query, positive_key, reduction="none")
        loss_mean = info_nce_loss(query, positive_key, reduction="mean")
        assert loss_mean.shape == ()
        assert torch.allclose(loss_mean, loss_none.mean())

    def test_reduction_sum(self):
        """Test reduction='sum' returns scalar sum."""
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        loss_none = info_nce_loss(query, positive_key, reduction="none")
        loss_sum = info_nce_loss(query, positive_key, reduction="sum")
        assert loss_sum.shape == ()
        assert torch.allclose(loss_sum, loss_none.sum())

    def test_invalid_reduction(self):
        """Test invalid reduction raises error."""
        query = torch.randn(8, 32)
        positive_key = torch.randn(8, 32)
        with pytest.raises(ValueError, match="Invalid reduction"):
            info_nce_loss(query, positive_key, reduction="invalid")


class TestInfoNCELossTemperature:
    """Test temperature parameter effects."""

    def test_different_temperatures(self):
        """Test loss varies with temperature."""
        torch.manual_seed(42)
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)

        loss_low_temp = info_nce_loss(query, positive_key, temperature=0.01)
        loss_high_temp = info_nce_loss(query, positive_key, temperature=1.0)

        # Lower temperature should give higher loss for random embeddings
        # (sharper distribution = harder task)
        assert loss_low_temp.item() != loss_high_temp.item()

    def test_temperature_in_module(self):
        """Test module stores temperature correctly."""
        loss_fn = InfoNCELoss(temperature=0.5)
        assert loss_fn.temperature == 0.5


class TestInfoNCELossEdgeCases:
    """Edge case handling tests."""

    def test_batch_size_one(self):
        """Test batch size 1 without negatives returns zero."""
        query = torch.randn(1, 32)
        positive_key = torch.randn(1, 32)
        loss = info_nce_loss(query, positive_key)
        assert loss.item() == 0.0

    def test_batch_size_one_reduction_none(self):
        """Test batch size 1 with reduction='none'."""
        query = torch.randn(1, 32)
        positive_key = torch.randn(1, 32)
        loss = info_nce_loss(query, positive_key, reduction="none")
        assert loss.shape == (1,)
        assert loss.item() == 0.0

    def test_batch_size_one_with_negatives(self):
        """Test batch size 1 with explicit negatives works."""
        query = torch.randn(1, 32)
        positive_key = torch.randn(1, 32)
        negative_keys = torch.randn(64, 32)
        loss = info_nce_loss(query, positive_key, negative_keys)
        assert loss.item() > 0  # Should have non-zero loss with negatives

    def test_batch_size_two(self):
        """Test batch size 2 (minimal case with negatives)."""
        query = torch.randn(2, 32)
        positive_key = torch.randn(2, 32)
        loss = info_nce_loss(query, positive_key)
        assert loss.item() >= 0

    def test_invalid_query_dim(self):
        """Test invalid query dimension raises error."""
        query = torch.randn(8)  # 1D instead of 2D
        positive_key = torch.randn(8, 32)
        with pytest.raises(ValueError, match="must be 2D"):
            info_nce_loss(query, positive_key)

    def test_shape_mismatch(self):
        """Test shape mismatch raises error."""
        query = torch.randn(8, 32)
        positive_key = torch.randn(16, 32)  # Different batch size
        with pytest.raises(ValueError, match="same shape"):
            info_nce_loss(query, positive_key)

    def test_negative_keys_dim_mismatch(self):
        """Test negative keys embedding dimension mismatch."""
        query = torch.randn(8, 32)
        positive_key = torch.randn(8, 32)
        negative_keys = torch.randn(64, 64)  # Wrong embedding dim
        with pytest.raises(ValueError, match="embedding dim must match"):
            info_nce_loss(query, positive_key, negative_keys)


class TestInfoNCELossNumericalStability:
    """Numerical stability tests."""

    def test_large_embeddings(self):
        """Test with large embedding values."""
        query = torch.randn(16, 64) * 100
        positive_key = torch.randn(16, 64) * 100
        loss = info_nce_loss(query, positive_key)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_small_temperature(self):
        """Test with very small temperature."""
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        loss = info_nce_loss(query, positive_key, temperature=0.001)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_stability_large_logits(self):
        """Test gradient stability with large logits."""
        query = torch.randn(8, 32) * 10
        positive_key = torch.randn(8, 32) * 10
        query.requires_grad_(True)
        positive_key.requires_grad_(True)
        loss = info_nce_loss(query, positive_key, temperature=0.01)
        loss.backward()
        assert query.grad is not None
        assert positive_key.grad is not None
        assert not torch.isnan(query.grad).any()
        assert not torch.isnan(positive_key.grad).any()


class TestInfoNCELossBehavior:
    """Behavioral correctness tests."""

    def test_loss_non_negative(self):
        """Test loss is always non-negative."""
        for _ in range(10):
            query = torch.randn(32, 64)
            positive_key = torch.randn(32, 64)
            loss = info_nce_loss(query, positive_key)
            assert loss.item() >= 0

    def test_loss_decreases_with_similarity(self):
        """Test loss is lower when positives are more similar."""
        torch.manual_seed(123)
        query = torch.randn(16, 64)

        # High similarity: positive is close to query
        positive_similar = query + torch.randn_like(query) * 0.1
        loss_similar = info_nce_loss(query, positive_similar)

        # Low similarity: positive is random
        positive_random = torch.randn(16, 64)
        loss_random = info_nce_loss(query, positive_random)

        assert loss_similar.item() < loss_random.item()

    def test_perfect_similarity_low_loss(self):
        """Test that identical embeddings give low loss."""
        query = torch.randn(16, 64)
        positive_key = query.clone()  # Identical
        loss = info_nce_loss(query, positive_key)
        # Loss should be relatively low (depends on negatives)
        assert loss.item() < 5.0  # Reasonable upper bound


class TestInfoNCELossTorchCompile:
    """torch.compile compatibility tests."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_functional(self):
        """Test functional API works with torch.compile."""
        # Use eager backend for portability (avoids C++ compiler issues)
        compiled_loss = torch.compile(info_nce_loss, backend="eager")
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)

        loss_regular = info_nce_loss(query, positive_key)
        loss_compiled = compiled_loss(query, positive_key)

        assert torch.allclose(loss_regular, loss_compiled, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_module(self):
        """Test module works with torch.compile."""
        loss_fn = InfoNCELoss()
        compiled_fn = torch.compile(loss_fn, backend="eager")
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)

        loss_regular = loss_fn(query, positive_key)
        loss_compiled = compiled_fn(query, positive_key)

        assert torch.allclose(loss_regular, loss_compiled, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_with_negatives(self):
        """Test torch.compile with explicit negatives."""
        compiled_loss = torch.compile(info_nce_loss, backend="eager")
        query = torch.randn(8, 32)
        positive_key = torch.randn(8, 32)
        negative_keys = torch.randn(64, 32)

        loss_regular = info_nce_loss(query, positive_key, negative_keys)
        loss_compiled = compiled_loss(query, positive_key, negative_keys)

        assert torch.allclose(loss_regular, loss_compiled, atol=1e-5)


class TestInfoNCELossDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Test on CPU."""
        query = torch.randn(16, 64, device="cpu")
        positive_key = torch.randn(16, 64, device="cpu")
        loss = info_nce_loss(query, positive_key)
        assert loss.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps(self):
        """Test on MPS (Apple Silicon)."""
        device = torch.device("mps")
        query = torch.randn(16, 64, device=device)
        positive_key = torch.randn(16, 64, device=device)
        loss = info_nce_loss(query, positive_key)
        assert loss.device.type == "mps"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test on CUDA."""
        device = torch.device("cuda")
        query = torch.randn(16, 64, device=device)
        positive_key = torch.randn(16, 64, device=device)
        loss = info_nce_loss(query, positive_key)
        assert loss.device.type == "cuda"


class TestInfoNCELossModuleAttributes:
    """Test module attributes and JIT compatibility."""

    def test_constants_defined(self):
        """Test __constants__ is defined for JIT."""
        assert hasattr(InfoNCELoss, "__constants__")
        assert "temperature" in InfoNCELoss.__constants__
        assert "reduction" in InfoNCELoss.__constants__

    def test_repr(self):
        """Test module repr."""
        loss_fn = InfoNCELoss(temperature=0.1, reduction="sum")
        repr_str = repr(loss_fn)
        assert "InfoNCELoss" in repr_str

    def test_module_inheritance(self):
        """Test proper inheritance."""
        from torch.nn.modules.loss import _Loss

        loss_fn = InfoNCELoss()
        assert isinstance(loss_fn, _Loss)
        assert isinstance(loss_fn, torch.nn.Module)


class TestInfoNCELossSymmetric:
    """Test symmetric / bidirectional InfoNCE (CLIP style)."""

    def test_symmetric_loss(self):
        """Test bidirectional InfoNCE computation."""
        query = torch.randn(16, 64, requires_grad=True)
        key = torch.randn(16, 64, requires_grad=True)

        loss_q2k = info_nce_loss(query, key)
        loss_k2q = info_nce_loss(key, query)
        sym_loss = 0.5 * (loss_q2k + loss_k2q)

        assert sym_loss.shape == ()
        assert sym_loss.item() >= 0
        sym_loss.backward()
        assert query.grad is not None
        assert key.grad is not None


class TestInfoNCELossJIT:
    """Test TorchScript compatibility."""

    def test_jit_script_module(self):
        """Test that InfoNCELoss can be scripted."""
        loss_fn = InfoNCELoss(temperature=0.07)
        scripted = torch.jit.script(loss_fn)
        query = torch.randn(8, 64)
        positive_key = torch.randn(8, 64)
        loss = scripted(query, positive_key)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_jit_script_with_negatives(self):
        """Test scripted module with explicit negatives."""
        loss_fn = InfoNCELoss(temperature=0.1)
        scripted = torch.jit.script(loss_fn)
        query = torch.randn(8, 64)
        positive_key = torch.randn(8, 64)
        negative_keys = torch.randn(32, 64)
        loss = scripted(query, positive_key, negative_keys)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_jit_script_gradient(self):
        """Test that gradients flow through scripted module."""
        loss_fn = InfoNCELoss(temperature=0.07)
        scripted = torch.jit.script(loss_fn)
        query = torch.randn(8, 64, requires_grad=True)
        positive_key = torch.randn(8, 64)
        loss = scripted(query, positive_key)
        loss.backward()
        assert query.grad is not None
        assert query.grad.shape == query.shape
