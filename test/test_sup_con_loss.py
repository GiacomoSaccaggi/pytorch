# Owner(s): ["module: nn"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    run_tests,
    TestCase,
)


class TestSupConLoss(TestCase):
    """Tests for nn.SupConLoss / F.sup_con_loss."""

    def _reference(self, features, labels, temperature, base_temperature):
        """Independent reference: Khosla et al. Eq. (2), L_out, via python loops."""
        normalized = F.normalize(features, dim=1)
        n = normalized.shape[0]
        similarity = normalized @ normalized.t() / temperature
        out = torch.zeros(n, dtype=features.dtype)
        for i in range(n):
            positives = [j for j in range(n) if j != i and labels[j] == labels[i]]
            if not positives:
                continue
            others = torch.stack([similarity[i, a] for a in range(n) if a != i])
            log_sum_exp = torch.logsumexp(others, 0)
            total = sum(similarity[i, p] - log_sum_exp for p in positives)
            out[i] = -(temperature / base_temperature) * (total / len(positives))
        return out

    # -- basic ------------------------------------------------------------
    def test_forward_with_labels(self):
        features = torch.randn(32, 128)
        labels = torch.randint(0, 10, (32,))
        loss = F.sup_con_loss(features, labels=labels)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(nn.SupConLoss(temperature=0.1)(features, labels=labels), loss)

    def test_forward_with_mask_matches_labels(self):
        features = torch.randn(16, 64)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self.assertEqual(
            F.sup_con_loss(features, mask=mask),
            F.sup_con_loss(features, labels=labels),
        )

    def test_self_supervised_2d_has_no_positives(self):
        """Each 2D sample is its own class, so there are no positive pairs."""
        loss = F.sup_con_loss(torch.randn(16, 64))
        self.assertEqual(loss.item(), 0.0)

    # -- correctness against the reference --------------------------------
    def test_matches_reference(self):
        for temperature, base in ((0.1, 0.07), (0.5, 0.5), (0.07, 1.0)):
            with self.subTest(temperature=temperature, base_temperature=base):
                features = torch.randn(20, 32)
                labels = torch.randint(0, 5, (20,))
                self.assertEqual(
                    F.sup_con_loss(
                        features,
                        labels=labels,
                        temperature=temperature,
                        base_temperature=base,
                        reduction="none",
                    ),
                    self._reference(features, labels, temperature, base),
                )

    def test_matches_reference_when_all_labels_equal(self):
        features = torch.randn(12, 16)
        labels = torch.zeros(12, dtype=torch.long)
        self.assertEqual(
            F.sup_con_loss(features, labels=labels, reduction="none"),
            self._reference(features, labels, 0.1, 0.07),
        )

    def test_loss_decreases_with_intra_class_similarity(self):
        torch.manual_seed(123)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        base = torch.randn(1, 64)
        clustered = torch.cat([base + torch.randn(4, 64) * 0.1, torch.randn(4, 64)])
        self.assertLess(
            F.sup_con_loss(clustered, labels=labels).item(),
            F.sup_con_loss(torch.randn(8, 64), labels=labels).item(),
        )

    # -- reduction --------------------------------------------------------
    def test_reduction_none_shape_and_sign(self):
        features = torch.randn(16, 64)
        labels = torch.randint(0, 4, (16,))
        loss = F.sup_con_loss(features, labels=labels, reduction="none")
        self.assertEqual(loss.shape, torch.Size([16]))
        self.assertTrue((loss >= 0).all())

    def test_mean_averages_only_over_anchors_with_positives(self):
        torch.manual_seed(9)
        features = torch.randn(10, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])  # 4 valid anchors
        unreduced = F.sup_con_loss(features, labels=labels, reduction="none")
        valid_count = (unreduced != 0).sum()
        self.assertEqual(valid_count.item(), 4)
        self.assertEqual(
            F.sup_con_loss(features, labels=labels, reduction="mean"),
            unreduced.sum() / valid_count,
        )

    def test_reduction_sum(self):
        features = torch.randn(16, 64)
        labels = torch.randint(0, 4, (16,))
        self.assertEqual(
            F.sup_con_loss(features, labels=labels, reduction="sum"),
            F.sup_con_loss(features, labels=labels, reduction="none").sum(),
        )

    def test_invalid_reduction_raises(self):
        features = torch.randn(8, 32)
        labels = torch.randint(0, 4, (8,))
        with self.assertRaisesRegex(ValueError, "not a valid value for reduction"):
            F.sup_con_loss(features, labels=labels, reduction="invalid")

    # -- temperature ------------------------------------------------------
    def test_temperatures_must_be_positive(self):
        features = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        for temperature in (0.0, -0.1):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                    F.sup_con_loss(features, labels=labels, temperature=temperature)
        for base in (0.0, -0.1):
            with self.subTest(base_temperature=base):
                with self.assertRaisesRegex(
                    ValueError, "base_temperature must be positive"
                ):
                    F.sup_con_loss(features, labels=labels, base_temperature=base)

    def test_temperature_and_base_temperature_affect_loss(self):
        torch.manual_seed(42)
        features = torch.randn(16, 64)
        labels = torch.randint(0, 4, (16,))
        self.assertNotEqual(
            F.sup_con_loss(features, labels=labels, temperature=0.01).item(),
            F.sup_con_loss(features, labels=labels, temperature=1.0).item(),
        )
        self.assertNotEqual(
            F.sup_con_loss(
                features, labels=labels, temperature=0.1, base_temperature=0.07
            ).item(),
            F.sup_con_loss(
                features, labels=labels, temperature=0.1, base_temperature=0.1
            ).item(),
        )

    def test_module_stores_temperatures(self):
        loss_fn = nn.SupConLoss(temperature=0.5, base_temperature=0.1)
        self.assertEqual(loss_fn.temperature, 0.5)
        self.assertEqual(loss_fn.base_temperature, 0.1)

    # -- gradients --------------------------------------------------------
    def test_gradient_flow_with_labels_and_mask(self):
        features = torch.randn(16, 64, requires_grad=True)
        F.sup_con_loss(features, labels=torch.randint(0, 4, (16,))).backward()
        self.assertFalse(torch.isnan(features.grad).any())

        features = torch.randn(16, 64, requires_grad=True)
        mask = torch.zeros(16, 16)
        mask[0, 1] = mask[1, 0] = mask[2, 3] = mask[3, 2] = 1
        F.sup_con_loss(features, mask=mask).backward()
        self.assertFalse(torch.isnan(features.grad).any())

    def test_gradcheck(self):
        features = torch.randn(8, 6, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        self.assertTrue(
            gradcheck(
                lambda x: F.sup_con_loss(x, labels=labels, temperature=0.5), (features,)
            )
        )

    def test_gradgradcheck(self):
        features = torch.randn(6, 5, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.assertTrue(
            gradgradcheck(
                lambda x: F.sup_con_loss(x, labels=labels, temperature=0.5), (features,)
            )
        )

    def test_no_positives_is_zero_and_differentiable(self):
        """Every label unique: loss is 0 but backward() must still work."""
        features = torch.randn(6, 8, requires_grad=True)
        loss = F.sup_con_loss(features, labels=torch.arange(6))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad, torch.zeros_like(features.grad))

    def test_partial_positives_gradient(self):
        features = torch.randn(8, 32, requires_grad=True)
        labels = torch.tensor([0, 0, 2, 3, 4, 5, 6, 7])
        loss = F.sup_con_loss(features, labels=labels)
        self.assertGreaterEqual(loss.item(), 0)
        loss.backward()
        self.assertFalse(torch.isnan(features.grad).any())

    def test_optimization_reduces_loss(self):
        torch.manual_seed(32)
        encoder = nn.Linear(32, 16)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.05)
        inputs = torch.randn(32, 32)
        labels = torch.randint(0, 4, (32,))
        losses = []
        for _ in range(60):
            loss = F.sup_con_loss(encoder(inputs), labels=labels, temperature=0.2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    # -- edge cases -------------------------------------------------------
    def test_batch_size_one_is_zero_and_differentiable(self):
        features = torch.randn(1, 32, requires_grad=True)
        loss = F.sup_con_loss(features, labels=torch.tensor([0]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertFalse(torch.isnan(features.grad).any())

    def test_batch_size_one_reduction_none(self):
        loss = F.sup_con_loss(
            torch.randn(1, 32), labels=torch.tensor([0]), reduction="none"
        )
        self.assertEqual(loss.shape, torch.Size([1]))
        self.assertEqual(loss.item(), 0.0)

    def test_empty_batch_does_not_crash(self):
        loss = F.sup_con_loss(torch.randn(0, 16), labels=torch.randint(0, 3, (0,)))
        self.assertFalse(torch.isnan(loss))
        self.assertEqual(
            F.sup_con_loss(
                torch.randn(0, 16), labels=torch.randint(0, 3, (0,)), reduction="none"
            ).shape,
            torch.Size([0]),
        )

    def test_all_same_label(self):
        loss = F.sup_con_loss(
            torch.randn(8, 32), labels=torch.zeros(8, dtype=torch.long)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0)

    def test_duplicate_identical_rows(self):
        features = torch.randn(1, 16).repeat(10, 1)
        loss = F.sup_con_loss(features, labels=torch.randint(0, 3, (10,)))
        self.assertTrue(torch.isfinite(loss))

    def test_non_contiguous_and_transposed_inputs(self):
        features = torch.randn(24, 32)[::2]
        self.assertFalse(features.is_contiguous())
        self.assertTrue(
            torch.isfinite(F.sup_con_loss(features, labels=torch.randint(0, 3, (12,))))
        )
        transposed = torch.randn(32, 12).t()
        self.assertTrue(
            torch.isfinite(
                F.sup_con_loss(transposed, labels=torch.randint(0, 3, (12,)))
            )
        )

    def test_labels_and_mask_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            F.sup_con_loss(
                torch.randn(8, 32), labels=torch.randint(0, 4, (8,)), mask=torch.eye(8)
            )

    def test_invalid_dimensions_raise(self):
        with self.assertRaisesRegex(ValueError, "must be 2D or 3D"):
            F.sup_con_loss(torch.randn(8), labels=torch.randint(0, 4, (8,)))
        with self.assertRaisesRegex(ValueError, "2D or 3D"):
            F.sup_con_loss(torch.randn(2, 3, 4, 5))

    def test_labels_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "must have shape"):
            F.sup_con_loss(torch.randn(8, 32), labels=torch.randint(0, 4, (16,)))

    def test_mask_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "must have shape"):
            F.sup_con_loss(torch.randn(8, 32), mask=torch.eye(16))

    # -- multi-view -------------------------------------------------------
    def test_multiview_self_supervised(self):
        loss = F.sup_con_loss(torch.randn(16, 2, 64))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss.item(), 0)

    def test_multiview_reduction_none_shape(self):
        loss = F.sup_con_loss(torch.randn(16, 2, 64), reduction="none")
        self.assertEqual(loss.shape, torch.Size([16, 2]))
        self.assertTrue((loss >= 0).all())

    def test_multiview_equals_manually_flattened(self):
        torch.manual_seed(4)
        batch_size, n_views, dim = 6, 3, 16
        features = torch.randn(batch_size, n_views, dim)
        labels = torch.randint(0, 3, (batch_size,))
        flat_labels = labels.view(-1, 1).repeat(1, n_views).view(-1)
        self.assertEqual(
            F.sup_con_loss(features, labels=labels),
            F.sup_con_loss(
                features.reshape(batch_size * n_views, dim), labels=flat_labels
            ),
        )

    def test_multiview_self_supervised_equals_index_labels(self):
        torch.manual_seed(5)
        batch_size, n_views, dim = 5, 2, 12
        features = torch.randn(batch_size, n_views, dim)
        index_labels = torch.arange(batch_size).view(-1, 1).repeat(1, n_views).view(-1)
        self.assertEqual(
            F.sup_con_loss(features),
            F.sup_con_loss(
                features.reshape(batch_size * n_views, dim), labels=index_labels
            ),
        )

    def test_multiview_square_mask_is_expanded_over_views(self):
        torch.manual_seed(6)
        features = torch.randn(4, 2, 8)
        labels = torch.tensor([0, 0, 1, 1])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self.assertEqual(
            F.sup_con_loss(features, mask=mask),
            F.sup_con_loss(features, labels=labels),
        )

    def test_multiview_single_view_equals_2d(self):
        torch.manual_seed(7)
        features = torch.randn(10, 16)
        labels = torch.randint(0, 3, (10,))
        self.assertEqual(
            F.sup_con_loss(features, labels=labels),
            F.sup_con_loss(features.unsqueeze(1), labels=labels),
        )

    def test_multiview_degenerate_shapes_raise(self):
        with self.assertRaisesRegex(ValueError, "Invalid features shape"):
            F.sup_con_loss(torch.randn(4, 0, 8))
        with self.assertRaisesRegex(ValueError, "Invalid features shape"):
            F.sup_con_loss(torch.randn(4, 2, 0))

    def test_multiview_shape_mismatches_raise(self):
        features = torch.randn(6, 2, 16)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            F.sup_con_loss(features, labels=torch.randint(0, 3, (12,)))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            F.sup_con_loss(features, mask=torch.eye(7))

    def test_multiview_gradient_flow(self):
        features = torch.randn(8, 2, 32, requires_grad=True)
        F.sup_con_loss(features).backward()
        self.assertFalse(torch.isnan(features.grad).any())

    def test_multiview_module(self):
        loss = nn.SupConLoss()(torch.randn(16, 3, 64))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss.item(), 0)

    # -- numerical stability ----------------------------------------------
    def test_extreme_input_scales(self):
        for scale in (1e-8, 1e-4, 1e4, 1e8):
            with self.subTest(scale=scale):
                features = torch.randn(12, 16) * scale
                loss = F.sup_con_loss(features, labels=torch.randint(0, 3, (12,)))
                self.assertTrue(torch.isfinite(loss))

    def test_extreme_temperatures(self):
        features = torch.randn(12, 16)
        labels = torch.randint(0, 3, (12,))
        for temperature in (1e-6, 1e-3, 1e3, 1e6):
            with self.subTest(temperature=temperature):
                loss = F.sup_con_loss(features, labels=labels, temperature=temperature)
                self.assertTrue(torch.isfinite(loss))

    def test_zero_vectors_do_not_produce_nan(self):
        loss = F.sup_con_loss(torch.zeros(12, 16), labels=torch.randint(0, 3, (12,)))
        self.assertFalse(torch.isnan(loss))

    def test_gradient_stability_with_large_logits(self):
        features = (torch.randn(16, 32) * 10).requires_grad_(True)
        labels = torch.randint(0, 4, (16,))
        F.sup_con_loss(features, labels=labels, temperature=0.01).backward()
        self.assertFalse(torch.isnan(features.grad).any())
        self.assertFalse(torch.isinf(features.grad).any())

    # -- invariances ------------------------------------------------------
    def test_invariant_to_input_rescaling(self):
        features = torch.randn(12, 16)
        labels = torch.randint(0, 3, (12,))
        self.assertEqual(
            F.sup_con_loss(features, labels=labels),
            F.sup_con_loss(features * 7.5, labels=labels),
        )

    def test_invariant_to_permutation(self):
        torch.manual_seed(26)
        features = torch.randn(12, 16)
        labels = torch.randint(0, 3, (12,))
        permutation = torch.randperm(12)
        self.assertEqual(
            F.sup_con_loss(features, labels=labels, reduction="sum"),
            F.sup_con_loss(
                features[permutation], labels=labels[permutation], reduction="sum"
            ),
        )

    def test_inputs_are_not_mutated(self):
        features = torch.randn(12, 16)
        labels = torch.randint(0, 3, (12,))
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features_original, mask_original = features.clone(), mask.clone()
        F.sup_con_loss(features, mask=mask)
        self.assertEqual(features, features_original)
        self.assertEqual(mask, mask_original)

    # -- dtypes -----------------------------------------------------------
    def test_dtype_is_preserved(self):
        labels = torch.randint(0, 3, (12,))
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                features = torch.randn(12, 16, dtype=dtype)
                self.assertEqual(F.sup_con_loss(features, labels=labels).dtype, dtype)
                self.assertEqual(
                    F.sup_con_loss(features, labels=labels, reduction="none").dtype,
                    dtype,
                )

    def test_dtype_preserved_when_some_anchors_lack_positives(self):
        """The no-positive mask must not silently downcast float64."""
        features = torch.randn(8, 16, dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(
            F.sup_con_loss(features, labels=labels, reduction="none").dtype,
            torch.float64,
        )

    def test_low_precision_dtypes(self):
        labels = torch.randint(0, 4, (16,))
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                features = torch.randn(16, 32, dtype=dtype)
                self.assertTrue(torch.isfinite(F.sup_con_loss(features, labels=labels)))

    def test_integer_label_dtypes(self):
        features = torch.randn(8, 16)
        base = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        expected = F.sup_con_loss(features, labels=base)
        for dtype in (torch.int8, torch.int16, torch.int32, torch.long):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    F.sup_con_loss(features, labels=base.to(dtype)), expected
                )

    def test_bool_mask_is_accepted(self):
        features = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        bool_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        self.assertEqual(
            F.sup_con_loss(features, mask=bool_mask),
            F.sup_con_loss(features, labels=labels),
        )

    # -- module plumbing --------------------------------------------------
    def test_module_attributes(self):
        for name in ("temperature", "base_temperature", "reduction"):
            self.assertIn(name, nn.SupConLoss.__constants__)
        loss_fn = nn.SupConLoss(temperature=0.1, base_temperature=0.05, reduction="sum")
        self.assertIn("SupConLoss", repr(loss_fn))
        self.assertIsInstance(loss_fn, nn.Module)
        self.assertEqual(list(loss_fn.parameters()), [])

    def test_module_matches_functional_for_all_reductions(self):
        features = torch.randn(12, 16)
        labels = torch.randint(0, 3, (12,))
        for reduction in ("none", "mean", "sum"):
            with self.subTest(reduction=reduction):
                loss_fn = nn.SupConLoss(
                    temperature=0.2, base_temperature=0.1, reduction=reduction
                )
                self.assertEqual(
                    loss_fn(features, labels=labels),
                    F.sup_con_loss(
                        features,
                        labels=labels,
                        temperature=0.2,
                        base_temperature=0.1,
                        reduction=reduction,
                    ),
                )

    def test_deepcopy_and_state_dict(self):
        import copy

        loss_fn = nn.SupConLoss(temperature=0.42)
        self.assertEqual(copy.deepcopy(loss_fn).temperature, 0.42)
        other = nn.SupConLoss()
        other.load_state_dict(loss_fn.state_dict())
        self.assertEqual(loss_fn.state_dict(), other.state_dict())

    # -- scripting and compilation ----------------------------------------
    def test_torchscript(self):
        scripted = torch.jit.script(nn.SupConLoss(temperature=0.1))
        features = torch.randn(12, 16, requires_grad=True)
        labels = torch.randint(0, 3, (12,))
        loss = scripted(features, labels)
        self.assertEqual(loss, F.sup_con_loss(features, labels=labels, temperature=0.1))
        loss.backward()
        self.assertEqual(features.grad.shape, features.shape)

    def test_torchscript_multiview(self):
        scripted = torch.jit.script(nn.SupConLoss())
        features = torch.randn(6, 2, 16)
        self.assertEqual(scripted(features), F.sup_con_loss(features))

    def test_compile_fullgraph(self):
        compiled = torch.compile(F.sup_con_loss, fullgraph=True, backend="eager")
        features = torch.randn(16, 32)
        labels = torch.randint(0, 4, (16,))
        self.assertEqual(
            compiled(features, labels=labels),
            F.sup_con_loss(features, labels=labels),
        )

    def test_compile_module(self):
        loss_fn = nn.SupConLoss()
        compiled = torch.compile(loss_fn, fullgraph=True, backend="eager")
        features = torch.randn(16, 64)
        labels = torch.randint(0, 4, (16,))
        self.assertEqual(
            compiled(features, labels=labels), loss_fn(features, labels=labels)
        )

    def test_compile_with_mask(self):
        compiled = torch.compile(F.sup_con_loss, backend="eager")
        features = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self.assertEqual(
            compiled(features, mask=mask), F.sup_con_loss(features, mask=mask)
        )

    def test_compile_multiview(self):
        compiled = torch.compile(F.sup_con_loss, fullgraph=True, backend="eager")
        features = torch.randn(8, 2, 32)
        self.assertEqual(compiled(features), F.sup_con_loss(features))

    def test_compile_dynamic_shapes(self):
        compiled = torch.compile(F.sup_con_loss, dynamic=True, backend="eager")
        for batch_size in (8, 16, 32):
            with self.subTest(batch_size=batch_size):
                features = torch.randn(batch_size, 24)
                labels = torch.randint(0, 3, (batch_size,))
                self.assertEqual(
                    compiled(features, labels=labels),
                    F.sup_con_loss(features, labels=labels),
                )

    def test_compile_backward(self):
        compiled = torch.compile(F.sup_con_loss, backend="eager")
        features = torch.randn(12, 16, requires_grad=True)
        compiled(features, labels=torch.randint(0, 3, (12,))).backward()
        self.assertFalse(torch.isnan(features.grad).any())


class TestSupConLossDevice(TestCase):
    """Device-parameterized coverage."""

    def test_device_of_output_matches_input(self, device):
        features = torch.randn(16, 64, device=device)
        labels = torch.randint(0, 4, (16,), device=device)
        loss = F.sup_con_loss(features, labels=labels)
        self.assertEqual(loss.device.type, torch.device(device).type)

    def test_gradient_on_device(self, device):
        features = torch.randn(16, 32, device=device, requires_grad=True)
        labels = torch.randint(0, 4, (16,), device=device)
        F.sup_con_loss(features, labels=labels).backward()
        self.assertIsNotNone(features.grad)
        self.assertFalse(torch.isnan(features.grad).any())

    def test_multiview_on_device(self, device):
        features = torch.randn(8, 2, 32, device=device)
        loss = F.sup_con_loss(features)
        self.assertEqual(loss.device.type, torch.device(device).type)

    @onlyCPU
    def test_cpu_double_precision(self, device):
        features = torch.randn(12, 16, device=device, dtype=torch.double)
        labels = torch.randint(0, 3, (12,), device=device)
        self.assertEqual(F.sup_con_loss(features, labels=labels).dtype, torch.double)


instantiate_device_type_tests(TestSupConLossDevice, globals())


if __name__ == "__main__":
    run_tests()
