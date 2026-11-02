import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bgr_refinement', ROOT/'payload/pem/modeling/pixel_decoder/bgr_refinement.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
BGR = MOD.BoundaryGuidedResidualRefinement


class BgrTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.model = BGR(channels=16, hidden=8, gain=0.25)
        self.x = torch.randn(2, 16, 8, 12)
        self.f = torch.randn(2, 16, 8, 12)
        self.b = torch.randn(2, 1, 8, 12)

    def test_zero_initialized_exact_identity(self):
        self.assertTrue(torch.equal(self.model(self.x, self.f, self.b), self.x))

    def test_zero_initialized_grad_to_last_layer(self):
        self.x.requires_grad_()
        y = self.model(self.x, self.f, self.b)
        y.square().mean().backward()
        self.assertGreater(self.model.detail[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(self.x.grad).all().item())

    def test_only_boundary_gate_scales_residual(self):
        with torch.no_grad():
            self.model.detail[-1].bias.fill_(2)
        weak = self.model(self.x, self.f, torch.full_like(self.b, -9))
        strong = self.model(self.x, self.f, torch.full_like(self.b, 9))
        self.assertGreater((strong - self.x).abs().mean().item(),
                           30 * (weak - self.x).abs().mean().item())

    def test_boundary_gate_detached(self):
        with torch.no_grad():
            self.model.detail[-1].weight.fill_(0.01)
        gate = self.b.clone().requires_grad_()
        fine = self.f.clone().requires_grad_()
        self.model(self.x, fine, gate).sum().backward()
        self.assertIsNone(gate.grad)
        self.assertIsNotNone(fine.grad)
        self.assertTrue(torch.isfinite(fine.grad).all().item())

    def test_zero_activation_preserves_base_after_training(self):
        with torch.no_grad():
            self.model.detail[-1].bias.fill_(1)
        self.assertTrue(torch.equal(self.model(self.x, self.f, self.b, activation=0), self.x))

    def test_activation_scales_residual_linearly(self):
        with torch.no_grad():
            self.model.detail[-1].bias.fill_(1)
        a = self.model(self.x, self.f, self.b, activation=0.5) - self.x
        b = self.model(self.x, self.f, self.b, activation=1.0) - self.x
        self.assertTrue(torch.allclose(2*a, b, rtol=1e-4, atol=1e-6))

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            self.model(self.x, self.f[:1], self.b)
        with self.assertRaises(ValueError):
            self.model(self.x, self.f, self.b[:, :, :3])

    def test_invalid_gain(self):
        with self.assertRaises(ValueError):
            BGR(channels=16, hidden=8, gain=0.0)
        with self.assertRaises(ValueError):
            BGR(channels=16, hidden=8, gain=1.5)

    def test_optimizer_step_updates_residual(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=0.05)
        target = self.x + 1
        output = self.model(self.x, self.f, self.b)
        loss = (output-target).square().mean()
        loss.backward()
        optim.step()
        self.assertFalse(torch.equal(self.model(self.x, self.f, self.b), self.x))


if __name__ == '__main__':
    unittest.main()
