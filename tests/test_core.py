"""Scientific and interface checks; tiny training runs are not quality benchmarks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from offpsl import Config, OfflineData, fit, load
from offpsl.data import preferences
from offpsl.losses import support_losses, scalarize


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_data_rejects_invalid_arrays_and_bounds(self):
        with self.assertRaises(ValueError):
            OfflineData(np.ones((3, 2)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            OfflineData(np.ones((3, 2)), np.full((3, 2), np.nan))
        with self.assertRaises(ValueError):
            OfflineData(np.ones((3, 2)), np.ones((3, 2)), np.zeros(2), np.full(2, 0.5))

    def test_manuscript_support_norm_and_gradient(self):
        class ZeroDenoiser(torch.nn.Module):
            def forward(self, x, sigma):
                return x * 0
        x = torch.tensor([[3., 4.]], requires_grad=True)
        sup, cov = support_losses(x, ZeroDenoiser(), Config())
        self.assertAlmostEqual(sup.item(), 25)
        self.assertEqual(cov.item(), 0)
        sup.backward()
        torch.testing.assert_close(x.grad, torch.tensor([[6., 8.]]))

    def test_smooth_score_bounds_max_score(self):
        y = torch.tensor([[0.2, 0.8], [-0.1, 0.5]])
        w = torch.tensor([[0.4, 0.6], [0.8, 0.2]])
        hard = (w * (y + 0.2).abs()).max(1).values
        smooth = scalarize(y, w, 0.05, 0.2)
        self.assertTrue(torch.all(smooth >= hard))
        self.assertTrue(torch.all(smooth <= hard + 0.05 * np.log(2) + 1e-6))

    def test_frozen_denoiser_keeps_input_gradient_and_unbiased_covariance(self):
        class HalfDenoiser(torch.nn.Module):
            def forward(self, x, sigma):
                return x / 2
        x = torch.tensor([[3., 4.]], requires_grad=True)
        with patch("torch.randn_like", side_effect=torch.zeros_like):
            sup, _ = support_losses(x, HalfDenoiser().requires_grad_(False), Config())
        sup.backward()
        torch.testing.assert_close(x.grad, x.detach() / 2)

        class IdentityDenoiser(torch.nn.Module):
            def forward(self, x, sigma):
                return x
        cfg = Config(perturbations=2, sigma_min=0.1, sigma_max=0.1)
        with patch("torch.randn_like", return_value=torch.tensor([[-1.], [1.]])):
            sup, cov = support_losses(torch.zeros(1, 1), IdentityDenoiser(), cfg)
        self.assertAlmostEqual(sup.item(), 0.01, places=6)
        self.assertAlmostEqual(cov.item(), 0.02, places=6)

    def test_training_reload_and_feasibility_for_both_methods(self):
        rng = np.random.default_rng(4)
        x = rng.random((40, 3), dtype=np.float32)
        x[:, 2] = 0.4  # A constant feature must remain supported.
        y = np.column_stack((x[:, 0] + x[:, 1], 2 - x[:, 0] + x[:, 1]))
        data = OfflineData(x, y)
        cfg = Config(hidden_dim=16, ensemble_size=2, surrogate_steps=5,
                     denoiser_steps=5, map_steps=5, batch_size=8)
        for method in ("offpsl", "stch-psl"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                model = fit(data, cfg, method, seed=7, verbose=False)
                w = preferences(9, 2)
                result = model.predict(w)
                self.assertTrue(np.isfinite(result["Y_pred"]).all())
                self.assertTrue(np.all(result["X"] >= data.lower))
                self.assertTrue(np.all(result["X"] <= data.upper))
                self.assertTrue(all(not p.requires_grad for p in model.surrogate.parameters()))
                if model.denoiser:
                    self.assertTrue(all(not p.requires_grad for p in model.denoiser.parameters()))
                path = Path(tmp) / "model.pt"
                model.save(path)
                restored = load(path)
                np.testing.assert_array_equal(result["X"], restored.predict(w)["X"])
                np.testing.assert_array_equal(result["Y_pred"], restored.predict(w)["Y_pred"])
                with self.assertRaises(ValueError):
                    restored.predict([[0.2, 0.2]])

    def test_three_objectives_and_unbounded_projection(self):
        rng = np.random.default_rng(2)
        x, y = rng.random((20, 2)), rng.random((20, 3))
        cfg = Config(hidden_dim=8, ensemble_size=2, surrogate_steps=2,
                     denoiser_steps=2, map_steps=2, batch_size=4, bounded_map=False)
        data = OfflineData(x, y)
        model = fit(data, cfg, seed=1, verbose=False)
        w = preferences(7, 3)
        result = model.predict(w)
        self.assertEqual(result["Y_pred"].shape, (7, 3))
        np.testing.assert_allclose(w.sum(1), 1)
        with torch.no_grad():
            for parameter in model.map.parameters():
                parameter.zero_()
            model.map.net[-1].bias.fill_(100)
        result = model.predict(w)
        np.testing.assert_allclose(result["X"], np.broadcast_to(data.upper, (7, 2)))
        self.assertTrue(np.all(result["X_raw"] > data.upper))


if __name__ == "__main__":
    unittest.main()
