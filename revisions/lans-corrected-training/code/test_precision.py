"""Precision-observation regressions; no research outputs are written."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from precise_features import PrecisionEnv
from features import NetworkEnv as LegacyEnv, T
from core import XS

torch.set_num_threads(1)


def float64_observation(env):
    """Condition the original preview algebra without intermediate rounding."""
    h = .01 * np.arange(1, 5)
    y = env.ref[0, env.k] + env.x[0]
    heading = env.ref[1, env.k] / env.vx + env.x[2]
    original_preview = y + env.vx * heading * h - env.ref[0, env.k + np.arange(1, 5)]
    residual = (original_preview - env.x[0] - env.vx * env.x[2] * h) / (10 * h**2)
    obs = np.r_[env.x / XS, env.vx / 25, residual,
                env.last_delay / 200,
                (np.mean(env.history) if env.history else 0.) / 200]
    obs[5:9] = obs[5:9] @ T
    return obs


class PrecisionObservationTests(unittest.TestCase):
    def test_eleven_features_match_float64_original_algebra(self):
        for path in ['dlc', 'step', 'sine', 'slalom']:
            env = PrecisionEnv(path=path, delay_mode='mixed')
            try:
                obs, _ = env.reset(seed=782)
                self.assertEqual(obs.shape, (11,))
                self.assertEqual(obs.dtype, np.float32)
                self.assertEqual(env.observation_space.shape, (11,))
                for _ in range(30):
                    np.testing.assert_allclose(env.obs(), float64_observation(env),
                                               rtol=2e-6, atol=2e-6)
                    env.step([0.])
            finally:
                env.close()

    def test_preview_channels_are_independent_of_present_errors(self):
        env = PrecisionEnv(path='slalom', delay_mode='none')
        try:
            env.reset(seed=11)
            env.k = 123
            expected = env.obs()[5:9].copy()
            for state in [np.zeros(4), np.array([.05, .02, .005, .002]),
                          np.array([1e4, -1e3, 1e2, -10.])]:
                env.x = state
                np.testing.assert_array_equal(env.obs()[5:9], expected)
        finally:
            env.close()

    def test_large_state_cancellation_is_reduced(self):
        precise = PrecisionEnv(path='slalom', delay_mode='none')
        legacy = LegacyEnv(path='slalom', delay_mode='none')
        try:
            for env in [precise, legacy]:
                env.reset(seed=11)
                env.k = 123
                env.x = np.array([100., 0., 10., 0.])
            # This numerical fixture intentionally exceeds the rollout failure
            # threshold. Compare only reference channels, whose algebra is state
            # independent, against their zero-state double-precision value.
            precise.x = np.zeros(4)
            expected = float64_observation(precise)[5:9]
            precise.x = legacy.x.copy()
            precise_error = float(np.max(abs(precise.obs()[5:9] - expected)))
            legacy_error = float(np.max(abs(legacy.obs()[5:9] - expected)))
            self.assertGreater(legacy_error, 1e-3)
            self.assertLess(precise_error, legacy_error / 1000)
        finally:
            precise.close(); legacy.close()

    def test_same_actions_produce_identical_physical_trajectories(self):
        for path in ['dlc', 'step', 'sine', 'slalom']:
            for delay, mass in [('none', 1100.), ('mixed', 1650.), ('fixed200', 2200.)]:
                precise = PrecisionEnv(path=path, delay_mode=delay, mass=mass)
                legacy = LegacyEnv(path=path, delay_mode=delay, mass=mass)
                try:
                    precise.reset(seed=782); legacy.reset(seed=782)
                    np.testing.assert_array_equal(precise.x, legacy.x)
                    for transport_delay in [0., 5., 6., 17., 130., 200.]:
                        self.assertEqual(precise.compose([.17], transport_delay),
                                         legacy.compose([.17], transport_delay))
                    for k in range(40):
                        action = [.02 * np.sin(k)]
                        _, rp, tp, trp, ip = precise.step(action)
                        _, rl, tl, trl, il = legacy.step(action)
                        np.testing.assert_array_equal(precise.x, legacy.x)
                        self.assertEqual((rp, tp, trp, ip), (rl, tl, trl, il))
                        if tp or trp:
                            break
                finally:
                    precise.close(); legacy.close()

    def test_deployed_ppo_does_not_call_offline_teacher(self):
        env = PrecisionEnv(kind='lans', delay_mode='none')
        try:
            with patch('core.teacher', side_effect=RuntimeError('offline teacher deployed')):
                policy = PPO('MlpPolicy', env, n_steps=16, batch_size=8,
                             seed=42, device='cpu', verbose=0)
                obs, _ = env.reset(seed=782)
                for _ in range(20):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertFalse(terminated or truncated)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
