import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / 'code', ROOT / 'code/WM_QAS', ROOT / 'code/WM_QAS/phase2_surrogate', ROOT / 'code/WM_QAS/analysis'):
    sys.path.insert(0, str(p))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import torch
from utils import get_config
from environment import CircuitEnv
from runner_baseline import BaselineRunner, baseline_noise_config, load_baseline_checkpoint
from noise_ptm.integration import build_evaluator
from eval_policy_traces import _reload_rlqas, _trace_rlqas, _episode_reductions
from policy_quality_table import bestofK


class ReleaseChecks(unittest.TestCase):
    def test_bundled_data(self):
        manifest = json.loads((ROOT / 'data/manifest.json').read_text())
        self.assertEqual(set(manifest), {p.name for p in (ROOT / 'data/mol_data').glob('*.npz')})
        for name, meta in manifest.items():
            payload = (ROOT / 'data/mol_data' / name).read_bytes()
            self.assertEqual(len(payload), meta['bytes'])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), meta['sha256'])

    def test_config_independent_of_cwd(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='dreamqas paths ') as td:
            try:
                os.chdir(td)
                self.assertEqual(get_config('analysis/', 'G0_v2_LiH4q.cfg')['env']['num_qubits'], 4)
                with self.assertRaises(FileNotFoundError):
                    get_config('analysis/', 'missing.cfg')
            finally:
                os.chdir(old)

    def test_checkpoint_noise_roundtrip(self):
        conf = get_config('analysis/', 'G0_v2_LiH4q.cfg')
        env = CircuitEnv(conf, device=torch.device('cpu'))
        settings = dict(mode='ptm', p1q=0.001, p2q=0.005, p_ro=0.002, basis_change=False)
        env.attach_noisy_evaluator(build_evaluator(env, **settings))
        with tempfile.TemporaryDirectory(prefix='dreamqas checkpoint ') as td:
            original = BaselineRunner(env, conf, torch.device('cpu'), td, seed=2,
                                      actor_hidden_dim=16, molecule='LiH4q',
                                      env_cfg='G0_v2_LiH4q', experiment_name='analysis/')
            original._save_ckpt(0)
            ck = torch.load(Path(td) / 'ckpt/ep0.pt', map_location='cpu', weights_only=False)
            self.assertEqual(ck['noise_config'], settings)
            restored, seed = _reload_rlqas(ck, 'cpu')
            self.assertEqual(seed, 2)
            self.assertEqual(baseline_noise_config(restored.env), settings)
            for key, val in original.actor.state_dict().items():
                self.assertTrue(torch.equal(val, restored.actor.state_dict()[key]))
            torch.manual_seed(99); np.random.seed(99)
            before = _trace_rlqas(original)
            torch.manual_seed(99); np.random.seed(99)
            after = _trace_rlqas(restored)
            for a, b in zip(before, after):
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)
            self.assertGreater(np.max(np.abs(before[0] - before[1])), 1e-3)
            legacy = dict(ck); del legacy['noise_config']
            with self.assertRaisesRegex(ValueError, 'no noise metadata'):
                load_baseline_checkpoint(legacy, 'cpu')
            clean = load_baseline_checkpoint(legacy, 'cpu', {'mode': 'off'})
            self.assertEqual(baseline_noise_config(clean.env), {'mode': 'off'})
            original.env.attach_noisy_evaluator(None)
            original._save_ckpt(1)
            clean_ck = torch.load(Path(td) / 'ckpt/ep1.pt', map_location='cpu', weights_only=False)
            self.assertEqual(baseline_noise_config(load_baseline_checkpoint(clean_ck, 'cpu').env), {'mode': 'off'})

    def test_diagnostics_off_checkpoint_reporting(self):
        from config import Config
        from runner import Runner
        from phase2_surrogate.eval_harness import save_checkpoint, evaluate_checkpoint
        from eval_policy_traces import _reload_dreamqas, _trace_dreamqas
        with tempfile.TemporaryDirectory(prefix='dreamqas frozen reporting ') as td:
            cfg = Config(molecule='LiH4q', env_cfg='G0_v2_LiH4q', device='cpu',
                         oracle_free=True, enable_ground_truth_diagnostics=False,
                         out_dir=td, wm_hidden=16, actor_hidden=16, wm_embed=8, wm_ensemble_K=2)
            original = Runner(cfg, _eval_mode=True)
            self.assertIsNone(original.env.min_energy)
            path = str(Path(td) / 'checkpoint.pt')
            save_checkpoint(original, 0, path)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            self.assertFalse(ck['cfg']['enable_ground_truth_diagnostics'])
            restored, _ = _reload_dreamqas(ck, 'cpu')
            self.assertTrue(np.isfinite(restored.env.min_energy))
            self.assertFalse(ck['cfg']['enable_ground_truth_diagnostics'])
            self.assertFalse(original.cfg.enable_ground_truth_diagnostics)
            torch.manual_seed(99); np.random.seed(99)
            obs, clean = _trace_dreamqas(restored)
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(clean).all())
            stats = evaluate_checkpoint(path, 1, device='cpu', eval_seed=99)
            self.assertEqual(stats['n'], 1)
            self.assertAlmostEqual(stats['episode_bests'][0], float(obs.min()), places=6)

    def test_common_budget_does_not_accept_smoke_or_later_checkpoints(self):
        from of_main_table import internal_bo1
        from of_ablation_table import seed_bo1
        with tempfile.TemporaryDirectory(prefix='dreamqas table budget ') as td:
            run = Path(td) / 'gru_energy_surrogate_LiH4q_s0_of'
            run.mkdir()
            pattern = str(Path(td) / 'gru_energy_surrogate_{m}_s*_of')
            f = run / 'eval_traces_ep15000.jsonl'
            for episode in (4, 30000):
                f.write_text(json.dumps({'episode': episode, 'ep_best': [1., 3.]}) + '\n')
                self.assertEqual(internal_bo1(pattern, 'LiH4q'), [])
                self.assertEqual(seed_bo1(pattern, 'LiH4q'), {})
            f.write_text(json.dumps({'episode': 15000, 'ep_best': [1., 3.]}) + '\n')
            self.assertEqual(internal_bo1(pattern, 'LiH4q'), [2.])
            self.assertEqual(seed_bo1(pattern, 'LiH4q'), {'0': 2.})

    def test_selected_prefix_and_order_statistic(self):
        result = _episode_reductions(np.array([9., 2., 3.]), np.array([1., 6., 5.]))
        self.assertEqual(result['best'], 2.)
        self.assertEqual(result['best_noiseless'], 6.)
        self.assertAlmostEqual(bestofK([1., 3.], 1), 2.)
        self.assertAlmostEqual(bestofK([1., 3.], 2), 1.5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
