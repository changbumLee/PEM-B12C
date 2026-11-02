import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bgr_runner', ROOT/'run_b12c_bgr_60k.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RunnerTests(unittest.TestCase):
    def test_training_uses_60k_and_full_batch(self):
        args = SimpleNamespace(smoke=False, batch_size=8, seed=2026,
                               workers=8, hidden=64, gain=.25)
        cmd = MOD.make_command(Path('/repo'), Path('/old_b12c.yaml'), args, Path('/out'))
        self.assertEqual(cmd[cmd.index('SOLVER.MAX_ITER')+1], '60000')
        self.assertEqual(cmd[cmd.index('SOLVER.IMS_PER_BATCH')+1], '8')
        self.assertEqual(cmd[cmd.index('MODEL.BGR.ENABLED')+1], 'True')
        self.assertEqual(cmd[cmd.index('MODEL.BGR.GAIN')+1], '0.25')
        self.assertEqual(cmd[cmd.index('MODEL.WEIGHTS')+1], MOD.WEIGHTS)

    def test_smoke_uses_batch8_and_warmup(self):
        args = SimpleNamespace(smoke=True, batch_size=8, seed=2026,
                               workers=8, hidden=64, gain=.25)
        cmd = MOD.make_command(Path('/repo'), Path('/old_b12c.yaml'), args, Path('/out'))
        self.assertEqual(cmd[cmd.index('SOLVER.MAX_ITER')+1], '3')
        self.assertEqual(cmd[cmd.index('SOLVER.IMS_PER_BATCH')+1], '8')
        self.assertIn('MODEL.CSM_SHAPE.WARMUP_END', cmd)

    def test_runner_is_not_qms(self):
        code = (ROOT/'run_b12c_bgr_60k.py').read_text()
        self.assertNotIn('qms_token_selector', code)
        self.assertIn("if not k.startswith('QMS_')", code)

if __name__ == '__main__':
    unittest.main()
