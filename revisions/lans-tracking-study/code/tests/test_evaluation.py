"""Evaluation regression checks; all generated files stay in temporary directories."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import evaluate
import report
import run_eval


class EvaluationTests(unittest.TestCase):
    def test_unknown_kind_is_rejected_by_loader_and_cli(self):
        with patch.object(evaluate.PPO, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'Unsupported controller'):
                evaluate.load('limted_lans', 72)
            load.assert_not_called()
        result = subprocess.run(
            [sys.executable, str(CODE / 'evaluate.py'), '--stage', 'test',
             '--kind', 'limted_lans'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_test_jobs_cover_every_report_reference(self):
        for selected in ['limited_ppo', 'rl_ppo', 'original_ppo']:
            jobs = set(run_eval.build_jobs(Path('/unused'), 'test', ppo_reference=selected))
            required = {(name, seed) for name, seeds in report.controller_spec(selected).items()
                        for seed in seeds}
            self.assertEqual(jobs, required)
            self.assertIn(('matched_mpc', -1), jobs)
        with self.assertRaises(ValueError):
            run_eval.build_jobs(Path('/unused'), 'test', only_limited=True)
        with self.assertRaises(ValueError):
            run_eval.build_jobs(Path('/unused'), 'test', ppo_reference='typo')

    def test_existing_csv_and_partial_trace_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            evaluate.ensure_fresh_output(out, 'limited_lans', 72)
            csv_file = out / 'limited_lans_72.csv'
            csv_file.write_text('original\n')
            with self.assertRaises(FileExistsError):
                evaluate.ensure_fresh_output(out, 'limited_lans', 72)
            evaluate.ensure_fresh_output(out, 'limited_lans', 72, overwrite=True)
            self.assertEqual(csv_file.read_text(), 'original\n')
            csv_file.unlink()
            (out / 'traces').mkdir()
            (out / 'traces/limited_lans_72_123_dlc.npz').touch()
            with self.assertRaises(FileExistsError):
                evaluate.ensure_fresh_output(out, 'limited_lans', 72)
            evaluate.ensure_fresh_output(out, 'limited_lans', 73)

    def test_report_with_unrestricted_ppo_selection(self):
        # A real alternate validation winner used to fail with KeyError because
        # the report never loaded its episodes. Exercise full report generation.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'results'
            (results / 'test').mkdir(parents=True)
            (results / 'validation_selection.json').write_text(json.dumps({'ppo_reference': 'rl_ppo'}))
            total = 0
            for kind, seeds in report.controller_spec('rl_ppo').items():
                for seed in seeds:
                    rows = [dict(kind=kind, train_seed=seed, **case, steps=1000,
                                 completed=True, rmse_penalized=.01,
                                 rmse_observed=.01, optimizer_failures=0)
                            for case in evaluate.cases('test')]
                    total += len(rows)
                    with (results / 'test' / f'{kind}_{seed}.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
            with patch.object(report, 'ROOT', root), patch.object(report, 'R', results), redirect_stdout(io.StringIO()):
                report.main()
            summary = json.loads((results / 'ranking_summary.json').read_text())
            self.assertIn('rl_ppo', summary['objective_matched'])
            self.assertEqual(total, 3500)
            self.assertIn('All3500 test episodes', (root / 'Tracking_Report.md').read_text())


if __name__ == '__main__':
    unittest.main()
