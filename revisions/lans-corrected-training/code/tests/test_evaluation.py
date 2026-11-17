"""Integrity and reporting tests; no research experiments run here."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
_spec = importlib.util.spec_from_file_location('corrected_report_tests', CODE / 'report.py')
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)
runner = report.runner
evaluation = report.evaluation


def protocol(root):
    data = {}
    for stage, (seeds, scales) in evaluation.PLANS.items():
        data[stage + '_seeds'] = list(seeds)
        data[stage + '_scales'] = [list(scale) for scale in scales]
    path = root / 'results/protocol.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return runner.digest(path)


def checkpoints(root):
    files = runner.expected_checkpoints(root)
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    (root / 'results/checkpoint_manifest.json').write_text(json.dumps({
        'checkpoints': {name: runner.digest(path) for name, path in files.items()}}))
    return files


def failed_row(kind, seed, case):
    return dict(kind=kind, train_seed=seed, **case, steps=1, completed='False',
                rmse_penalized=float(np.sqrt((.04 + 999 * 6.25) / 1001)),
                rmse_observed=float(np.sqrt(.04 / 2)), optimizer_failures=0)


class EvaluationTests(unittest.TestCase):
    def test_plans_and_job_coverage(self):
        self.assertEqual(len(runner.jobs()), 21)
        self.assertEqual(len(runner.expected_checkpoints()), 18)
        for stage, count in [('development', 84), ('test', 140)]:
            cases = list(evaluation.cases(stage))
            self.assertEqual(len(cases), count)
            self.assertEqual(len({report.case_key(case) for case in cases}), count)
            self.assertEqual(len({(c['path'], c['delay'], c['mass']) for c in cases}), 14)
        self.assertEqual(set(evaluation.PLANS['development'][0]) & set(evaluation.PLANS['test'][0]), set())

    def test_manifest_missing_policy_and_changed_policy_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'study'
            protocol(root)
            with self.assertRaises(FileNotFoundError):
                runner.preflight('test', root)
            files = checkpoints(root)
            self.assertEqual(runner.preflight('test', root)['status'], 'running')
            next(iter(files.values())).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.preflight('test', root)
            self.assertFalse((root / 'results/test/campaign.json').exists())

    def test_preflight_preserves_partial_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'study'
            protocol(root)
            checkpoints(root)
            out = root / 'results/development'
            out.mkdir()
            log = out / 'corrected_lans_82.log'
            log.write_text('failed startup evidence')
            with self.assertRaises(FileExistsError):
                runner.preflight('development', root)
            self.assertEqual(log.read_text(), 'failed startup evidence')

    def test_raw_metric_recomputation_detects_changed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.npz'
            np.savez_compressed(path, trace=np.zeros((1, 13)), initial_error=.2)
            row = failed_row('corrected_lans', 82, next(evaluation.cases('development')))
            self.assertEqual(len(report.verify_trace(path, row)), 64)
            row['rmse_penalized'] = 0.
            with self.assertRaisesRegex(ValueError, 'does not match raw trace'):
                report.verify_trace(path, row)

    def test_duplicate_cases_are_rejected_even_with_right_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'study'
            protocol(root)
            out = root / 'results/development'
            out.mkdir()
            row = failed_row('corrected_lans', 82, next(evaluation.cases('development')))
            report.write_csv(out / 'corrected_lans_82.csv', [row] * 84)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                report.load_campaign(root, 'development')

    def test_complete_development_report_retains_all_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'study'
            protocol_hash = protocol(root)
            files = checkpoints(root)
            out = root / 'results/development'
            (out / 'traces').mkdir(parents=True)
            for kind, seed in runner.jobs():
                rows = [failed_row(kind, seed, case) for case in evaluation.cases('development')]
                for row in rows:
                    np.savez_compressed(out / 'traces' / (report.trace_key(row) + '.npz'),
                                        trace=np.zeros((1, 13)), initial_error=.2)
                report.write_csv(out / f'{kind}_{seed}.csv', rows)
                path = evaluation.checkpoint(kind, seed)
                if path is not None:
                    import os
                    path = root / os.path.relpath(path, evaluation.ROOT)
                marker = dict(status='completed', episodes=84, stage='development', kind=kind,
                              seed=seed, protocol_sha256=protocol_hash,
                              checkpoint_sha256=runner.digest(path) if path else None)
                (out / f'{kind}_{seed}.run.json').write_text(json.dumps(marker))
            (out / 'corrected_lans_82.log').write_text('synthetic fixture only')
            with redirect_stdout(io.StringIO()):
                audit = report.generate(root, 'development')
            self.assertEqual(audit['episodes'], 1764)
            self.assertEqual(audit['raw_traces'], 1764)
            self.assertEqual(audit['completed'], 0)
            self.assertEqual(len(audit['checkpoint_sha256']), 18)
            self.assertEqual(len(audit['log_sha256']), 1)
            aggregate = json.loads((out / 'aggregate.json').read_text())
            self.assertEqual(aggregate['corrected_lans']['completed'], 0)
            self.assertEqual(len(report.read_csv(out / 'summary.csv')), 126)
            self.assertEqual(len(report.read_csv(out / 'paired_differences.csv')), 112)
            self.assertIn('including failures', (out / 'REVIEW.md').read_text())
            self.assertFalse((root / 'REVIEW.md').exists())
            # An unaccounted trace cannot be silently mixed into this campaign.
            (out / 'traces/extra.npz').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'Unaccounted'):
                report.load_campaign(root, 'development')


if __name__ == '__main__':
    unittest.main()
