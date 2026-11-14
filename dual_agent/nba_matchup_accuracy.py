"""Evaluate saved NBA history without writing to Supabase or promoting a model.

Run: python -m dual_agent.nba_matchup_accuracy --output nba_accuracy_report.json
This is a retrospective diagnostic, not an untouched prospective benchmark.
"""
import argparse
import copy
import json
import os
import re
import traceback
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

from . import nba_unified_winner as winner
from .nba_pregame_snapshots import Client


class StorageReadsOnly:
    """Allow object downloads and listings only; reject all network mutations."""
    def __init__(self, session, project_url, bucket):
        self.session = session
        self.origin = urlparse(project_url).netloc
        self.bucket = bucket

    def request(self, method, url, **kwargs):
        parsed = urlparse(url)
        method = method.upper()
        allowed = parsed.scheme == 'https' and parsed.netloc == self.origin
        if method == 'GET':
            root = '/storage/v1/object/authenticated/' + self.bucket + '/'
            allowed = allowed and parsed.path.startswith((
                root + 'nba/', root + 'sports/connected_winners_v1/nba/'))
        elif method == 'POST':
            prefix = (kwargs.get('json') or {}).get('prefix', '')
            allowed = allowed and parsed.path == '/storage/v1/object/list/' + self.bucket
            allowed = allowed and prefix.startswith(('nba/', 'sports/connected_winners_v1/nba/'))
        else:
            allowed = False
        if not allowed:
            raise RuntimeError('Accuracy evaluation blocked a non-read storage request')
        # Do not follow redirects with project credentials to another destination.
        kwargs['allow_redirects'] = False
        return self.session.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)


class EvaluationStore:
    """Preserve production reads and experiment history; divert writes to memory."""
    def __init__(self, source):
        self.source = source
        self.env, self.bucket = source.env, source.bucket
        self.http = source.http
        self.overlay = {}
        self.downloads = 0
        self.last_read = None

    def headers(self):
        return self.source.headers()

    def read(self, path, **kwargs):
        if path in self.overlay:
            return copy.deepcopy(self.overlay[path])
        if path == winner.ROOT + 'controller_readiness.json.gz':
            # A diagnostic should actually evaluate, not exit on unchanged inputs.
            return None
        self.last_read = path
        self.downloads += 1
        if self.downloads % 100 == 0:
            print(f'Read {self.downloads} saved objects for evaluation', flush=True)
        return self.source.read(path, **kwargs)

    def write(self, path, value, **kwargs):
        self.overlay[path] = copy.deepcopy(value)


def evaluate(source):
    original_http = source.http
    source.http = StorageReadsOnly(original_http, source.env['SUPABASE_URL'], source.bucket)
    sandbox = EvaluationStore(source)
    try:
        report = copy.deepcopy(winner.train(sandbox))
        report['would_activate_under_training_rules'] = bool(report.pop('activated', False))
        report['activated'] = False
        report['live_model_changed'] = False
        report['evaluation_only'] = True
        report['storage_writes'] = 0
        report['saved_objects_read'] = sandbox.downloads
        report['interpretation'] = (
            'Retrospective candidate versus reconstructed base model on chronological later games. '
            'Configuration selection uses validation only. Existing experiment history is retained. '
            'This diagnostic does not reserve a new test window or promote any model; repeated '
            'inspection means these results must not be described as an untouched benchmark.')
        return report
    except Exception as exc:
        exc.nba_evaluation_diagnostics = {
            "saved_objects_read": sandbox.downloads, "last_read_path": sandbox.last_read}
        raise
    finally:
        source.http = original_http


def failure_report(exc):
    """Retain useful failure evidence without exposing credentials or request URLs."""
    message = str(exc)
    for name, value in os.environ.items():
        if value and (any(token in name.upper() for token in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))
                      or name == 'SUPABASE_URL'):
            message = message.replace(value, '[redacted]')
    message = re.sub(r'https?://[^\s\"\']+', '[redacted-url]', message)
    frames = [{'file': Path(frame.filename).name, 'function': frame.name, 'line': frame.lineno}
              for frame in traceback.extract_tb(exc.__traceback__)]
    return {'evaluation_only': True, 'activated': False, 'live_model_changed': False,
            'status': 'evaluation_failed', 'error_type': type(exc).__name__,
            'error_message': message[:1000], 'error_trace': frames,
            'dependencies': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
            **getattr(exc, 'nba_evaluation_diagnostics', {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='nba_accuracy_report.json')
    args = parser.parse_args()
    destination = Path(args.output)
    try:
        env = dict(os.environ)
        # Client requires this setting, but this read-only evaluation never calls BDL.
        env['BALLDONTLIE_API_KEY'] = 'unused-offline-evaluation'
        result = evaluate(Client(env=env))
    except Exception as exc:
        result = failure_report(exc)
        destination.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return 1
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
