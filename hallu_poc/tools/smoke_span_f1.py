"""CPU check for CE + character soft-F1 on 300 train / 120 test answers.

Creates a tiny ModernBERT, trains matched controls, verifies completed-run skips,
fits decoding on dev, and computes paired comparisons. These are smoke scores only.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--work-dir', default='work_test/span_f1')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir).resolve()
    data = Path(args.data_dir).resolve()
    tiny = work / 'tiny'
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false'}
    def command(*parts):
        subprocess.run([sys.executable, '-u', *map(str, parts)], cwd=project, env=env, check=True)
    if not (tiny / 'config.json').exists():
        command(project / 'tools' / 'make_tiny_modernbert.py', '--data-dir', data, '--out', tiny)
    common = ['--data-dir', str(data), '--work-dir', str(work), '--limit', '300', '--test-limit', '120',
              '--device', 'cpu', '--no-fp16', '--max-length', '1024', '--batch-tokens', '1024']
    def run(*parts):
        command(project / 'poc.py', *parts, *common)
    for name, weight in [('tiny_ce', '0'), ('tiny_f1w020', '0.2')]:
        train = ['train', '--name', name, '--base', tiny, '--targets', 'hard', '--seed', '42',
                 '--epochs', '1', '--lr', '1e-3', '--checkpoint-every', '25', '--span-f1-weight', weight]
        run(*train)
        run(*train)
        comparison = ['--against', 'tiny_ce:tuned'] if weight != '0' else []
        run('levers', '--name', name, '--fit', 'dev', *comparison)
    run('report', '--compare', 'tiny_ce', 'tiny_f1w020')


if __name__ == '__main__':
    main()
