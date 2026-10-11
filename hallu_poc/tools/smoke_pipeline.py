"""CPU integration check on 300 train / 120 test answers; scores are not research results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--work-dir', default='work_test/smoke')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    tiny = work / 'tiny'
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false'}

    def command(*parts):
        subprocess.run([sys.executable, '-u', *map(str, parts)], cwd=project,
                       check=True, env=env)

    if not (tiny / 'config.json').exists():
        command(project / 'tools' / 'make_tiny_modernbert.py', '--data-dir', data_dir, '--out', tiny)
    common = ['--data-dir', str(data_dir), '--work-dir', str(work),
              '--limit', '300', '--test-limit', '120', '--device', 'cpu', '--no-fp16',
              '--max-length', '1024', '--batch-tokens', '1024']

    def run(*parts):
        command(project / 'poc.py', *parts, *common)

    run('predict', '--name', 'tiny_reference', '--model', tiny, '--with-dev')
    run('levers', '--name', 'tiny_reference', '--fit', 'fixed')
    for targets in ('hard', 'soft'):
        train = ['train', '--name', f'tiny_{targets}', '--base', tiny, '--targets', targets,
                 '--seed', '43', '--epochs', '1', '--lr', '1e-3', '--checkpoint-every', '25']
        run(*train)
        run(*train)  # Finished run must be validated and skipped.
        run('levers', '--name', f'tiny_{targets}', '--fit', 'dev')
    run('ensemble', '--name', 'tiny_ensemble', '--members', 'tiny_hard', 'tiny_soft',
        '--fit', 'dev', '--against', 'tiny_reference:baseline')
    run('ensemble', '--name', 'tiny_ensemble', '--members', 'tiny_hard', 'tiny_soft',
        '--fit', 'dev', '--against', 'tiny_reference:baseline')
    run('report', '--compare', 'tiny_hard', 'tiny_soft')


if __name__ == '__main__':
    main()
