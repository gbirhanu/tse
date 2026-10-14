"""CPU check for Part E on about 300 train / 120 test answers; scores mean nothing.

Step 1: train two tiny seeds, average their weights (soup), predict, fit on dev,
        and build the matching output-average ensemble for comparison.
Step 2: train from a tiny 3-label NLI-shaped checkpoint.
Step 3: make synthetic summaries (needs spaCy + en_core_web_sm) and train with them.
Every command runs twice where a finished step must be verified and skipped.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--work-dir', default='work_test/part_e')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir).resolve()
    data = Path(args.data_dir).resolve()
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false'}

    def command(*parts):
        subprocess.run([sys.executable, '-u', *map(str, parts)], cwd=project, env=env, check=True)

    tiny, tiny_nli = work / 'tiny', work / 'tiny_nli'
    if not (tiny / 'config.json').exists():
        command(project / 'tools' / 'make_tiny_modernbert.py', '--data-dir', data, '--out', tiny)
    if not (tiny_nli / 'config.json').exists():
        command(project / 'tools' / 'make_tiny_modernbert.py', '--data-dir', data, '--out', tiny_nli, '--nli')
    common = ['--data-dir', str(data), '--work-dir', str(work), '--limit', '300', '--test-limit', '120',
              '--device', 'cpu', '--no-fp16', '--max-length', '1024', '--batch-tokens', '1024']

    def run(*parts):
        command(project / 'poc.py', *parts, *common)

    def train(name, base, seed, *extra):
        parts = ['train', '--name', name, '--base', base, '--targets', 'hard', '--seed', seed,
                 '--epochs', '1', '--lr', '1e-3', '--checkpoint-every', '25', *extra]
        run(*parts)
        run(*parts)

    # Step 1
    for seed in ('43', '44'):
        train(f'tiny_s{seed}', tiny, seed)
        run('levers', '--name', f'tiny_s{seed}', '--fit', 'dev')
    for _ in range(2):
        run('soup', '--name', 'tiny_soup', '--members', 'tiny_s43', 'tiny_s44')
    run('levers', '--name', 'tiny_soup', '--fit', 'dev', '--against', 'tiny_s43:tuned')
    run('ensemble', '--name', 'tiny_ens', '--members', 'tiny_s43', 'tiny_s44', '--fit', 'dev')
    # Step 2
    train('tiny_nli_s43', tiny_nli, '43')
    run('levers', '--name', 'tiny_nli_s43', '--fit', 'dev', '--against', 'tiny_s43:tuned')
    # Step 3
    run('synth', '--name', 'tiny_syn')
    run('synth', '--name', 'tiny_syn')
    train('tiny_syn_s43', tiny, '43', '--extra-train', 'tiny_syn')
    run('levers', '--name', 'tiny_syn_s43', '--fit', 'dev', '--against', 'tiny_s43:tuned')
    run('report', '--runs', 'tiny_s43', 'tiny_s44', 'tiny_soup', 'tiny_ens', 'tiny_nli_s43',
        'tiny_syn_s43', '--compare', 'tiny_s43', 'tiny_syn_s43')


if __name__ == '__main__':
    main()
