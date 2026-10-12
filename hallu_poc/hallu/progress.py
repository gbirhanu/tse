"""Small text progress bars, including when launched as a Colab subprocess."""
import sys

from tqdm import tqdm


def progress(iterable=None, *, total=None, initial=0, desc, unit='items', disable=False):
    # Do not auto-disable for non-TTY output: Colab runs poc.py in a subprocess.
    # tqdm excludes the initial count from its rate estimate after a resume.
    return tqdm(iterable, total=total, initial=initial, desc=desc, unit=unit,
                disable=disable, file=sys.stderr, ascii=True, mininterval=1.0,
                bar_format='{desc}: {percentage:3.0f}%|{bar:12}| '
                           '{n_fmt}/{total_fmt} {unit} [{elapsed} elapsed, ETA {remaining}]')
