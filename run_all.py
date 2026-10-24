#!/usr/bin/env python3
"""Rebuild all main-text exhibits; optionally regenerate simulations from original seeds."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-simulation', action='store_true',
                        help='Regenerate all simulation figures from the original seeds.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    start = time.monotonic()
    commands = [
        ['code/empirical/run.py'],
        ['code/empirical/export_tables.py'],
        (['code/simulation/historical.py', '--workers', str(args.workers)]
         if args.fresh_simulation else ['code/simulation/run.py']),
    ]
    for script, *options in commands:
        subprocess.run([sys.executable, str(ROOT/script), *options], cwd=ROOT, check=True)
    print(f'Complete: {time.monotonic()-start:.1f}s. Results: {ROOT / "output"}')


if __name__ == '__main__':
    main()
