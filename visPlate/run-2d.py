import os
from lotus import run
from pathlib import Path


def run_sim():
    run(16, f'{cwd}/vis')


if __name__ == "__main__":
    cwd = Path.cwd()
    run_sim()

