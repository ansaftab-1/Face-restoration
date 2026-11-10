#!/usr/bin/env python3

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
_venv_scripts = os.path.dirname(os.path.abspath(sys.executable))
if _venv_scripts not in os.environ.get('PATH', ''):
	os.environ['PATH'] = _venv_scripts + os.pathsep + os.environ.get('PATH', '')

from facefusion import conda, core

if __name__ == '__main__':
	conda.setup()
	core.cli()

