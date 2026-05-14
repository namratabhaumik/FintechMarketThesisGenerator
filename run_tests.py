#!/usr/bin/env python
"""Simple test runner - just run all tests."""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pytest", "tests/unit/", "-q"], check=False)
