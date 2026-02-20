#!/usr/bin/env python
"""Simple test runner - just run all tests."""

import subprocess
import sys

subprocess.run(["python", "-m", "pytest", "tests/unit/", "-q"], check=False)
