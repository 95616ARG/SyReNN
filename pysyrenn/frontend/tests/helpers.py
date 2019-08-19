"""Helper methods for the other tests.
"""
import sys
import numpy as np
import pytest
import coverage
import os
# We need to do this here, otherwise it won't catch method/class declarations.
# Also, helpers imports should be before all other local imports.
cov = coverage.Coverage(data_file="%s/coverage.cov" % os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
cov.start()

def main(script_name, file_name):
    """Test runner that supports Bazel test and the coverage_report.sh script.

    Tests should import heplers before importing any other local scripts, then
    call main(__file__) after declaring their tests.
    """
    if script_name != "__main__":
        return
    exit_code = pytest.main([file_name, "-s"])
    cov.stop()
    cov.save()
    sys.exit(exit_code)
