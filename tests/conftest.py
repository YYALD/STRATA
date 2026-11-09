import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "methods"), os.path.join(ROOT, "examples")):   # strata/, methods, examples
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def example(tmp_path_factory):
    """A small synthetic profile: 36 stations over 20-160 m (the 64-point profile window needs 126 m), 16 gates, two
    anomalies."""
    import make_synthetic_profile as msp
    out = tmp_path_factory.mktemp("example")
    return msp.write_example(str(out), n_stations=36, n_gates=16, z0=20.0, z1=160.0,
                             anomalies=((70.0, 5.0, 0.10, 0.3), (125.0, 7.0, -0.08, 1.5)))
