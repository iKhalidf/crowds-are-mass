"""Resolve local paths without changing the caller's working directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CROWD_DATA_ROOT", ROOT / "data")).resolve()
os.environ["CROWD_DATA_ROOT"] = str(DATA_ROOT)
CROWDMAC = (ROOT / "third_party" / "crowdmac").as_posix()
STREAMCDF = ROOT.as_posix()
COUNTS = (DATA_ROOT / "counts").as_posix()
for path in (str(ROOT), CROWDMAC):
    if path not in sys.path:
        sys.path.insert(0, path)
(ROOT / "runs").mkdir(exist_ok=True)
