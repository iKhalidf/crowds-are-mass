# Data preparation

The repository does not redistribute raw annotations, videos, or trained weights.
Obtain the appropriate datasets from their original providers and follow their
access and reuse conditions.

## Annotation layout

The supplied CrowdMAC adapter expects CSVs grouped by dataset and split:

```text
data/
  gcs/train/<block>.csv
  gcs/test/<block>.csv
  indoor/train/<recording>.csv
  indoor/test/<recording>.csv
  counts/counts_train.npz     # optional GCS count diagnostics
  counts/counts_test.npz      # optional GCS count diagnostics
```

CSV columns, in order, are `trackId,frame,x,y,sceneId`. Coordinates are in the
source image coordinate system registered in
`third_party/crowdmac/img_size.py`. Density maps are generated on an 80×80 grid
with sigma 3; the model uses a storage scale of 100 density units per person.
The adapter uses the first four columns; identities are not required for the
density representation.

The recorded GCS split holds out eight blocks and yields 1,448 test clips at
stride one. GCS annotation CSVs and a definitive block split manifest were not
present in the supplied archive; do not invent a replacement split and claim
the same experiment. Indoor CSVs existed in the local backup but are excluded
from this repository pending dataset redistribution decisions. The test
recordings recorded in the converter are `acs_ec_recording_1771860689` and
`acs_eg_recording_1771862331`.

## External data location

By default, scripts use the repository's `data/` directory. Set an absolute
`CROWD_DATA_ROOT` to use another location:

```powershell
$env:CROWD_DATA_ROOT = 'D:/crowd-data'
```

```sh
export CROWD_DATA_ROOT=/path/to/crowd-data
```

## Frame sampling

The short task is 8 observed plus 12 forecast frames. GCS is already sampled at
1.25 fps; the loader uses step 1. The indoor adapter uses step 10 over source
frame indices. Its native frame rate is not established by this archive; the
same frame counts do not establish the same time horizon across datasets.

The supplied loader omits windows containing unannotated frames and shuffles
its trajectory list during construction, including for test data. Reports must
align clips by their saved ground truth instead of assuming common row indices.

`scripts/convert_dots.py --help` describes optional MOT/indoor dot conversion.
Set `CROWD_ANNOTATIONS_ROOT` to the parent of the original dot-annotation folders.
The converter preserves original frame numbers; loader subsampling defines the
actual windows. Conversion alone does not establish a new paper result.
