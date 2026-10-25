# Third-party code

`third_party/crowdmac/` contains the supplied local snapshot of the official
[CrowdMAC repository](https://github.com/Fujiry0/CrowdMAC), by Ryo Fujii,
Ryo Hachiuma, and Hideo Saito, associated with their WACV 2025 paper.

The upstream MIT license is preserved as `third_party/crowdmac/LICENSE`.
The precise upstream commit of the supplied snapshot is unknown; the source
hashes in `docs/source_manifest.json` identify the local files used here.
`third_party/crowdmac/UPSTREAM.json` records the upstream source and license.

The supplied snapshot already includes local GCS/MOT/indoor dataset registrations
and research configurations. The repository preparation additionally changes:

- `videomae.py`: resolve the data root from `CROWD_DATA_ROOT`, defaulting to this
  repository's `data/` directory.
- `utils_videomae.py`: extract dataset names with `pathlib` for Windows portability.

CrowdMAC acknowledges VideoMAE as a source for its implementation. Existing
source notices are retained. The original research code is packaged separately
in `streamcdf/` and `scripts/`.

```bibtex
@inproceedings{FUJII2025CrowdMAC,
  author = {Ryo Fujii and Ryo Hachiuma and Hideo Saito},
  title = {CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year = {2025}
}
```
