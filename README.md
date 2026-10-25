# Crowds Are Mass

**Rethinking Evaluation for Crowd Density Forecasting**  
Khalid AlAmoudi and Alala Bahamid

Research code and [LaTeX manuscript](paper.tex) studying how evaluation changes
when crowd density maps retain their headcount. Normalized divergence can miss
large proportional count errors; count error alone misses placement. The paper
evaluates forecasts using unbalanced optimal transport:

```text
D_kappa = transport cost + kappa * unmatched mass
```

Unmatched mass depends on placement as well as the difference in totals. The
reported reference JS computation follows CrowdMAC's mixture-before-normalization
implementation; it is not standard JS between independently normalized maps.

## Reproduce the tables in minutes

Python 3.10 or newer is recommended. This workflow needs only NumPy and runs on
CPU. It uses the bundled numeric per-clip predictions and saved metric values;
it does not retrain models or recompute transport from density maps.

```sh
python -m pip install -r requirements-results.txt
python scripts/reproduce_tables.py
```

This writes `results/tables/gcs.md`, `results/tables/indoor.md`, and machine-readable
CSV/JSON summaries. The script validates clip alignment, finite arrays, and the
transport decomposition before reporting results. See
[reproducibility details](docs/REPRODUCIBILITY.md).

## Main GCS result

Lower is better in all columns. The same 1,448 clips and predictions are scored.

| Model | countMAE (people) | Reference ADJS | D_kappa |
|---|---:|---:|---:|
| Stream-CDF | 6.16 | 0.1150 | **256.3** |
| Persistence | **5.92** | **0.0646** | 262.8 |
| CrowdMAC | 12.29 | 0.0932 | 392.8 |
| Holt + last | 10.75 | 0.0667 | 278.4 |
| Holt + uniform | 10.75 | 0.2844 | 868.3 |

Stream-CDF moves from fourth under reference ADJS to first under transport in
every GCS dynamics stratum. Indoors, the transport differences are smaller and
persistence slightly wins pooled: 81.5 versus Stream-CDF's 82.3. These are
single-run results, not evidence of a universal ranking.

## Repository layout

```text
paper.tex                   Corrected manuscript source
streamcdf/                  Canonical model, transport solver, reference divergences
scripts/                    Training, evaluation, baselines, conversion, table reproduction
configs/                    Recorded settings for the paper's Stream-CDF runs
third_party/crowdmac/        Supplied CrowdMAC snapshot and upstream MIT license
results/predictions/        Numeric-only per-clip artifacts for GCS and indoor
results/tables/             Reconstructed tables and summaries
docs/                       Data instructions, provenance, reproducibility limits
tests/                      CPU checks for metrics, causality, alignment, and results
```

The old local backup folders are retained on the author's machine and excluded
from Git. Raw datasets, checkpoint weights, and training logs are also excluded.

## Train or evaluate models

Install a matching PyTorch/torchvision build for your hardware, then:

```sh
python -m pip install -r requirements.txt
```

Prepare the annotation CSVs as described in [DATA.md](docs/DATA.md). GCS data and
checkpoint downloads are not supplied here. Model training is intended for a
CUDA GPU; CPU training is allowed but slow.

The following commands explicitly reproduce the recorded Stream-CDF settings:

```sh
python scripts/train.py --dataset gcs --horizon short --epochs 20 --eps 2.0 --pool 2 --iters 30 --dim 256 --depth 4 --heads 4 --dropout 0 --temporal absolute --out_act softplus --out_gain 10 --delta_bound 50 --clip 750 --debias 1 --seed 0
python scripts/train.py --dataset indoor --horizon short --epochs 40 --eps 2.0 --pool 2 --iters 30 --dim 256 --depth 4 --heads 4 --dropout 0 --temporal absolute --out_act softplus --out_gain 10 --delta_bound 50 --clip 750 --debias 1 --seed 0
```

Both use **8 observed frames → 12 future frames**. On GCS this corresponds to
6.4 seconds observed and 9.6 seconds forecast. Indoor sampling differs; do not
assume the same elapsed time without verifying the recording's native rate.

```sh
python scripts/evaluate.py --ckpt runs/YOUR_FINAL_CHECKPOINT.pt --dataset gcs
python scripts/persistence.py --dataset gcs --horizon short
python scripts/count_baselines.py --dataset gcs --horizon short --count holt --spatial last
python scripts/count_baselines.py --dataset gcs --horizon short --count holt --spatial uniform
python scripts/evaluate_crowdmac.py --ckpt checkpoints/YOUR_CROWDMAC_CHECKPOINT.pth --dataset gcs
```

Training uses a debiased objective with pooled maps. Reported transport evaluation
uses raw transport-plus-unmatched cost, full resolution, float64, epsilon 0.25,
and 60 iterations. `--fast` changes that precision and must not be mixed with the
reference results. Outputs go to `runs/`, which is ignored by Git.

## Checks

```sh
python -m pip install pytest
python -m pytest -q
```

The table checks need NumPy; solver and model checks also need PyTorch. GitHub
Actions runs both on CPU. Full training and density-map evaluation additionally
need the training dependencies, original data, and appropriate compute.

## Manuscript and release status

The manuscript is included without further content changes. Its referenced
`figs/`, `orcid.pdf`, and `arxiv.sty` assets were absent from the supplied
workspace, so this snapshot cannot yet reproduce the paper PDF. No pretrained
weights or new experimental results are claimed. See the
[release checklist](docs/RELEASE.md) for the remaining artifacts.

Use [CITATION.cff](CITATION.cff) for attribution. No DOI or publication venue is
invented. A reuse license for the original materials has not yet been selected;
see [LICENSE.md](LICENSE.md). CrowdMAC retains its separate
[upstream MIT license](third_party/crowdmac/LICENSE) and
[attribution](THIRD_PARTY_NOTICES.md).
