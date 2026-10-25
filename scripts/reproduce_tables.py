"""Reconstruct paper tables from numeric per-clip archives, using only NumPy."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "streamcdf": "Stream-CDF",
    "persistence": "persistence",
    "crowdmac": "CrowdMAC",
    "holt-last": "Holt + last",
    "holt-uniform": "Holt + uniform",
}
METRICS = ("d_kappa_per_clip", "transport_per_clip", "unmatched_per_clip", "adjs_per_clip")


def load_predictions(path):
    """Read only numeric arrays; never deserialize Python objects."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("gt", "pred", "obs", *METRICS)}
    gt, pred, obs = (data[key] for key in ("gt", "pred", "obs"))
    if gt.ndim != 2 or gt.shape[1] != 12 or pred.shape != gt.shape:
        raise ValueError(f"{path}: expected matching (N, 12) future count arrays")
    if obs.shape != (len(gt), 8) or not len(gt):
        raise ValueError(f"{path}: expected nonempty (N, 8) observation array")
    for key, value in data.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{path}: non-finite or nonnumeric {key}")
        if key in METRICS and value.shape != (len(gt),):
            raise ValueError(f"{path}: invalid per-clip metric shape for {key}")
    if not np.allclose(data[METRICS[0]], data[METRICS[1]] + 8 * data[METRICS[2]],
                       rtol=1e-7, atol=1e-6):
        raise ValueError(f"{path}: D_kappa != transport + 8 * unmatched")
    return data


def align_predictions(data):
    """Match every clip by its future-count signature, rejecting ambiguity."""
    reference = next(iter(data.values()))["gt"]
    keys = lambda gt: [tuple(row) for row in np.round(gt.astype(np.float64), 3)]
    reference_keys = keys(reference)
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("Duplicate reference ground-truth signatures; clip IDs required")
    aligned = {}
    for model, arrays in data.items():
        model_keys = keys(arrays["gt"])
        if len(set(model_keys)) != len(model_keys):
            raise ValueError(f"{model}: duplicate ground-truth signatures; clip IDs required")
        if set(model_keys) != set(reference_keys):
            raise ValueError(f"{model}: clip sets differ; refusing a partial comparison")
        lookup = {key: index for index, key in enumerate(model_keys)}
        order = np.asarray([lookup[key] for key in reference_keys])
        aligned[model] = {key: value[order] for key, value in arrays.items()}
        if not np.allclose(aligned[model]["gt"], reference, rtol=0, atol=0.001):
            raise ValueError(f"{model}: ground truth disagrees after alignment")
    return aligned


def summarize(data):
    data = align_predictions(data)
    gt = next(iter(data.values()))["gt"]
    span = np.ptp(gt, axis=1)
    q25, q50, q75 = np.quantile(span, [0.25, 0.5, 0.75])
    strata = {
        "flat": span <= q25,
        "moderate": (span > q25) & (span <= q50),
        "dynamic": (span > q50) & (span <= q75),
        "surge": span > q75,
        "all": np.ones(len(gt), dtype=bool),
    }
    rows = []
    for stratum, select in strata.items():
        if not select.any():
            raise ValueError(f"Empty {stratum} stratum")
        for model, arrays in data.items():
            rows.append({
                "stratum": stratum, "n": int(select.sum()), "model": MODELS[model],
                "countMAE": float(np.abs(arrays["gt"][select] - arrays["pred"][select]).mean()),
                "D_kappa": float(arrays["d_kappa_per_clip"][select].mean()),
                "unmatched": float(arrays["unmatched_per_clip"][select].mean()),
                "transport": float(arrays["transport_per_clip"][select].mean()),
                "ADJS": float(arrays["adjs_per_clip"][select].mean()),
            })
    return {"clips": len(gt), "kappa": 8, "quartiles": [float(q25), float(q50), float(q75)],
            "rows": rows}


def markdown(dataset, summary):
    lines = [f"# {dataset.upper()} results", "",
             f"Clips: {summary['clips']}. Kappa: 8. Lower is better.", "",
             "Strata use future-count range, with quartiles "
             + ", ".join(f"{value:.4f}" for value in summary["quartiles"]) + " people.", "",
             "ADJS follows CrowdMAC's reference mixture computation, not standard normalized JS.",
             "Unmatched mass depends on both count and placement.", "",
             "| Stratum | n | Model | countMAE | D_kappa | Unmatched (people) | Transport | ADJS |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['stratum']} | {row['n']} | {row['model']} | "
                     f"{row['countMAE']:.2f} | {row['D_kappa']:.1f} | "
                     f"{row['unmatched']:.2f} | {row['transport']:.1f} | {row['ADJS']:.4f} |")
    lines += ["", "Reconstructed from stored per-clip arrays; no training or full-map transport was rerun.", ""]
    return "\n".join(lines)


def build_outputs(predictions_root):
    manifest = json.loads((predictions_root / "manifest.json").read_text(encoding="utf-8"))
    outputs = {}
    for dataset in ("gcs", "indoor"):
        files = manifest["datasets"][dataset]
        data = {}
        for model in MODELS:
            info = files[model]
            path = predictions_root / dataset / f"{model}.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            data[model] = load_predictions(path)
        summary = summarize(data)
        outputs[f"{dataset}.md"] = markdown(dataset, summary)
        outputs[f"{dataset}.json"] = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(summary["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["rows"])
        outputs[f"{dataset}.csv"] = buffer.getvalue()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--check", action="store_true", help="Validate tracked outputs without writing")
    args = parser.parse_args()
    outputs = build_outputs(args.predictions_root)
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = args.output_dir / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Out-of-date result: {path}; run scripts/reproduce_tables.py")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(f"{'Validated' if args.check else 'Wrote'} {len(outputs)} result files for GCS and indoor.")


if __name__ == "__main__":
    main()
