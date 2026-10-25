from pathlib import Path

import numpy as np
import pytest

from scripts.reproduce_tables import align_predictions, build_outputs, load_predictions, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_archived_gcs_paper_results():
    folder = ROOT / "results" / "predictions" / "gcs"
    data = {model: load_predictions(folder / f"{model}.npz")
            for model in ("streamcdf", "persistence", "crowdmac", "holt-last", "holt-uniform")}
    summary = summarize(data)
    assert summary["clips"] == 1448
    assert np.allclose(summary["quartiles"], [8.03250885, 10.7682991, 15.26572418])
    all_rows = {row["model"]: row for row in summary["rows"] if row["stratum"] == "all"}
    assert all_rows["Stream-CDF"]["D_kappa"] == pytest.approx(256.2685874)
    assert all_rows["persistence"]["countMAE"] == pytest.approx(5.9204, abs=5e-5)
    assert all_rows["CrowdMAC"]["ADJS"] == pytest.approx(0.09321, abs=5e-6)
    for stratum in ("flat", "moderate", "dynamic", "surge"):
        rows = [row for row in summary["rows"] if row["stratum"] == stratum]
        assert min(rows, key=lambda row: row["D_kappa"])["model"] == "Stream-CDF"
        assert sorted(rows, key=lambda row: row["ADJS"])[3]["model"] == "Stream-CDF"


def test_complete_row_reordering_preserves_results():
    path = ROOT / "results" / "predictions" / "gcs" / "streamcdf.npz"
    arrays = load_predictions(path)
    reverse = {key: value[::-1] for key, value in arrays.items()}
    result = align_predictions({"streamcdf": arrays, "persistence": reverse})
    assert np.array_equal(result["streamcdf"]["pred"], result["persistence"]["pred"])


def test_missing_or_ambiguous_clips_are_rejected():
    data = {"gt": np.asarray([[1, 2], [3, 4]], dtype=float)}
    with pytest.raises(ValueError, match="clip sets differ"):
        align_predictions({"a": data, "b": {"gt": data["gt"][:1]}})
    with pytest.raises(ValueError, match="Duplicate reference"):
        align_predictions({"a": {"gt": np.ones((2, 2))}})


def test_tracked_tables_are_reproducible():
    for name, content in build_outputs(ROOT / "results" / "predictions").items():
        assert (ROOT / "results" / "tables" / name).read_text(encoding="utf-8") == content
