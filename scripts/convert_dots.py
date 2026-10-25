"""
convert_dots.py -- turn MOT / IndoorCrowd dot annotations into the CSV format
CrowdMAC's get_trajectories_gcs already reads, so every existing script
(train_v5.py, persistence_baseline.py, crowdmac_eval.py, count_baselines.py,
ridge_baseline.py) works on the new datasets with only --dataset changed.

    python convert_dots.py --dataset mot
    python convert_dots.py --dataset indoor

Output layout, mirroring what get_trajectories() expects:

    <root>/<dataset>/train/<sceneId>.csv
    <root>/<dataset>/test/<sceneId>.csv

with columns  trackId,frame,x,y,sceneId  -- the GCS schema.

trackId IS FAKE, and that is fine: utils_videomae.py already writes
"'trackId'(fake)" for fdst, vscrowd, gcs and others. Density forecasting never
uses identity, only per-frame point sets, so unlinked dots lose nothing.

SPLIT BY WHOLE SEQUENCE, NEVER BY CLIP. get_trajectories_gcs slides a window
with stride 1, so neighbouring clips overlap by seq_len-1 frames. Splitting
clips at random would put near-duplicates on both sides and inflate every
score. Holding out entire sequences is the only honest option.

--------------------------------------------------------------------------
THE STEP PARAMETER -- read this before trusting any cross-dataset number
--------------------------------------------------------------------------
GCS is natively 1.25 fps, so its 20-frame clip spans 16 s (6.4 s observed,
9.6 s predicted). That is the forecasting task your GCS results describe.

MOT17 runs at 25-30 fps and IndoorCrowd's rate is unknown here. At step=1 a
20-frame MOT clip spans under a second -- a completely different problem, and
"the model beats persistence on MOT" would not be comparable to the GCS claim
in any meaningful sense.

--step subsamples frames to match the TIME horizon. --fps states the assumed
native rate and is recorded in the manifest so the assumption is visible
rather than buried. Defaults assume 25 fps -> step 20 -> 16 s clips, matching
GCS. CHANGE --fps IF YOU KNOW THE TRUE RATE.

Cost, measured on these annotations (stride=1, seq_len=20):

    step   MOT clips   IndoorCrowd clips   recordings surviving
      1      3542           3190                13 / 13
     10      2822           1418                 6 / 13
     20      2207            553                 4 / 13
     24      1967            334                 3 / 13

MOT tolerates any step. INDOORCROWD DOES NOT: most of its recordings are
under 500 frames, so a large step drops them entirely and "13 sequences"
silently becomes 3. Prefer reporting IndoorCrowd at a smaller step with the
shorter time horizon stated plainly, over a matched horizon computed from
three recordings.

--------------------------------------------------------------------------
IMAGE SIZE
--------------------------------------------------------------------------
generate_heatmap() scales BOTH axes by fmap_size[0] / orig_max_size[0] -- a
single factor, so it assumes a square canvas. Passing the true 1920x1080
would push y beyond the grid. This writes a SQUARE size equal to the larger
dimension (1920 for MOT, 1280 for IndoorCrowd), which preserves aspect ratio
and keeps every point inside the 80x80 map. The crowd then occupies the upper
band of the map rather than its full height -- the same treatment GCS gets
from its own square IMG_SIZES entry.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

from _paths import ROOT, DATA_ROOT
SRC = os.environ.get("CROWD_ANNOTATIONS_ROOT", str(ROOT / "annotations"))
DST = str(DATA_ROOT)

SPECS = {
    "mot": dict(
        root=f"{SRC}/MOT_dot_annotations/MOT_dot_annotations",
        layout="flat",              # <root>/<seq>/<frame>.txt
        img=1920,
        fps=25.0,
        # Density varies 3x across these four sequences (gt 66, gt_1 85,
        # gt_2 178, gt_3 227 people/frame), so the split is a real choice, not
        # a formality. gt_1 holds out ~41% of surviving clips and leaves train
        # spanning both the sparse and dense regimes; holding out gt_2 instead
        # puts the middle density in test and leaves train bimodal.
        test=["gt_1"],
    ),
    "indoor": dict(
        root=f"{SRC}/IndoorCrowd_dot_annotations/IndoorCrowd_dot_annotations",
        layout="nested",            # <root>/<location>/<recording>/<frame>.txt
        img=1280,
        fps=25.0,
        # hold out one recording per location so test covers all three scenes
        # Chosen AFTER seeing clip yields: the longest recordings are exactly
        # the ones that survive subsampling, so holding those out inverted the
        # split (82% of clips in test at step 20). This pair gives ~25% test
        # with both locations on both sides.
        # ie_central is a single 62-frame recording and drops out at every
        # step >= 4 -- IndoorCrowd is effectively TWO scenes, not three, and
        # they differ systematically in density (acs_ec 28-32 people/frame,
        # acs_eg 11-15). Any split here is partly a test of generalising
        # across scene, which belongs in the writeup.
        test=["acs_ec_recording_1771860689", "acs_eg_recording_1771862331"],
    ),
}


def frame_num(fn):
    digits = "".join(c for c in os.path.splitext(fn)[0] if c.isdigit())
    return int(digits) if digits else None


def read_dots(path):
    pts = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return pts


def sequences(spec):
    """Yields (sceneId, {frame_no: [(x, y), ...]})."""
    root = spec["root"]
    if spec["layout"] == "flat":
        seqs = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
        for s in seqs:
            yield s, os.path.join(root, s)
    else:
        for loc in sorted(os.listdir(root)):
            lp = os.path.join(root, loc)
            if not os.path.isdir(lp):
                continue
            for rec in sorted(os.listdir(lp)):
                rp = os.path.join(lp, rec)
                if os.path.isdir(rp):
                    yield rec, rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPECS))
    ap.add_argument("--step", type=int, default=None,
                    help="frame subsample. Default: round(fps / 1.25) so a "
                         "20-frame clip spans 16 s, matching GCS.")
    ap.add_argument("--fps", type=float, default=None,
                    help="ASSUMED native frame rate. Recorded in the manifest.")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    spec = SPECS[a.dataset]
    fps = a.fps if a.fps is not None else spec["fps"]
    step = a.step if a.step is not None else max(1, round(fps / 1.25))
    span = a.seq_len * step / fps

    print(f"{a.dataset}: assuming {fps:g} fps, step {step} -> "
          f"{a.seq_len}-frame clip spans {span:.1f}s "
          f"(GCS reference: 16.0s)")
    if abs(span - 16.0) > 2.0:
        print(f"  WARNING: clip span {span:.1f}s differs materially from GCS's "
              f"16.0s. Cross-dataset numbers describe different tasks.")

    out_root = os.path.join(a.dst, a.dataset)
    sizes, stats, kept = {}, [], defaultdict(int)

    for scene, path in sequences(spec):
        files = sorted(f for f in os.listdir(path) if f.endswith(".txt"))
        rows, counts = [], []
        for f in files:
            n = frame_num(f)
            if n is None:
                continue
            pts = read_dots(os.path.join(path, f))
            counts.append(len(pts))
            for i, (x, y) in enumerate(pts):
                rows.append((i + 1, n, x, y, scene))   # trackId is fake
        if not rows:
            continue

        n_frames = len(counts)
        usable = max(0, n_frames - a.seq_len * step + 1)
        split = "test" if scene in spec["test"] else "train"
        kept[split] += usable
        sizes[scene] = [spec["img"], spec["img"]]
        stats.append((scene, split, n_frames, usable,
                      sum(counts) / max(1, len(counts))))

        if not a.dry_run:
            d = os.path.join(out_root, split)
            os.makedirs(d, exist_ok=True)
            rows.sort(key=lambda r: (r[1], r[0]))
            with open(os.path.join(d, f"{scene}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["trackId", "frame", "x", "y", "sceneId"])
                w.writerows(rows)

    print(f"\n{'sequence':<40}{'split':>7}{'frames':>8}{'clips':>8}{'ppl/frm':>9}")
    for s, sp, nf, us, pf in stats:
        flag = "" if us else "   <- too short for this step, DROPPED"
        print(f"{s:<40}{sp:>7}{nf:>8}{us:>8}{pf:>9.1f}{flag}")
    tot = kept["train"] + kept["test"]
    share = 100 * kept["test"] / tot if tot else 0
    print(f"\nclips: {kept['train']} train / {kept['test']} test "
          f"({share:.0f}% test)")
    if share > 40:
        print(f"  WARNING: test is {share:.0f}% of clips. The long recordings "
              f"are the ones that survive subsampling, so holding them out "
              f"inverts the split. Change SPECS['{a.dataset}']['test'].")
    if kept["test"] == 0 or kept["train"] == 0:
        print("  ERROR: one side is empty.")

    surviving = sum(1 for _, _, _, u, _ in stats if u)
    if surviving < len(stats):
        print(f"NOTE: {len(stats) - surviving}/{len(stats)} sequences dropped "
              f"entirely at step {step}. Lower --step to keep them, at the "
              f"cost of a shorter time horizon.")

    print("\nPaste into utils_videomae.py's IMG_SIZES:")
    print(f'    "{a.dataset}": {sizes},')
    print("\nAnd add to get_trajectories():")
    print(f'    if dataset == "{a.dataset}":\n'
          f'        trajectories = get_trajectories_gcs(\n'
          f'            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len,\n'
          f'            obs_frames=obs_frames, step={step})')
    if not a.dry_run:
        print(f"\nwrote: {out_root}/{{train,test}}/")


if __name__ == "__main__":
    main()
