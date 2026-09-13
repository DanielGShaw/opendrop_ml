"""6.1b -- does a fit window scaled to drop width lower the experimental noise
floor faster than it erodes the reflective step?

For each window rule (fixed n rows, or n = f * widest width), measure the
bottom-minus-top step at the widest point on:
  * the 171 experimental images (no GT; the ones reaching the cascade's decision
    are all genuine high-angle drops, so their max step IS the noise floor), and
  * synthetic reflective cells 75-95 deg at three resolutions, plus a protected
    high-angle sample.
Then, per rule, set floor = experimental max + 2 deg and ratio > 3, and count
which synthetic reflective cells would be caught. Measurement only.

Run from OpenDrop_OP/paper_results/offgrid/:  python phase6_window_vs_noise.py
"""
import glob, io, contextlib, os, sys
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper_results/, for config
from config import OFFGRID_RESULTS_DIR as RESULTS_DIR, PAPER_RESULTS_DIR
from phase6_window_sweep import load_or_render, isolate_contour, check_for_high_angle, _find_point_index, linear_fit, calculate_contact_angle
from phase6_reflective_sweep import CPID, _branch_from_stdout

OUT = os.path.join(RESULTS_DIR, "phase6_window_vs_noise.csv")
FIXED = [10, 15, 20, 30, 40]
FRACS = [0.02, 0.03, 0.05, 0.07, 0.10]


def steps(contour, wp):
    i1 = _find_point_index(contour, wp[0], "w"); i2 = _find_point_index(contour, wp[1], "w")
    width = float(wp[1][0] - wp[0][0]); out = {"width": width}
    rules = [(f"n{n}", n) for n in FIXED] + [(f"f{f}", max(10, int(round(width * f)))) for f in FRACS]
    for name, n in rules:
        b1, _, _ = linear_fit(contour[max(0, i1 - n):i1]); t1, _, _ = linear_fit(contour[i1:i1 + n])
        b2, _, _ = linear_fit(contour[i2:i2 + n]);          t2, _, _ = linear_fit(contour[max(0, i2 - n):i2])
        k = max(calculate_contact_angle(b1, "left") - calculate_contact_angle(t1, "left"),
                calculate_contact_angle(b2, "right") - calculate_contact_angle(t2, "right"))
        out[f"kink_{name}"] = k
        out[f"arc_{name}"] = np.degrees(n / (width / 2))
        out[f"n_{name}"] = n
    return out


def main():
    rows = []
    # experimental
    for p in sorted(glob.glob(os.path.join(PAPER_RESULTS_DIR, "data", "experimental", "*"))):
        img = cv2.imread(p)
        if img is None: continue
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, c, _, _ = isolate_contour(img, DISPLAY=False, blur=True); CPID(c, display=True)
            cpg, wp = check_for_high_angle(c, display=False)
            r = dict(set="experimental", name=os.path.basename(p), angle=np.nan, scaler=np.nan, roughness=np.nan,
                     branch=_branch_from_stdout(buf.getvalue()), prefer="CP_guesses"); r.update(steps(c, wp)); rows.append(r)
        except Exception as e:
            pass
    # synthetic: reflective band + protected sample
    cells = [(a, 0.55, s, r) for r in (-0.06, -0.045, -0.03, -0.015) for a in (75, 80, 85, 90, 95) for s in (1.0, 2.83, 4.0)]
    cells += [(a, 0.55, s, r) for r in (-0.045, 0.0) for a in (100, 110, 130, 150, 170) for s in (1.0, 2.83, 4.0)]
    for a, b, s, rr in cells:
        img, (gl, gr), _ = load_or_render(a, b, s, rr)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, c, _, _ = isolate_contour(img, DISPLAY=False, blur=True); CPID(c, display=True)
        cpg, wp = check_for_high_angle(c, display=False)
        e = lambda p, q: float(np.hypot(p[0] - q[0], p[1] - q[1]))
        wpe = (e(wp[0], gl) + e(wp[1], gr)) / 2; cpe = (e(cpg[0], gl) + e(cpg[1], gr)) / 2
        prefer = "widest_points" if wpe + 3 < cpe else "CP_guesses" if cpe + 3 < wpe else "either"
        r = dict(set="synthetic", name=f"{a}_{s}_{rr}", angle=a, scaler=s, roughness=rr,
                 branch=_branch_from_stdout(buf.getvalue()), prefer=prefer); r.update(steps(c, wp)); rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)

    ex = df[(df.set == "experimental") & df.branch.isin(["high_angle_cp_guesses", "reflective_kink", "reflective_large_bottom", "reflective_similar"])]
    syn = df[(df.set == "synthetic") & (df.prefer != "either")]
    wp_cells = syn[syn.prefer == "widest_points"]; cpg_cells = syn[syn.prefer == "CP_guesses"]
    print(f"experimental drops at the decision: {len(ex)};  synthetic labelled: {len(syn)} (want widest_points {len(wp_cells)}, CP_guesses {len(cpg_cells)})\n")
    print(f"{'window':8s} {'exp max':>8s} {'exp p95':>8s} {'floor':>6s} | {'85deg@560 step':>15s} {'85deg@200':>10s} | caught(wp) by angle@scaler1.0 / 2.83 / 4.0 | FP syn | FP exp")
    names = [f"n{n}" for n in FIXED] + [f"f{f}" for f in FRACS]
    for nm in names:
        k = f"kink_{nm}"; a = f"arc_{nm}"
        floor = ex[k].max() + 2
        s85_560 = wp_cells[(wp_cells.angle == 85) & (wp_cells.scaler == 1.0)][k]
        s85_200 = wp_cells[(wp_cells.angle == 85) & (wp_cells.scaler == 2.83)][k]
        fire = lambda d: (d[k] > floor) & (d[k] / d[a] > 3)
        caught = wp_cells[fire(wp_cells)]
        by = " / ".join(",".join(f"{int(x)}" for x in sorted(caught[caught.scaler == s].angle.unique())) or "-" for s in (1.0, 2.83, 4.0))
        fp_syn = int(fire(cpg_cells).sum()); fp_exp = int(fire(ex).sum())
        print(f"{nm:8s} {ex[k].max():8.1f} {ex[k].quantile(.95):8.1f} {floor:6.1f} | {s85_560.median():7.1f} (n={len(s85_560)}) {s85_200.median():10.1f} | "
              f"{len(caught):3d}/{len(wp_cells)}  {by:28s} | {fp_syn:6d} | {fp_exp:6d}")
    print("\nfloor = experimental max + 2 deg; 'caught' = synthetic reflective cells (75-95 deg) routed to widest_points under ratio>3 & step>floor.")
    print("FP syn = protected synthetic high-angle cells misrouted; FP exp = experimental drops that would be misrouted.")


if __name__ == "__main__":
    main()
