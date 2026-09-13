"""Phase 6.1 Step 2 prototyping, part 2 -- curvature-removed kink (option 2)
and a per-cell view of where the short kink fails, by resolution.

For each side of the widest point, measure the local tangent angle with short
sliding fits at offsets 3..m rows above and below, fit angle-vs-offset linearly
on each side, extrapolate both to offset 0 and take the difference. That is the
tangent step at the point with the smooth-arc curvature removed.

Same 180 cells as phase6_window_sweep.py. Run from OpenDrop_OP/paper_results/offgrid/.
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase6_window_sweep import (ANGLES, BOND, ROUGHNESS, SCALERS, PREFER_MARGIN_PX, RESULTS_DIR,
                                 load_or_render, isolate_contour, check_for_high_angle,
                                 _find_point_index, linear_fit, calculate_contact_angle)
from phase6_separability import best_threshold

OUT = os.path.join(RESULTS_DIR, "phase6_kink_refine.csv")

def side_angles(contour, i, side, step, m, direction):
    """tangent angles at offsets step..m (rows) from index i, going `direction` (+1 below, -1 above)."""
    out = []
    for off in range(step, m + 1, step):
        lo, hi = (i + direction * off - step // 2, i + direction * off + step // 2 + 1)
        lo, hi = max(0, min(lo, hi)), max(lo, hi)
        seg = contour[lo:hi]
        if len(seg) < 3: continue
        mfit, _, _ = linear_fit(seg)
        out.append((off, calculate_contact_angle(mfit, side)))
    return out

def curvature_removed_kink(contour, i, side, step, m):
    above = side_angles(contour, i, side, step, m, -1)
    below = side_angles(contour, i, side, step, m, +1)
    if len(above) < 2 or len(below) < 2: return np.nan
    fa = np.polyfit([o for o, _ in above], [a for _, a in above], 1)
    fb = np.polyfit([o for o, _ in below], [a for _, a in below], 1)
    return float(np.polyval(fb, 0) - np.polyval(fa, 0))  # bottom minus top at the point

def main():
    rows = []
    cells = [(a, BOND, s, r) for r in ROUGHNESS for a in ANGLES for s in SCALERS]
    for k, (a, b, s, r) in enumerate(cells):
        img, (gl, gr), src = load_or_render(a, b, s, r)
        _, c, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
        cpg, wp = check_for_high_angle(c, display=False)
        i1 = _find_point_index(c, wp[0], "k"); i2 = _find_point_index(c, wp[1], "k")
        e = lambda p, q: float(np.hypot(p[0]-q[0], p[1]-q[1]))
        wpe = (e(wp[0], gl) + e(wp[1], gr)) / 2; cpe = (e(cpg[0], gl) + e(cpg[1], gr)) / 2
        row = dict(angle=a, scaler=s, roughness=r, width=float(wp[1][0]-wp[0][0]), wp_err=wpe, cpg_err=cpe,
                   prefer=("widest_points" if wpe + PREFER_MARGIN_PX < cpe else
                           "CP_guesses" if cpe + PREFER_MARGIN_PX < wpe else "either"))
        # raw short kink (as in window sweep, n=10)
        t1, _, _ = linear_fit(c[i1:i1+10]); b1, _, _ = linear_fit(c[i1-10:i1])
        t2, _, _ = linear_fit(c[i2-10:i2]); b2, _, _ = linear_fit(c[i2:i2+10])
        row["kink_raw10"] = max(calculate_contact_angle(b1,"left")-calculate_contact_angle(t1,"left"),
                                calculate_contact_angle(b2,"right")-calculate_contact_angle(t2,"right"))
        span = i2 - i1
        for step, m in ((4, 16), (4, 24), (6, 30), (6, 48)):
            row[f"kink_cr_s{step}_m{m}"] = max(curvature_removed_kink(c, i1, "left", step, m),
                                               curvature_removed_kink(c, i2, "right", step, m))
        for frac in (0.05, 0.10, 0.20):  # window scaled with drop size
            m = max(12, int(span * frac)); step = max(4, m // 5)
            row[f"kink_cr_f{frac}"] = max(curvature_removed_kink(c, i1, "left", step, m),
                                          curvature_removed_kink(c, i2, "right", step, m))
        rows.append(row)
        if (k+1) % 45 == 0: print(f"  {k+1}/{len(cells)}")
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    lab = df[df.prefer != "either"]
    feats = [c for c in df.columns if c.startswith("kink")]
    print("\nmisassignments (n labelled: " + ", ".join(f"{s}:{(lab.scaler==s).sum()}" for s in SCALERS) + ")")
    print(f"{'feature':18s} " + " ".join(f"{s:>6}" for s in SCALERS) + "  pooled")
    for f in feats:
        errs = [best_threshold(lab[lab.scaler==s][f].dropna(), lab[lab.scaler==s].dropna(subset=[f]).prefer)["min_errors"] for s in SCALERS]
        pooled = best_threshold(lab[f].dropna(), lab.dropna(subset=[f]).prefer)
        print(f"{f:18s} " + " ".join(f"{e:>6}" for e in errs) + f"  {pooled['min_errors']:>6}   (T={pooled['best_T']:.1f}, {pooled['rule']})")
    # where does the best feature fail, per scaler?
    for f in ("kink_raw10",) + tuple(sorted(feats, key=lambda f: best_threshold(lab[f].dropna(), lab.dropna(subset=[f]).prefer)["min_errors"])[:1]):
        print(f"\n{f}: misassigned cells per scaler")
        for s in SCALERS:
            sub = lab[lab.scaler==s].dropna(subset=[f]); r = best_threshold(sub[f], sub.prefer)
            above = sub[f] > r["best_T"]
            wrong = sub[(above & (sub.prefer=="widest_points")) | (~above & (sub.prefer=="CP_guesses"))] if r["rule"]=="x>T -> CP_guesses" \
                    else sub[(above & (sub.prefer=="CP_guesses")) | (~above & (sub.prefer=="widest_points"))]
            print(f"  scaler {s} (width~{sub.width.median():.0f}px): T={r['best_T']:.1f} -> " +
                  (", ".join(f"({int(w.angle)}°, r={w.roughness}, {w.prefer[:2]}, {f}={w[f]:.1f})" for _, w in wrong.iterrows()) or "clean"))

if __name__ == "__main__":
    main()
