"""Phase-1 acceptance gate (SYNTHETIC_DATA_REANALYSIS.md, A.4).

Re-renders one existing on-grid image and checks that image dimensions and
contact-point coordinates match the filename's recorded values. This
validates the dpi/bbox settings in create_contours_args6.py's draw_drop()
before generating hundreds of new images. Must pass before Phase 1 proceeds.

Run:
    python acceptance_gate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_files"))

from create_contours_args6 import draw_drop  # noqa: E402

from config import SYNTHETIC_ONGRID_DATA_DIR  # noqa: E402

# Reference image: paper_results/data/synthetic/100.0_0.0_1.0_-0.03_CPL_97_447_CPR_647_447_.png
REF_ANGLE = 100.0
REF_BOND = 0.0
REF_SCALER = 1.0
REF_ROUGHNESS = -0.03
REF_SHAPE = (492, 744, 3)
REF_CPL = (97, 447)
REF_CPR = (647, 447)
CP_TOLERANCE_PX = 1


def main():
    ref_path = os.path.join(
        SYNTHETIC_ONGRID_DATA_DIR, "100.0_0.0_1.0_-0.03_CPL_97_447_CPR_647_447_.png"
    )
    if not os.path.exists(ref_path):
        print(f"FAIL: reference image not found at {ref_path}")
        sys.exit(1)

    img, CPs_img = draw_drop(REF_ANGLE, REF_BOND, REF_SCALER, REF_ROUGHNESS, 5000, dpi=64)

    print(f"Rendered image shape: {img.shape} (expected {REF_SHAPE})")
    print(f"Rendered CPs: {CPs_img} (expected CPL={REF_CPL}, CPR={REF_CPR})")

    ok = True

    if img.shape != REF_SHAPE:
        print(f"FAIL: shape mismatch — got {img.shape}, expected {REF_SHAPE}")
        ok = False
    else:
        print("PASS: image shape matches")

    left_x, left_z = CPs_img[0]
    right_x, right_z = CPs_img[1]

    if abs(left_x - REF_CPL[0]) > CP_TOLERANCE_PX or abs(left_z - REF_CPL[1]) > CP_TOLERANCE_PX:
        print(f"FAIL: left CP {left_x, left_z} not within {CP_TOLERANCE_PX}px of {REF_CPL}")
        ok = False
    else:
        print("PASS: left CP within tolerance")

    if abs(right_x - REF_CPR[0]) > CP_TOLERANCE_PX or abs(right_z - REF_CPR[1]) > CP_TOLERANCE_PX:
        print(f"FAIL: right CP {right_x, right_z} not within {CP_TOLERANCE_PX}px of {REF_CPR}")
        ok = False
    else:
        print("PASS: right CP within tolerance")

    if not ok:
        print("\nACCEPTANCE GATE: FAILED — do not proceed to full generation.")
        sys.exit(1)

    print("\nACCEPTANCE GATE: PASSED")


if __name__ == "__main__":
    main()
