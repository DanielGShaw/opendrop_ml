#!/usr/bin/env python
"""Combine the bias/disagreement plot and the CP close-up composite into one
image, matching the old paper's single-figure layout for Figure 3
(PART2 §7.3.5, item C).

Reads the two already-generated figures:
  - figures/synthetic/synthetic_bias_and_disagreement.png (panels a, b)
  - figures/CA_tangent_plots/synthetic_composite.png (i)-iv) close-ups)
and stacks them vertically (bias/disagreement on top, close-ups below) into
one PNG. The close-up composite is much higher resolution than needed for
this combined figure, so it is downscaled to the bias/disagreement panel's
width rather than upscaling the smaller image.

Writes: figures/synthetic/Figure3-updated.png

Run from OpenDrop_OP/:
    python paper_results/plot/combine_figure3.py
"""

import os
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # both source images are legitimate, just large

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import FIGURES_DIR

TOP_IMAGE = os.path.join(FIGURES_DIR, "synthetic", "synthetic_bias_and_disagreement.png")
BOTTOM_IMAGE = os.path.join(FIGURES_DIR, "CA_tangent_plots", "synthetic_composite.png")
OUTPUT_IMAGE = os.path.join(FIGURES_DIR, "synthetic", "Figure3-updated.png")

# White band between the two panels so they don't visually collide.
GAP_PX = 40


def main():
    top = Image.open(TOP_IMAGE).convert("RGB")
    bottom = Image.open(BOTTOM_IMAGE).convert("RGB")

    # Downscale the larger image to the smaller one's width, preserving
    # aspect ratio, rather than upscaling (which would just blur the smaller
    # one without adding real detail).
    target_width = min(top.width, bottom.width)

    def fit_width(img, width):
        if img.width == width:
            return img
        height = round(img.height * width / img.width)
        return img.resize((width, height), Image.LANCZOS)

    top = fit_width(top, target_width)
    bottom = fit_width(bottom, target_width)

    combined = Image.new(
        "RGB", (target_width, top.height + GAP_PX + bottom.height), "white"
    )
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height + GAP_PX))

    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    combined.save(OUTPUT_IMAGE)
    print("Wrote {} ({}x{})".format(OUTPUT_IMAGE, combined.width, combined.height))


if __name__ == "__main__":
    main()
