# Changelog

## 1.0.0 (unreleased)

First public release of opendrop_ml. It builds on the OpenDrop port developed by
the University of Melbourne student team, and adds the Conan-MLII contact-angle
model and the pipelines that support it.

### Added

- **Conan-MLII contact-angle model.** A Conv1D model that predicts contact
  angles from 20° to 180°, with an all-angle preprocessing pipeline: OPTICS
  clustering and automatic contact-point identification (CPID). The model is
  trained on Bond numbers up to 1 and validated against synthetic ground
  truth up to Bond number 1.5.
- **Video mode.** Two-pass contact-angle analysis of videos, including
  needle-in-drop and tilted-surface pipelines.
- **Paper results pipeline.** `paper_results/` regenerates every CSV and figure
  in the Conan-MLII paper from the committed data (see
  `paper_results/README.md`).
- **Cross-platform smoke test.** A manually triggered GitHub Actions workflow
  builds the package on Windows and Linux, runs the test suite, and checks the
  contact-angle results against the macOS reference.

### Fixed

- **Windows results.** Contact angles computed on Windows differed from macOS
  and Linux for a large minority of images, by up to 169° in extreme cases.
  Two geometry calculations overflowed on Windows, where numpy's default
  integer is 32-bit. Windows and Linux now match macOS on all 171 experimental
  images.
- **Contact-point ties.** When two candidate contact points scored equally,
  the choice between them depended on the platform. The choice is now
  deterministic.
- **Crashes on sparse contours.** Contact-point identification now raises a
  clear error instead of crashing on contours with too few points.
- **Reflective surfaces.** Contact points on reflective surfaces are detected
  more reliably, including drops below 90°.

### Changed

- Removed unused dependencies (`numba`, `llvmlite`, `Keras-Preprocessing`,
  `tensorboard-plugin-wit`) and added `pandas`, which the video figures need.
