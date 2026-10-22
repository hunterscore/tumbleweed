# tumbleweed
tumbling through orbit 

**snake rattle**

**saloon door swings open**

Monocular pose estimation of the Tango spacecraft on [SPEED+](https://purl.stanford.edu/wv398fc4383): from a single camera image, recover the spacecraft's rotation and translation relative to the camera, scored with the SPEC pose error metric.

Two approaches so far:

1. **Direct regression (finished).** ResNet18 regressing translation and a unit quaternion, trained with a geodesic rotation loss plus translation MSE. Kept as the baseline.
2. **Keypoint heatmaps + PnP (in progress).** ResNet18 with a deconvolution head predicting a heatmap for each of 11 known points on Tango. The pose comes from the 2D-3D correspondences via RANSAC PnP.

## Results

`synthetic/validation`, 11,994 images. Translation error is normalized by distance to the target, as in SPEC. Lower is better for every column.

| model | mean rot. error | median rot. error | p95 rot. error | flips > 90° | median trans. error | SPEC score |
|---|---|---|---|---|---|---|
| regression baseline | 17.9° | 11.6° | 53.6° | 2.5% | 0.034 | 0.354 |

Regression locates the spacecraft well, but its rotation estimates have a long tail. That motivates the keypoint approach. See `notebooks/regression_baseline.ipynb`.

For the keypoint pipeline, the geometry is validated on its own in `notebooks/pnp_validation.ipynb`. With exact keypoints, PnP recovers the pose with no error. With 2 px of keypoint noise, the median rotation error is 0.3°. The first detector training run finds the corners and antenna tips but confuses Tango's left/right mirror-symmetric keypoint pairs, so PnP fails on most frames. Fixing that is the current work.

## Layout

```
tumbleweed/
    data/speedplus.py            SPEED+ dataset (reads label JSON)
    data/camera.py               camera.json intrinsics loader
    data/transforms.py           image preprocessing, heatmap target rendering
    geometry/quaternion.py       batched quaternion math
    geometry/projection.py       3D -> 2D projection and drawing helpers
    geometry/tango_keypoints.py  11-point Tango keypoint model (body frame)
    geometry/heatmaps.py         sub-pixel heatmap peak decoding
    geometry/pnp.py              RANSAC PnP pose recovery
    models/pose_resnet.py        regression baseline
    models/keypoint_resnet.py    heatmap network + PnP wrapper
    losses.py                    regression losses, weighted heatmap MSE
    metrics.py                   SPEC score, flip rate, error distributions
    train.py                     regression training
    train_keypoints.py           keypoint training
    eval.py                      evaluation for either model (--model)
scripts/
    sanity_check_keypoints.py    projects the keypoint model through GT poses
    sanity_check_projection.py   draws a GT axis triad on an image
    analyze_errors.py            text summary of a per-sample error CSV
    export_labels_csv.py         flattens label JSON to CSV
notebooks/
    regression_baseline.ipynb    final regression results
    pnp_validation.ipynb         PnP and decoder checks, trained detector overlays
data/README.md                   expected dataset layout
```

## Setup

```
uv sync --extra dev
```

Put the SPEED+ dataset in `data/speedplusv2/` as described in `data/README.md`.

## Sanity check

Before training, confirm that label parsing, the quaternion convention and the camera intrinsics agree. To do this, project the keypoint model through a few ground-truth poses:

```
uv run python scripts/sanity_check_keypoints.py --data-root data/speedplusv2 --indices 0 12000 25630
```

The numbered points should land on the body corners (0-7) and the antenna tips (8-10).

## Train / eval

```
# regression baseline
uv run python -m tumbleweed.train --data-root data/speedplusv2
uv run python -m tumbleweed.eval --data-root data/speedplusv2 --checkpoint checkpoints/best.pt --advanced

# keypoints + PnP
uv run python -m tumbleweed.train_keypoints --data-root data/speedplusv2
uv run python -m tumbleweed.eval --model keypoint --checkpoint checkpoints/keypoint/best.pt --advanced
```

Add `--device mps` on Apple silicon.
