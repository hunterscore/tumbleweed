# tumbleweed
tumbling through orbit 

**snake rattle**

**saloon door swings open**

Spacecraft pose estimation on [SPEED+](https://purl.stanford.edu/wv398fc4383) (Tango spacecraft): a ResNet18 direct-regression baseline (translation + quaternion) trained with a geodesic rotation loss + translation MSE, evaluated with the SPEC pose error metric.

## Layout

```
tumbleweed/
    data/speedplus.py      SPEED+ Dataset (reads label JSON)
    data/camera.py         camera.json intrinsics loader
    data/transforms.py     image preprocessing
    geometry/quaternion.py batched quaternion math (rotation matrix, geodesic angle)
    geometry/projection.py 3D -> 2D projection for the sanity-check script
    models/pose_resnet.py  ResNet18 backbone + translation/quaternion heads
    losses.py               geodesic rotation loss + translation MSE
    metrics.py               SPEC pose error score
    train.py / eval.py       train/eval loops
scripts/
    sanity_check_projection.py   projects a GT pose onto its image
data/README.md              expected dataset layout
```

## Setup

```
pip install -e .
```

Drop the SPEED+ dataset into `data/speedplusv2/` per `data/README.md`.

## Before touching the model

Verify label parsing, quaternion convention, and camera intrinsics line up by projecting a ground-truth pose onto its image:

```
python scripts/sanity_check_projection.py --data-root data/speedplusv2 --index 0 --output /tmp/sanity_check.png
```

The drawn axis triad (X=red, Y=green, Z=blue) should sit at the spacecraft's origin. If it doesn't, something upstream of the model (label parsing / quaternion convention / camera.json parsing) is wrong.

## Train / eval

```
python -m tumbleweed.train --data-root data/speedplusv2
python -m tumbleweed.eval --data-root data/speedplusv2 --checkpoint checkpoints/best.pt
```
