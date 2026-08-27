"""Dataset class for SPEED+ (Spacecraft Pose Estimation Dataset, next-gen).

Expected on-disk layout, matching the official SPEED+ release::

    <root>/
        camera.json
        synthetic/
            images/
                img000001.jpg
                ...
            train.json
            validation.json
        lightbox/
            images/
            test.json          # unlabeled: no pose fields
        sunlamp/
            images/
            test.json          # unlabeled: no pose fields

Each entry in a label JSON looks like::

    {
        "filename": "img000001.jpg",
        "r_Vo2To_vbs_true": [x, y, z],
        "q_vbs2tango_true": [q0, q1, q2, q3]
    }

``r_Vo2To_vbs_true`` is the translation (meters) from the camera (VBS)
origin to the Tango origin, expressed in the camera frame.
``q_vbs2tango_true`` is the scalar-first (w, x, y, z) unit quaternion
rotating the camera frame into the Tango body frame. The lightbox/sunlamp
test domains ship without pose labels; samples from those splits omit
"quaternion"/"translation" and only carry "image"/"filename".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpeedPlusDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str = "synthetic",
        split: str = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Args:
            root: path to the SPEED+ dataset root (contains camera.json).
            domain: "synthetic", "lightbox", or "sunlamp".
            split: "train" / "validation" for the synthetic domain, or
                "test" for lightbox/sunlamp.
            transform: optional callable applied to the sample dict; see
                ``tumbleweed.data.transforms.build_transforms``.
        """
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.transform = transform

        self.image_dir = self.root / domain / "images"
        label_path = self.root / domain / f"{split}.json"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}. Expected SPEED+ layout "
                f"is <root>/<domain>/<split>.json alongside <root>/<domain>/images/."
            )

        with open(label_path) as f:
            self.entries: List[Dict[str, Any]] = json.load(f)

        self.has_pose = len(self.entries) > 0 and "q_vbs2tango_true" in self.entries[0]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        filename = entry["filename"]
        image = Image.open(self.image_dir / filename).convert("RGB")

        sample: Dict[str, Any] = {"image": image, "filename": filename}

        if "q_vbs2tango_true" in entry:
            q = np.asarray(entry["q_vbs2tango_true"], dtype=np.float32)
            t = np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float32)
            sample["quaternion"] = torch.from_numpy(q)
            sample["translation"] = torch.from_numpy(t)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
