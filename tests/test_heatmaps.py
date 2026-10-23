import torch

from tumbleweed.geometry.heatmaps import decode_heatmaps, find_peaks


def _gaussian(height, width, u, v, amplitude=1.0, sigma=2.0):
    vs = torch.arange(height, dtype=torch.float32)[:, None]
    us = torch.arange(width, dtype=torch.float32)[None, :]
    return amplitude * torch.exp(-((us - u) ** 2 + (vs - v) ** 2) / (2 * sigma**2))


def test_find_peaks_returns_both_blobs_strongest_first():
    # The mirror-pair case: one heatmap, two blobs of different strength.
    hm = _gaussian(96, 160, 40.3, 30.7) + _gaussian(96, 160, 110.6, 60.2, amplitude=0.7)
    coords, values = find_peaks(hm[None, None], k=2)
    assert torch.allclose(coords[0, 0, 0], torch.tensor([40.3, 30.7]), atol=0.25)
    assert torch.allclose(coords[0, 0, 1], torch.tensor([110.6, 60.2]), atol=0.25)
    assert values[0, 0, 0] > values[0, 0, 1]


def test_each_image_and_keypoint_uses_its_own_heatmap():
    # Different peak per (image, keypoint). N, K and k all differ, so mixed-up axes can't hide.
    n, num_kp = 2, 3
    truth = torch.tensor([[[10.0, 20.0], [50.0, 30.0], [90.0, 70.0]],
                          [[130.0, 15.0], [25.0, 80.0], [70.0, 45.0]]])
    hm = torch.stack([torch.stack([_gaussian(96, 160, *truth[i, j]) for j in range(num_kp)])
                      for i in range(n)])
    coords, _ = find_peaks(hm, k=4)
    assert torch.allclose(coords[:, :, 0], truth, atol=0.25)


def test_top_candidate_matches_decode_heatmaps():
    torch.manual_seed(0)
    hm = torch.randn(4, 11, 96, 160)
    coords, values = find_peaks(hm, k=4)
    expected_coords, expected_conf = decode_heatmaps(hm)
    assert torch.allclose(coords[:, :, 0], expected_coords)
    assert torch.allclose(values[:, :, 0], expected_conf)
