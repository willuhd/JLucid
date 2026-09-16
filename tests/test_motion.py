import numpy as np
import pytest

from jlucid.qc.motion import compute_fd, scrub_mask


def test_compute_fd_scale():
    rp = np.zeros((10, 6))
    rp[1, 0] = 0.5  # 0.5 degree roll change (AFNI -1Dfile unit)
    rp[2, 3] = 1.0  # 1 mm dS change
    fd, mean_fd = compute_fd(rp)
    # fd[0] is 0 by convention; frame 1 carries the 0.5 deg roll scaled by
    # 50 mm (50 * deg2rad(0.5) ~ 0.436 mm), frame 2 adds the 1 mm translation
    assert fd[0] == pytest.approx(0.0)
    assert fd[1] == pytest.approx(50.0 * np.deg2rad(0.5))
    assert fd[2] == pytest.approx(50.0 * np.deg2rad(0.5) + 1.0)
    assert mean_fd == pytest.approx(fd.mean())


def test_scrub_mask_neighborhood():
    fd = np.zeros(10)
    fd[5] = 1.0
    mask = scrub_mask(fd, threshold=0.5, n_before=1, n_after=2)
    assert mask[4:8].tolist() == [False, False, False, False]
    assert mask[0:4].all() and mask[8:].all()


def test_compute_fd_requires_6cols():
    import pytest

    with pytest.raises(ValueError):
        compute_fd(np.zeros((5, 3)))
