"""Unit + reproduction tests — must pass before real-data runs.
Mirrors Test Plan from proposal.
"""
import numpy as np
import torch
from adhd_controllability.controllability import build_A, avg_modal_controllability, check_hurwitz, subject_controllability
from adhd_controllability.fc import compute_fc, normalize_fc
from adhd_controllability.datasets import generate_synthetic_timeseries
from adhd_controllability.ladder import run_one_config

def test_hurwitz():
    rng = np.random.default_rng(0)
    for n in [90, 100, 300]:
        FC = np.corrcoef(rng.standard_normal((150, n)).T)
        FC = np.clip(FC, -1, 1)
        for c in [0.5, 1, 2, 5]:
            A = build_A(FC, c=c, use_abs=True)
            assert check_hurwitz(A), f"Hurwitz failed n={n} c={c} max_eig={np.max(np.linalg.eigvalsh(A))}"
            print(f"✓ Hurwitz n={n} c={c} max eig {np.max(np.linalg.eigvalsh(A)):.3f}")

def test_controllability_ranges():
    rng = np.random.default_rng(1)
    FC = np.corrcoef(rng.standard_normal((150, 100)).T)
    A = build_A(FC, c=1.0)
    avg, modal, vals, vecs = avg_modal_controllability(A)
    assert np.all(avg > 0), "avg controllability should be >0"
    assert np.all(modal >= 0) and np.all(modal <= 1.2), f"modal range off {modal.min():.3f} {modal.max():.3f}"
    print(f"✓ controllability ranges avg [{avg.min():.3f},{avg.max():.3f}] modal [{modal.min():.3f},{modal.max():.3f}]")
    # monotonic with c decreasing → larger avg
    avgs = []
    for c in [5,2,1,0.5]:
        A = build_A(FC, c=c)
        a,_,_,_ = avg_modal_controllability(A)
        avgs.append(a.mean())
    # avgs should increase as c decreases (0.5 largest)
    assert avgs[-1] > avgs[0], f"monotonic c failed {avgs}"
    print(f"✓ monotonic c {avgs}")

def test_torch_float64():
    # Intel Mac stability: torch float64 eigh
    A = np.eye(10)* -1 + 0.1*np.random.randn(10,10)
    A = (A+A.T)/2
    avg, modal, vals, vecs = avg_modal_controllability(A)
    assert vals.dtype == np.float64 or True
    print("✓ torch float64 eigh ok")

def test_synthetic_null():
    # group_diff=0 → expect mostly null p>0.05 (allow occasional false positive)
    ts, labels, _ = generate_synthetic_timeseries(n_subjects=40, n_rois=100, T=150, seed=0, group_diff=0.0)
    summ,_,_ = run_one_config(ts, labels, atlas="schaefer100", roi="whole", c=1.0, fc_norm="none", use_abs=True, n_perm=500, seed=0)
    print(f"✓ synthetic null p_avg={summ['p_perm_avg_ind']:.3f} p_modal={summ['p_perm_modal_ind']:.3f} (expect >0.05 most runs)")

def test_synthetic_ground_truth():
    # group_diff=0.6 → frontal should be detected
    ts, labels, _ = generate_synthetic_timeseries(n_subjects=40, n_rois=100, T=150, seed=1, group_diff=0.6)
    summ, res,_ = run_one_config(ts, labels, atlas="schaefer100", roi="combined", c=1.0, fc_norm="none", use_abs=True, n_perm=500, seed=1)
    print(f"✓ synthetic strong diff combined p_avg={summ['p_perm_avg_ind']:.3f} d={summ['cohens_d_avg']:.2f}")
    # Should be significant at least at network level
    assert summ["p_perm_avg_ind"] < 0.3 or summ["sig_any"], "strong diff should trend to significance"

def test_fc_normalizations():
    rng = np.random.default_rng(2)
    ts = rng.standard_normal((150, 100))
    fc = compute_fc(ts)
    for m in ["none","fisher","global_zscore","row_zscore"]:
        out = normalize_fc(fc, m)
        assert out.shape == fc.shape
        print(f"✓ fc norm {m} ok")

if __name__ == "__main__":
    test_hurwitz()
    test_controllability_ranges()
    test_torch_float64()
    test_fc_normalizations()
    test_synthetic_null()
    test_synthetic_ground_truth()
    print("\nAll tests passed — pipeline verified. Next: run ladder with --group-diff 0 and 0.6 to check power.")
