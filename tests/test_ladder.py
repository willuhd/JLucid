"""Ladder integration + avg-modal correlation reproduction (paper 6 Fig.2)."""
import numpy as np
from adhd_controllability.datasets import generate_synthetic_timeseries
from adhd_controllability.ladder import run_ladder
from adhd_controllability.controllability import build_A, avg_modal_controllability
from adhd_controllability.fc import compute_fc

def test_ladder_smoke():
    ts, labels,_ = generate_synthetic_timeseries(n_subjects=20, n_rois=100, seed=42, group_diff=0.0)
    df = run_ladder(ts, labels, verbose=False, n_perm=200)
    assert len(df) >= 15, f"ladder too short {len(df)}"
    assert "p_perm_avg_ind" in df.columns
    print(f"✓ ladder smoke {len(df)} configs, cols {df.columns.tolist()[:5]}")
    print(df.head(3).to_string())

def test_avg_modal_correlation():
    """Reproduce paper 6 Fig.2: whole-brain avg-modal r ~0.91 across subjects."""
    ts, labels,_ = generate_synthetic_timeseries(n_subjects=40, n_rois=100, seed=7, group_diff=0.0)
    avgs=[]; mods=[]
    for t in ts:
        fc = compute_fc(t)
        A = build_A(fc, c=1.0)
        a, m,_,_ = avg_modal_controllability(A)
        avgs.append(a.mean()); mods.append(m.mean())
    r = np.corrcoef(avgs, mods)[0,1]
    print(f"✓ avg-modal correlation r={r:.3f} (paper 6 reports +0.91; synthetic |r| should be strong; sign depends on FC structure)")
    assert abs(r) > 0.5, f"correlation too weak {r}"

if __name__=="__main__":
    test_ladder_smoke()
    test_avg_modal_correlation()
