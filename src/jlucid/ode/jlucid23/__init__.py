"""JLucid 2.3 — stay-only velocity field + precision invert.

The 2.0 residual NODE is one map for a jump process: it helps antipodal
flips and taxes stays. 2.1/2.2 gated that invert and slid along the same
stay–overall Pareto. 2.3 splits the object the thesis needs:

- a *stay* field (last-step velocity + small residual), trained only on
  dwells, with an identifiable Jacobian ∂f/∂z
- a *discrete* invert that may fire only when a high-precision flip
  detector agrees *and* the last step is already falling

See ``.agents/step-3.md`` §4.3. Mixed-cosine benchmark; not the thesis field.
"""
