"""JLucid 3 — stay-only residual NODE + discrete LEiDA router.

Thesis split (see ``.agents/step-3.md`` §2 and §4.4):

- Continuous: ``StayField`` is a short-shot residual NODE trained **only
  on dwell pairs** (label stays). Its Jacobian is a within-state
  linearization (Paper 6 object).
- Discrete: ``DiscreteRouter`` predicts flip / next-state from the V1
  window plus EiDA extras (V2, eigen-gap). Not a NODE.

This is not JLucid 2.2 (persist/jump invert gate). Mixed next-V1 cosine
is a diagnostic, not the training objective.
"""
