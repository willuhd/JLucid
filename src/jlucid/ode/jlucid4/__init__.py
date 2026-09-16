"""JLucid 4 — stay field + identifiable J + discrete dest/when.

Continuous (stay-only, in-dwell)::

    z ← z + A(p) z + b(p)                 # J = A(p), ε = 0
    V_base = unit(V + γ(z0) (V − Vprev))
    Vhat   = unit(V_base + Dec(z))

Discrete is not a NODE: frozen dest logits + a when-head. Invert is not
in the field. Mixed next-V1 cosine is not trained.

See ``.agents/step-3.md`` §4.5.
"""
