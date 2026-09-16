"""JLucid 4 eval: stay vs persist/half-step, J smoke, packed discrete."""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import SEED, SHOT_BATCH, WINDOW_TR
from jlucid.ode.jlucid4.shots import collate_shots, dwell_index


def cosine_rows(a, b):
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(an * bn, 1e-12)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _subject_mean(values: np.ndarray, sids: np.ndarray) -> tuple[float, int]:
    if len(values) == 0:
        return float("nan"), 0
    means = [float(values[sids == s].mean())
             for s in np.unique(sids) if (sids == s).any()]
    if not means:
        return float("nan"), 0
    return float(np.mean(means)), int(len(means))


def bootstrap_mean_ci(x, *, n_boot: int = 2000, seed: int = SEED,
                      alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_boot": int(n_boot)}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = x[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(n), "n_boot": int(n_boot)}


def bootstrap_paired_diff_ci(a, b, *, n_boot: int = 2000, seed: int = SEED,
                             alpha: float = 0.05) -> dict:
    return bootstrap_mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
                             n_boot=n_boot, seed=seed, alpha=alpha)


def _score_model_on_index(model, bundle, idx, device, *, window: int,
                          horizon: int, batch: int,
                          decode_mode: str | None = None) -> dict:
    empty = {d: np.array([]) for d in range(1, horizon + 1)}
    if not idx:
        return {"sid": np.array([]), "model": empty, "persist": empty}
    sid_all: list[str] = []
    model_h = {d: [] for d in range(1, horizon + 1)}
    persist_h = {d: [] for d in range(1, horizon + 1)}
    model.eval()
    for s0 in range(0, len(idx), batch):
        bi = idx[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, horizon, device)
        out = model(b["windows"], b["p0"], b["v1_now"],
                    b["v2_now"], b["gap_now"], n_steps=horizon,
                    decode_mode=decode_mode)
        hat = out["v_stay"].detach().cpu().numpy()
        y = b["v1_future"].detach().cpu().numpy()
        now = b["v1_now"].detach().cpu().numpy()
        for d in range(1, horizon + 1):
            model_h[d].append(cosine_rows(hat[:, d - 1], y[:, d - 1]))
            persist_h[d].append(cosine_rows(now, y[:, d - 1]))
        sid_all.extend(s for s, _ in bi)
    return {
        "sid": np.asarray(sid_all),
        "model": {d: np.concatenate(model_h[d]) for d in model_h},
        "persist": {d: np.concatenate(persist_h[d]) for d in persist_h},
    }


def _pack_delta(model_cos: np.ndarray, persist_cos: np.ndarray,
                sids: np.ndarray) -> dict:
    sm, nsub = _subject_mean(model_cos, sids)
    sp, nsub_p = _subject_mean(persist_cos, sids)
    gains = []
    for s in np.unique(sids) if len(sids) else []:
        m = sids == s
        if m.any():
            gains.append(float(model_cos[m].mean() - persist_cos[m].mean()))
    return {
        "n_shots": int(len(model_cos)),
        "n_subjects": int(nsub),
        "model_subject": sm if nsub else None,
        "persist_subject": sp if nsub_p else None,
        "gain_subject": float(sm - sp) if nsub else None,
        "model_shot": float(model_cos.mean()) if len(model_cos) else None,
        "persist_shot": float(persist_cos.mean()) if len(persist_cos) else None,
        "gain_shot": (float(model_cos.mean() - persist_cos.mean())
                      if len(model_cos) else None),
        "frac_subjects_beat_persist": (
            float(np.mean(np.asarray(gains) > 0)) if gains else None),
        "n_subjects_gain_pos": int(np.sum(np.asarray(gains) > 0)) if gains else 0,
    }


def half_step_on_index(bundle: dict, idx: list[tuple[str, int]],
                       horizon: int, gamma: float = 0.5) -> dict:
    if not idx:
        return {"sid": np.array([]),
                "model": {d: np.array([]) for d in range(1, horizon + 1)},
                "persist": {d: np.array([]) for d in range(1, horizon + 1)}}
    model_h = {d: [] for d in range(1, horizon + 1)}
    persist_h = {d: [] for d in range(1, horizon + 1)}
    sids = []
    for sid, t in idx:
        v1 = bundle[sid]["v1"]
        prev = v1[t - 1] if t >= 1 else v1[t]
        cur = v1[t].copy()
        now = v1[t]
        for d in range(1, horizon + 1):
            pred = _unit(cur + gamma * (cur - prev))
            prev, cur = cur, pred
            model_h[d].append(float(pred @ v1[t + d]))
            persist_h[d].append(float(now @ v1[t + d]))
        sids.append(sid)
    return {
        "sid": np.asarray(sids),
        "model": {d: np.asarray(model_h[d], dtype=np.float64) for d in model_h},
        "persist": {d: np.asarray(persist_h[d], dtype=np.float64) for d in persist_h},
    }


def _with_ci(pack: dict, model_cos, persist_cos, sids, *,
             n_boot: int = 2000, seed: int = SEED) -> dict:
    out = dict(pack)
    if len(sids) == 0:
        return out
    m_s, p_s = [], []
    for s in np.unique(sids):
        m = sids == s
        m_s.append(float(model_cos[m].mean()))
        p_s.append(float(persist_cos[m].mean()))
    m_s = np.asarray(m_s)
    p_s = np.asarray(p_s)
    g_ci = bootstrap_paired_diff_ci(m_s, p_s, n_boot=n_boot, seed=seed)
    c_ci = bootstrap_mean_ci(m_s, n_boot=n_boot, seed=seed)
    out["gain_subject_ci95"] = [g_ci["lo"], g_ci["hi"]]
    out["model_subject_ci95"] = [c_ci["lo"], c_ci["hi"]]
    out["gain_excludes_0"] = bool(g_ci["lo"] > 0 or g_ci["hi"] < 0)
    out["n_boot"] = int(n_boot)
    return out


@torch.no_grad()
def evaluate(model, bundle, sids, device, *, window=None,
             horizons: tuple[int, ...] = (1, 2, 3), batch=SHOT_BATCH,
             decode_mode: str | None = None, n_boot: int = 2000,
             seed: int = SEED) -> dict:
    window = int(window or getattr(model, "window", WINDOW_TR))
    sub = {s: bundle[s] for s in sids}
    model.eval()
    result: dict = {
        "window": window,
        "decode_mode": decode_mode or getattr(model, "decode_mode", None),
        "gamma_mode": getattr(model.stay, "gamma_mode", None),
        "eps": 0.0,
        "horizons": {},
        "nested": {},
        "half_step": {},
    }
    for h in horizons:
        idx = dwell_index(sub, window, h)
        scored = _score_model_on_index(
            model, bundle, idx, device, window=window, horizon=h,
            batch=batch, decode_mode=decode_mode)
        pack = _pack_delta(scored["model"][h], scored["persist"][h], scored["sid"])
        result["horizons"][str(h)] = _with_ci(
            pack, scored["model"][h], scored["persist"][h], scored["sid"],
            n_boot=n_boot, seed=seed)
        half = half_step_on_index(sub, idx, h, gamma=0.5)
        hp = _pack_delta(half["model"][h], half["persist"][h], half["sid"])
        vs = _pack_delta(scored["model"][h], half["model"][h], scored["sid"])
        result["horizons"][str(h)]["half_subject"] = hp["model_subject"]
        result["horizons"][str(h)]["half_shot"] = hp["model_shot"]
        result["horizons"][str(h)]["gain_vs_half_subject"] = vs["gain_subject"]
        result["horizons"][str(h)]["gain_vs_half_shot"] = vs["gain_shot"]
        if len(scored["sid"]):
            mh, hh = [], []
            for s in np.unique(scored["sid"]):
                m = scored["sid"] == s
                mh.append(float(scored["model"][h][m].mean()))
                hh.append(float(half["model"][h][m].mean()))
            vci = bootstrap_paired_diff_ci(
                np.asarray(mh), np.asarray(hh), n_boot=n_boot, seed=seed)
            result["horizons"][str(h)]["gain_vs_half_ci95"] = [vci["lo"], vci["hi"]]
            result["horizons"][str(h)]["gain_vs_half_excludes_0"] = bool(
                vci["lo"] > 0 or vci["hi"] < 0)
        result["half_step"][str(h)] = hp

    max_h = max(horizons)
    nested_idx = dwell_index(sub, window, max_h)
    nested = _score_model_on_index(
        model, bundle, nested_idx, device, window=window, horizon=max_h,
        batch=batch, decode_mode=decode_mode)
    for d in range(1, max_h + 1):
        result["nested"][str(d)] = _pack_delta(
            nested["model"][d], nested["persist"][d], nested["sid"])

    idx1 = dwell_index(sub, window, 1)
    gammas = []
    for s0 in range(0, len(idx1), batch):
        bi = idx1[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        out = model(b["windows"], b["p0"], b["v1_now"],
                    b["v2_now"], b["gap_now"], n_steps=1)
        gammas.append(out["gamma"].detach().cpu().numpy().reshape(-1))
    if gammas:
        g = np.concatenate(gammas)
        result["gamma_stats"] = {
            "mode": getattr(model.stay, "gamma_mode", None),
            "mean": float(g.mean()), "median": float(np.median(g)),
            "p10": float(np.percentile(g, 10)), "p90": float(np.percentile(g, 90)),
            "n": int(g.size),
        }

    h1 = result["horizons"].get("1", {})
    result["delta1_gain_subject"] = h1.get("gain_subject")
    result["delta2_gain_subject"] = result["horizons"].get("2", {}).get("gain_subject")
    result["delta3_gain_subject"] = result["horizons"].get("3", {}).get("gain_subject")
    result["n_subjects_d1"] = h1.get("n_subjects")
    result["n_subjects_d3"] = result["horizons"].get("3", {}).get("n_subjects")
    result["stay_safe_d1"] = bool(
        result["delta1_gain_subject"] is not None
        and result["delta1_gain_subject"] >= -0.002)
    result["win_d3"] = bool(
        result["delta3_gain_subject"] is not None
        and result["delta3_gain_subject"] >= 0.025
        and (result["n_subjects_d3"] or 0) >= 25)
    return result


def format_table(ev: dict, title: str = "JLucid 4") -> str:
    lines = [f"=== {title} (in-dwell shots only) ==="]
    for h in sorted(ev.get("horizons", {}), key=int):
        r = ev["horizons"][h]
        vs = r.get("gain_vs_half_subject")
        vs_s = f"  vs half {vs:+.4f}" if vs is not None else ""
        lines.append(
            f"  Δ={h}  n={r['n_shots']} subj={r['n_subjects']}  "
            f"model {r['model_subject']:.4f}  persist {r['persist_subject']:.4f}  "
            f"gain {r['gain_subject']:+.4f}{vs_s}  "
            f"beat {r['n_subjects_gain_pos']}/{r['n_subjects']}"
        )
    lines.append(
        f"  stay-safe Δ=1: {ev.get('stay_safe_d1')}  "
        f"win Δ=3: {ev.get('win_d3')}"
    )
    return "\n".join(lines)
