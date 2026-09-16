"""JLucid 3 eval: stay-field vs persist on dwells; discrete flip/next-state."""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import FLIP_COS_THRESH, SHOT_BATCH
from jlucid.ode.jlucid3.shots import collate_shots, load_bundle, shot_index, stay_index


def cosine_rows(a, b):
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(an * bn, 1e-12)


def _auc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def evaluate(model, bundle, sids, device, *, window=None, n_steps=1,
             batch=SHOT_BATCH) -> dict:
    window = int(window or model.window)
    sub = {s: bundle[s] for s in sids}
    stay_idx = stay_index(sub, window, n_steps)
    all_idx = shot_index(sub, window, 1)
    model.eval()

    stay_cos, stay_p, stay_sid = [], [], []
    for s0 in range(0, len(stay_idx), batch):
        bi = stay_idx[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, n_steps, device)
        out = model(b["windows"], b["p0"], b["v1_now"],
                    b["v2_now"], b["gap_now"], n_steps=n_steps)
        y = b["v1_future"][:, 0].cpu().numpy()
        hat = out["v_stay"][:, 0].cpu().numpy()
        now = b["v1_now"].cpu().numpy()
        stay_cos.append(cosine_rows(hat, y))
        stay_p.append(cosine_rows(now, y))
        stay_sid.extend(s for s, _ in bi)
    stay_cos = np.concatenate(stay_cos) if stay_cos else np.array([])
    stay_p = np.concatenate(stay_p) if stay_p else np.array([])

    flip_true, flip_score, lab_true, lab_pred, sw_true = [], [], [], [], []
    persist_lab = []
    for s0 in range(0, len(all_idx), batch):
        bi = all_idx[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        out = model(b["windows"], b["p0"], b["v1_now"],
                    b["v2_now"], b["gap_now"], n_steps=1)
        now = b["v1_now"].cpu().numpy()
        y = b["v1_future"][:, 0].cpu().numpy()
        l0 = b["labels_now"].cpu().numpy()
        l1 = b["labels_future"][:, 0].cpu().numpy()
        flip_true.append(cosine_rows(now, y) < FLIP_COS_THRESH)
        flip_score.append(torch.sigmoid(out["flip_logit"]).cpu().numpy())
        lab_true.append(l1)
        lab_pred.append(out["state_logits"].argmax(-1).cpu().numpy())
        sw_true.append(l0 != l1)
        persist_lab.append(l0 == l1)
    flip_true = np.concatenate(flip_true)
    flip_score = np.concatenate(flip_score)
    lab_true = np.concatenate(lab_true)
    lab_pred = np.concatenate(lab_pred)
    sw_true = np.concatenate(sw_true)
    persist_lab = np.concatenate(persist_lab)

    # subject-mean stay cosine
    # (discrete metrics below; stay_sid used for per-subject stay cosine)
    stay_sid = np.asarray(stay_sid)
    subj = []
    for s in np.unique(stay_sid):
        m = stay_sid == s
        subj.append((stay_cos[m].mean(), stay_p[m].mean()))
    sm, sp = np.mean(subj, axis=0) if subj else (float("nan"), float("nan"))

    # next-state acc overall and on true switches
    acc_all = float((lab_pred == lab_true).mean())
    acc_sw = float((lab_pred[sw_true] == lab_true[sw_true]).mean()) if sw_true.any() else float("nan")
    acc_persist = float(persist_lab.mean())
    # flip metrics at 0.5 and at 5% call
    call05 = flip_score >= 0.5
    prec05 = float(flip_true[call05].mean()) if call05.any() else float("nan")
    rec05 = float(call05[flip_true].mean()) if flip_true.any() else float("nan")
    tau5 = float(np.quantile(flip_score, 0.95))
    call5 = flip_score >= tau5
    prec5 = float(flip_true[call5].mean()) if call5.any() else float("nan")
    rec5 = float(call5[flip_true].mean()) if flip_true.any() else float("nan")

    return {
        "n_stay": int(len(stay_cos)),
        "n_all": int(len(flip_true)),
        "stay_cos_shot": float(stay_cos.mean()) if len(stay_cos) else None,
        "stay_persist_shot": float(stay_p.mean()) if len(stay_p) else None,
        "stay_gain_shot": float(stay_cos.mean() - stay_p.mean()) if len(stay_cos) else None,
        "stay_cos_subject": float(sm),
        "stay_persist_subject": float(sp),
        "stay_gain_subject": float(sm - sp),
        "flip_auc": _auc(flip_true, flip_score),
        "flip_rate": float(flip_true.mean()),
        "flip_prec_0.5": prec05, "flip_rec_0.5": rec05,
        "flip_prec_top5": prec5, "flip_rec_top5": rec5,
        "state_acc_all": acc_all,
        "state_acc_on_switch": acc_sw,
        "state_persist_label": acc_persist,
        "n_subjects": int(len(subj)),
    }
