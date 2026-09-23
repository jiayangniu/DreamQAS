import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from circuit_rules import Shadow, _dict_actions


def _ens_std(x):
    return x.std(0) if x.shape[0] > 1 else torch.zeros_like(x[0])


@torch.no_grad()
def _encode_seeds(wm, seeds, device):
    N = len(seeds)
    hstate = wm.init_state(N, device)
    feats = torch.zeros(N, wm.hidden, device=device)
    preds0 = torch.zeros(wm.K, N, device=device)
    for i, pre in enumerate(seeds):
        hs = wm.init_state(1, device)
        f = p = None
        for a in pre:
            f, hs, p = wm.step(torch.tensor([int(a)], device=device), hs)
        hstate[..., i, :] = hs[..., 0, :]
        feats[i] = f[0]
        preds0[:, i] = p[:, 0]
    return hstate, feats, preds0


def _encode_seeds_batched(wm, seeds, device):
    N = len(seeds)
    hstate = wm.init_state(N, device)
    feats = torch.zeros(N, wm.hidden, device=device)
    preds0 = torch.zeros(wm.K, N, device=device)
    lens = [len(s) for s in seeds]
    T = max(lens) if lens else 0
    if T == 0:
        return hstate, feats, preds0
    acts = torch.zeros(N, T, dtype=torch.long, device=device)
    for i, pre in enumerate(seeds):
        if len(pre):
            acts[i, :len(pre)] = torch.as_tensor(
                [int(a) for a in pre], dtype=torch.long, device=device)
    lens_t = torch.as_tensor(lens, device=device)
    cur = wm.init_state(N, device)
    for t in range(T):
        f, cur, p = wm.step(acts[:, t], cur)
        done = (lens_t == t + 1)
        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            feats[idx] = f[idx]
            preds0[:, idx] = p[:, idx]
            hstate[..., idx, :] = cur[..., idx, :]
    return hstate, feats, preds0


def _build_shadows(seeds, n_qubits):
    tr = _dict_actions(n_qubits)
    shadows = []
    for pre in seeds:
        S = Shadow(n_qubits)
        for a in pre:
            S.mask_indices()
            S.commit(tr[int(a)])
        shadows.append(S)
    return shadows


def _rollout_core(wm, actor, seeds, n_qubits, cfg, device, max_depth=None):
    wm.eval()
    N = len(seeds)
    action_size = actor.action_size
    tr = _dict_actions(n_qubits)

    beta = getattr(cfg, "pessimism_beta", 0.0)
    _enc = (_encode_seeds_batched if getattr(cfg, "imag_batched_encode", True)
            else _encode_seeds)
    hstate, feat, preds0 = _enc(wm, seeds, device)
    shadows = _build_shadows(seeds, n_qubits)
    phi_prev = -(preds0.mean(0) + beta * _ens_std(preds0))

    logps, rewards, alive_hist, disag_hist = [], [], [], []
    alive = torch.ones(N, dtype=torch.bool, device=device)
    dcap = getattr(cfg, "imag_depth_cap", False) and max_depth is not None
    remain = torch.tensor([max_depth - len(s) for s in seeds], device=device) if dcap else None

    for t in range(cfg.imag_horizon):
        mask_t = torch.zeros(N, action_size, dtype=torch.bool, device=device)
        for i, sh in enumerate(shadows):
            ids = sh.mask_indices()
            if ids:
                mask_t[i, ids] = True
        alive = alive & (~mask_t).any(1)
        if dcap:
            alive = alive & (remain > t)
        dist = actor(feat.detach(), mask_t)
        action = dist.sample()
        logps.append(dist.log_prob(action))

        acts_cpu = action.cpu().tolist()
        alive_cpu = alive.cpu().tolist()
        for i, sh in enumerate(shadows):
            if alive_cpu[i]:
                sh.commit(tr[acts_cpu[i]])
        feat, hstate, ph = wm.step(action, hstate)
        disagree = _ens_std(ph)
        phi = -(ph.mean(0) + beta * disagree)
        alive = alive & (disagree <= cfg.imag_conf_tau)
        rewards.append((phi - phi_prev) * alive.float())
        phi_prev = phi
        disag_hist.append(disagree)
        alive_hist.append(alive.float())
        if alive.sum() == 0:
            break

    H = len(rewards)
    logps = torch.stack(logps[:H], 1)
    rewards = torch.stack(rewards, 1)
    alive_mask = torch.stack(alive_hist, 1)

    returns = torch.zeros_like(rewards)
    R = torch.zeros(N, device=device)
    for t in reversed(range(H)):
        R = rewards[:, t] + cfg.imag_gamma * cfg.imag_lambda * R
        returns[:, t] = R
    m = alive_mask
    adv = returns - returns.mean(0, keepdim=True)
    if getattr(cfg, "imag_adv_normalize", False):
        alive_adv = adv[m.bool()]
        std = alive_adv.std().clamp_min(1e-6) if alive_adv.numel() > 1 else adv.new_ones(())
        adv = adv / std
    with torch.no_grad():
        diag = {
            "imag_return": float((returns * m).sum() / m.sum().clamp_min(1)),
            "imag_advantage": float((adv * m).sum() / m.sum().clamp_min(1)),
            "imag_horizon_used": float(m.sum(1).mean()),
            "imag_disagree": float(torch.stack(disag_hist).mean()),
            "n_seeds": N,
        }
    return logps, adv, m, diag


def imagine_and_loss(wm, actor, seeds, n_qubits, cfg, device, max_depth=None, seed_fn=None):
    if not seeds:
        return None, {}
    budget = int(getattr(cfg, "imag_transition_budget", 0))
    if budget <= 0:
        logps, adv, m, diag = _rollout_core(wm, actor, seeds, n_qubits, cfg, device, max_depth)
        loss = -(logps * adv.detach() * m).sum() / m.sum().clamp_min(1.0)
        diag["n_trusted"] = float(m.sum())
        return loss, diag
    lp_acc, adv_acc = [], []
    gen_seeds = 0; rolls = 0; raw_wm_q = 0; diag_last = {}
    cur_seeds = seeds
    MAX_ROLLS = 128
    while rolls < MAX_ROLLS:
        logps, adv, m, diag_last = _rollout_core(wm, actor, cur_seeds, n_qubits, cfg, device, max_depth)
        mb = m.bool()
        lp_acc.append(logps[mb]); adv_acc.append(adv[mb].detach())
        gen_seeds += len(cur_seeds); rolls += 1
        raw_wm_q += int(logps.shape[0] * logps.shape[1])
        if sum(x.numel() for x in lp_acc) >= budget or seed_fn is None:
            break
        cur_seeds = seed_fn()
        if not cur_seeds:
            break
    flat_lp = torch.cat(lp_acc); flat_adv = torch.cat(adv_acc)
    trusted = int(flat_lp.numel())
    if trusted >= budget:
        sel = torch.randperm(trusted, device=flat_lp.device)[:budget]
        flat_lp = flat_lp[sel]; flat_adv = flat_adv[sel]; used = budget
    else:
        used = trusted
        print(f"[imag_budget] WARNING: only {trusted}/{budget} trusted transitions after {rolls} rolls")
    loss = -(flat_lp * flat_adv).sum() / max(used, 1)
    diag = dict(diag_last)
    diag.update({"imag_budget": budget, "requested": budget, "generated_seeds": gen_seeds,
                 "trusted": trusted, "used": used, "rolls": rolls, "raw_wm_queries": raw_wm_q,
                 "n_trusted": float(used)})
    return loss, diag


@torch.no_grad()
def sample_imagined_circuits(wm, actor, seeds, n_qubits, cfg, device, max_depth=None):
    if not seeds:
        return []
    wm.eval()
    N = len(seeds)
    action_size = actor.action_size
    tr = _dict_actions(n_qubits)

    hstate, feat, preds0 = _encode_seeds(wm, seeds, device)
    shadows = _build_shadows(seeds, n_qubits)
    seqs = [list(map(int, s)) for s in seeds]
    n_imagined = [0] * N
    last_logerr = wm.denorm(preds0.mean(0))
    last_disagree = _ens_std(preds0)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    dcap = getattr(cfg, "imag_depth_cap", False) and max_depth is not None
    remain = torch.tensor([max_depth - len(s) for s in seeds], device=device) if dcap else None

    for t in range(cfg.imag_horizon):
        mask_t = torch.zeros(N, action_size, dtype=torch.bool, device=device)
        for i, sh in enumerate(shadows):
            ids = sh.mask_indices()
            if ids:
                mask_t[i, ids] = True
        alive = alive & (~mask_t).any(1)
        if dcap:
            alive = alive & (remain > t)
        dist = actor(feat, mask_t)
        action = dist.sample()
        acts_cpu = action.cpu().tolist()
        alive_cpu = alive.cpu().tolist()
        for i, sh in enumerate(shadows):
            if alive_cpu[i]:
                sh.commit(tr[acts_cpu[i]])
                seqs[i].append(acts_cpu[i])
                n_imagined[i] += 1
        feat, hstate, ph = wm.step(action, hstate)
        logerr = wm.denorm(ph.mean(0)); disagree = _ens_std(ph)
        for i in range(N):
            if alive_cpu[i]:
                last_logerr[i] = logerr[i]
                last_disagree[i] = disagree[i]
        alive = alive & (disagree <= cfg.imag_conf_tau)
        if alive.sum() == 0:
            break

    return [{"seq": seqs[i], "n_imagined": n_imagined[i],
             "pred_logerr": float(last_logerr[i]), "disagree": float(last_disagree[i])}
            for i in range(N)]


if __name__ == "__main__":
    from config import Config
    from surrogate_wm import SurrogateWM
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.wm_policy import WMDiscreteActor
    cfg = Config()
    N_Q, A = 4, 24
    dev = "cpu"
    wm = SurrogateWM(A, cfg).to(dev)
    actor = WMDiscreteActor(feature_dim=cfg.wm_hidden, action_size=A,
                            hidden_dim=cfg.actor_hidden, num_layers=cfg.actor_layers).to(dev)
    seeds = [torch.randint(0, A, (int(l),)).tolist() for l in torch.randint(2, 10, (16,))]
    loss, diag = imagine_and_loss(wm, actor, seeds, N_Q, cfg, dev)
    print("loss:", float(loss))
    print("diag:", {k: round(v, 4) for k, v in diag.items()})
    loss.backward()
    g = sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None)
    print(f"actor grad flows: {float(g):.4f} (should be >0); wm grad is None (detached): "
          f"{all(p.grad is None for p in wm.parameters())}")
