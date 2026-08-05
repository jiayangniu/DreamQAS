# DreamQAS Architecture Specification (energy-surrogate WM + known-dynamics imagination)

> Purpose: pin down the entire model **from data → structure → flow → loss** in one pass, precise to the shape of every tensor and the value of every hyperparameter, **reproducible in code as written**.
> Code location: `phase2_surrogate/` (`config.py` / `surrogate_wm.py` / `imagine.py` / `buffer.py` / `runner.py`) + reuses `environment.py` / `agent/wm_policy.py` / `circuit_rules.py`.
> Default configuration = **canonical full method "AC"** (single source of truth: `phase2_surrogate/config.py`): `independent_ensemble=1, wm_ensemble_K=3, rpf_beta=3, dagger=1, dir_reweight=1, pessimism_beta=1, fidelity_tau=0.70, imag_horizon=15, reward_kind=energy` — and **`pot_head=0, popart=0`** (both proven harmful, dropped 2026-07). Each ablation flips ONE flag from this base; a startup `config_hash` + canonical-diff (`config.mechanism_diff`) guards against old-default drift across runs.

> **Empirical status (2026-07-18, from `campaign_v1` — spec below unchanged; sources: `code/WM_QAS/analysis/outputs/main_results/`):**
> - **Confidence/uncertainty machinery is molecule-dependent, not universally load-bearing**: ensemble disagreement–error corr ≈0 on LiH-4q but **0.74–0.75 on BeH₂-8q/10q**; on 8q/10q the DAgger disagreement selector correctly targets WM blind spots (MAE top-value 0.12–0.14 vs top-disagree 0.81–0.98). Do not read the σ-gate (`τ_σ`) as verified-effective on small molecules — system reliability there comes from DAgger + the fidelity gate.
> - **DAgger** (§ flow below): ≈neutral on final best-of-1, does **not** heal point-calibration, and (corrected 2026-07-22, honest W=100 accounting) buys **no CI-clean VQE-efficiency** (`DAGGER_EFFICIENCY_W100_AUDIT.md`; the earlier "1.28–1.48×" was a W=400/DAgger-excluded artifact) — its empirical role is **verification + a collapse safety-net** (rescues the 1/5 BeH₂-6q imagination collapse), not efficiency and not a quality booster.
> - **The WM is a decision-useful ranker, not a calibrated oracle**: pairwise ranking fidelity 0.77–0.95 vs calibration MAE 0.22–0.73 log₁₀ (1.7–5.3×); the action-ranking probe (T1.a, 5-seed imag-start→¼-budget) shows per-prefix ρ grows to 0.39–0.53 and the top-ranked next action is near-optimal on the harder molecules (regret 0.03–0.18). Multi-step horizon fidelity (v2 probe, 1/4-budget, 5 seeds) shows **no significant trend with H** (pooled slope −0.006±0.021 /H, p=0.28) — the empirical face of exact dynamics (no learned-transition error accumulates). *(The v1 "flat 0.91/0.23/0.22" was a retracted truncation artifact.)*
> - **Imagination strength (`imag_n_seeds`, `imag_loss_weight`) is non-monotone and molecule-dependent** (dose sweep, 2 mol × 2 budgets, 2026-07-21; `IMAGINATION_STRENGTH.md` §3): the canonical `imag_n_seeds=64, imag_loss_weight=1` is a *safe common* point but not the per-molecule optimum — LiH-4q converges best at λ=10 (imagined-gradient share ~65%, best-of-1 0.009 at full budget), whereas on BeH₂-6q λ≥4 **collapses** and λ=1 is optimal. Over-broad imagination (M=256) hurts at convergence on both. Read λ/M as task-tunable, not universal.
> - Code guard added 2026-07-17: `imagine.py::_ens_std` returns zeros for a single-member ensemble (`wm_ensemble_K=1`), so single-model ablations cannot NaN the σ-dependent paths.
> - The "next steps (8q/10q crossover)" pointer in the eval note below is the 07-10 framing — superseded; see `RESEARCH_PLAN.md §0.5 (2026-07-18)`.

---

## 0. One-sentence architecture

> **Qualitative (locked, 2026-07-10): DreamQAS is model-based RL on a known-dynamics / energy-feedback world model.** The circuit prefix **evolves exactly in symbolic space** (circuit rules, `Shadow`); **the only thing learned is the energy surrogate** (predicting the prefix's post-VQE log-error + epistemic uncertainty). It **does not learn** transition dynamics, and **does not learn** variational-angle dynamics — the world model is **structure-level**, folding VQE parameter optimization into the energy feedback. **The composite (exact dynamics + surrogate) supports multi-step VQE-free imagination rollouts** to train the policy. Lineage: between **AlphaZero** (known dynamics + learned value) and **Dreamer** (imagination policy gradient); unlike **MuZero**, the latent dynamics are not learned. → We cannot claim to have learned a complete environment model or parameter dynamics, but we can legitimately claim **imagination-based policy learning with a structure-level world model**.

Treat QAS, which builds the circuit sequentially, as finite-step RL. The **energy-surrogate world model (WM)** is a GRU encoder + K-head ensemble that supervises regression of the circuit's `log10(true error)`; it simultaneously outputs **① representation h_t (fed to the actor) ② value ȳ_t (scores imagination) ③ uncertainty σ_t (gating/verification/pessimism)**. The policy (actor) picks gates on the WM representation, updated jointly by **real REINFORCE (spends VQE)** + **imagined REINFORCE (known-dynamics rollout, free)**; imagination is only enabled when the WM's self-check fidelity reaches the threshold, and stays honest via **DAgger (VQE-verified imagined circuits fed back)**. The VQE savings come from: imagination providing extra policy gradient, and the shared representation improving sample efficiency.

> **Evaluation metric (locked): best_err is biased (rewards luck, not learning)** → the main metric switches to **best-of-K@final-policy (K=1 = pure policy quality)** + **VQE-to-target-policy-level**. Cross-molecule conclusions, publication strategy, and next steps (8q/10q crossover) are in [`RESEARCH_PLAN.md §0`](../RESEARCH_PLAN.md).

```
        ┌──────────── real episode (1 VQE per step) ────────────┐
 actor ─▶ pick gate a_t ─▶ env.step(VQE) ─▶ true error ε_t ─▶ buffer.add
   ▲          │ h_t (detach)                              │ (kept in full)
   │ real REINFORCE  reward = −Δy (true value)            ▼
   │ (+ imag, fidelity-gated)                     WM supervised refresh (every R=20 iter × 200 steps)
   │          ▲                                    Huber + DIR + per-head bootstrap
   │ imag REINFORCE ◀── known-dynamics rollout ◀────────────────────────── energy-surrogate WM
   │  (free gradient)   Shadow legality mask + WM.step                     ├─ h_t : representation → actor (detach)
   │                    pessimistic potential Φ + confidence gate τ_σ      ├─ ȳ_t : value → imagination scoring
   │                            │                                          └─ σ_t : uncertainty → gate/verify/pessimism
   │              DAgger: every C=20 iter VQE-verify top-ȳ / top-σ imagined circuits → buffer (counts against budget)
```

**Gradient isolation (key invariant)**: the actor always eats `h_t.detach()`; the WM is trained only by its own supervised loss (`wm_opt`). The two share the **representation**, not the **gradient**. The actor and the WM each have an independent optimizer.

---

## 1. Notation quick-reference (locked)

| Symbol | Meaning | Code name |
|---|---|---|
| $n$ / $A$ / $L$ | qubit count / action count=$n(n{+}2)$ / max layers | `N` / `A` / `num_layers` |
| $a_t\in\mathcal A$ / $s_t\equiv a_{1:t}$ | action / state=prefix | action idx |
| $E(\cdot)$ / $E_0$ / $\varepsilon_t=\lvert E-E_0\rvert$ | energy / ground state / true error (mHa) | `energy`/`min_energy`/`true_error` |
| $\varepsilon_{\min}{=}10^{-3}$ / $\varepsilon_{\rm chem}{=}1.6$ | error floor / chemical accuracy | `err_floor_mHa` |
| $\mathbf m_t\in\{0,1\}^A$ | illegal-action mask (1=illegal) | `illegal_action_new` |
| $N_{\rm VQE}$ | budget = cumulative VQE calls | `vqe_calls` |
| $y_t=\log_{10}\max(\varepsilon_t,\varepsilon_{\min})$ | **log-error (WM target & real reward potential)** | — |
| $\pi_\theta$ / $\bar h_t$ | policy (parameters $\theta$) / actor feature (= mean of WM hidden state, detach) | `actor` / `feat` |
| $f^{(k)}_\phi,g^{(k)}_\phi,p^{(k)}$ | member $k$'s GRU / head / **frozen** RPF prior | `grus/heads/priors[k]` |
| $h^{(k)}_t$ / $\bar h_t{=}\frac1K\sum_k h^{(k)}_t$ | member hidden state / mean (representation) | — |
| $\hat y^{(k)}_t{=}g^{(k)}(h^{(k)}_t){+}\beta_{\rm rpf}p^{(k)}(h^{(k)}_t)$ | member prediction (RPF) | `_pred_k` |
| $\bar y_t{=}\frac1K\sum_k\hat y^{(k)}$ / $\sigma_t{=}\mathrm{std}_k\hat y^{(k)}$ | ensemble mean (value) / disagreement (uncertainty) | `predict` mean/std |
| $K$ / $\beta_{\rm rpf}$ / $\beta_{\rm pes}$ | ensemble size / RPF scale / pessimism weight | `wm_ensemble_K/rpf_beta/pessimism_beta` |
| $\mathcal B$ / $w,w_{\max}$ | replay buffer (full) / DIR inverse-density weight | `ReplayBuffer` / `dir_*` |
| $H$ / $M$ / $W$ | imagination steps / seed count / frontier window | `imag_horizon/n_seeds/seed_window` |
| $\Phi_t{=}-(\bar y_t{+}\beta_{\rm pes}\sigma_t)$ | pessimistic potential | `phi` |
| $\tau_\sigma$ / $\lambda$ / $\gamma$ / $\omega_{\rm im}$ | confidence stop / λ-return / discount / imagination loss weight | `imag_conf_tau/lambda/gamma/loss_weight` |
| $\mathrm{fid}$ / $\tau_{\rm fid}$ | held-out pairwise ranking accuracy / imagination-on threshold | `fidelity` / `fidelity_tau` |
| $R$ / $C$ / $n_{\rm warm}$ | WM refresh period / verification period / warmup episodes | `wm_refresh_every/calib_every/warmup_eps` |
| $n_{\rm top},n_{\rm dis}$ / $n_{\rm ep}$ / $i$ | verification top-ȳ/top-σ count / real episodes per iteration / iteration index | `calib_n_top/disagree`/`real_eps_per_iter`/`it` |

---

## 2. Per-molecule constants (given by env, not learned by us)

| Quantity | LiH4q | LiH6q | Source |
|---|---|---|---|
| `num_qubits` $n$ | 4 | 6 | cfg `[env]` |
| `action_size` $A=n(n{+}2)$ | 24 | 48 | `dictionary_of_actions(n)` |
| `num_layers` $L$ (episode step cap) | 40 | 50 | cfg `[env]` (BeH2 6q/8q/10q = 50) |
| `n_iterations` | 1000 | 1500 | `N_ITERS` |
| `wm_hidden` $D` | 256 | 384 | runner (6q widened) |
| `fake_min_energy` $E_0$(Ha) | −10.0717 | −10.12 | cfg (VQE uses it to compute error) |
| geometry / mapping | Li..H 3.4Å / parity | jordan_wigner | cfg `[problem]` |

**VQE cost model**: `env.step(gate)` internally runs COBYLA angle optimization (`global_iters=1000`, CPU) and returns the energy; **each added gate = 1 VQE call**, `vqe_calls += 1`. Budget = total `vqe_calls`.

**Action encoding** `_dict_actions(n): idx → [ctrl, offset, rot_qubit, rot_axis]`:
- CNOT: `[c, o, n, 0]`, control=`c`∈[0,n), target=`(c+o)%n`, o∈[1,n) → `n(n-1)` in total (the first `n(n-1)` idx).
- Rotation: `[n, 0, q, axis]`, qubit=`q`∈[0,n), axis∈{1,2,3}={Rx,Ry,Rz} → `3n` in total.
- Illegality rule `compute_illegal` (a bit-for-bit port of `CircuitEnv.illegal_action_new`): cannot place consecutive rotations on the same qubit, recursive CNOT constraints, etc.; `Shadow` mirrors it bit-exactly (`tests/test_circuit_rules.py` guarantees bit-exact).

---

## 3. Data structures and shapes

### 3.1 Single episode (`collect_episode` output, feeds buffer + real REINFORCE)
```python
ep = {
  "acts":        list[int]          # length T ≤ L+1, each ∈ [0,A)
  "errs":        float32[T]         # per-step true error ε_t (mHa)
  "feats":       Tensor[T, D]       # per-step actor input feature h_t (already detached)
  "masks":       bool[T, A]         # per-step illegal mask m_t (True=illegal)
  "env_rewards": float32[T]         # env's own reward (curriculum uses it; energy does not)
  "disagree":    float32[T]         # ensemble disagreement σ at each step's reached state (P3 curiosity/diagnostic)
}
```

### 3.2 Buffer internal entry (`buffer.add`)
```python
eps[i] = {
  "acts":    int64[T]     # action sequence
  "err":     float32[T]   # true-error trajectory
  "idx":     int          # arrival order (for recency)
  "min_err": float        # min(err), for elite ranking & quality priority
  "wm_err":  float        # this entry's most recent WM prediction error (active-learning priority, initial 1.0)
}
```
- **Energy target is stationary** (the circuit's true error never changes) → **keep everything**, never drop old data.
- `elite_N=1000`; the actual buffer has no hard cap (`buffer_max_episodes` is a legacy cfg leftover, unused by the v2 runner).

### 3.3 Training batch (`sample_batch` → `_collate`, `B=batch_size=64`)
| Tensor | Shape | Description |
|---|---|---|
| `acts` | `long[B, T]` | `T=max len in batch`, right-side 0 padding |
| `err`  | `float[B, T]` | true error, 0 at padding |
| `mask` | `float[B, T]` | valid position 1 / padding 0 |

**Batch composition** (proportions, summing to 1): elite 0.25 + prioritized 0.35 + stratified 0.20 + random 0.20.
- elite: sampled from the top `elite_N` entries with smallest `min_err`.
- prioritized: `p ∝ recency × quality(1/(min_err+1)) × (0.1+wm_err)`.
- stratified: uniformly sampled across `min_err` deciles (ensures cross-scale comparison, prevents resolution collapse after convergence).
- random: uniform.

---

## 4. Module architecture (input/output/dimension per structure)

### 4.1 Energy-surrogate WM — `EnsembleSurrogateWM` (default, `independent_ensemble=True`)

**K=3 fully independent members**, each member:
| Submodule | Definition | Parameter shape |
|---|---|---|
| `embeds[k]` | `nn.Embedding(A, 64)` | `[A, 64]` |
| `grus[k]` | `nn.GRU(64, D, num_layers=2, batch_first, dropout=0.1)` | 2-layer GRU |
| `heads[k]` | `_head(D)` (trainable) | see below |
| `priors[k]` | `_head(D)` (**frozen**, `requires_grad=False`) → RPF | see below |

`_head(D) = Sequential( LayerNorm(D) → Linear(D,128) → SiLU → Linear(128,1) )`, outputs a scalar.
PopArt buffer: `pa_mean` (scalar), `pa_std` (scalar) — identity when `popart=0`, ignored in the main line.

**Member prediction (RPF)**: `_pred_k(k, h) = (heads[k](h) + β_rpf · priors[k](h)).squeeze(-1)`. The frozen prior makes members **naturally disagree** where data is sparse → σ is an effective OOD signal.

**Method I/O shapes** (`B`=batch, `T`=sequence length, `M`=imagination seed count):

| Method | Input | Output | Purpose |
|---|---|---|---|
| `encode_all(acts)` | `long[B,T]` | `[K,B,T,D]` | per-member full-sequence hidden states |
| `encode_seq(acts)` | `long[B,T]` | `[B,T,D]` (K mean) | actor feature (training mode) |
| `head_preds(h_all)` | `[K,...,D]` | `[K,...]` (normalized) | per-member log-err prediction |
| `predict(acts)` | `long[B,T]` | `ȳ[B,T]`(denorm), `σ[B,T]`(denorm std), `h̄[B,T,D]` | inference/fidelity |
| `seq_preds(acts)` | `long[B,T]` | `[K,B,T]` (normalized) | **WM training target alignment** |
| `init_state(B)` | `B` | `[K, 2, B, D]` | imagination initial hidden |
| `encode_to_state(acts)` | `long[B,T]` | `h̄[B,D]`, `hstate[K,2,B,D]` | seed imagination from real prefix |
| `step(action, hstate)` | `action[B]`, `hstate[K,2,B,D]` | `feat[B,D]`(K mean), `new hstate[K,2,B,D]`, `preds[K,B]`(normalized) | **single-step known-dynamics advance** |

> **Known-dynamics consistency**: `step` advancing gate-by-gate == `encode_seq` encoding the whole segment (bit-identical in eval mode, `surrogate_wm.py __main__` smoke-checks `max|Δ|~0`). Transition is not learned.

(Alternative encoders: `SurrogateWM` = shared-trunk GRU, `RSSMSurrogateWM` = adds DreamerV3-style categorical latent; neither is the default, `build_wm` dispatches by `independent_ensemble`. The RSSM KL is only a representation regularizer, unused in the main line.)

### 4.2 Actor — `WMDiscreteActor` (+ optional `PotentialActor`)

```python
WMDiscreteActor(feature_dim, action_size=A, hidden_dim=512, num_layers=3):
    net = MLP(feature_dim, A, 512, 3)     # Linear→LN→SiLU ×2 → Linear(512,A)
```
- `feature_dim = D + (1 if pot_head else 0)`.
- `forward(features[B, feat_dim], ill_mask[B,A]) → Categorical(logits[B,A])`, illegal positions `masked_fill(-1e9)`.

**PotentialActor (wraps a layer when `pot_head=1`, M2)**:
```python
pot = Sequential( Linear(D,128) → SiLU → Linear(128,1) )     # predicts episode return
forward(feat, mask) = base_actor( cat([feat, pot(feat)], -1), mask )   # internally builds base with feature_dim=D+1
pot_value(feat) = pot(feat).squeeze(-1)                       # regressed to return by real_reinforce
```
It is **transparent** to `collect_episode`/`imagine.py` (same `(feat, mask)` interface).

---

## 5. Full flow (one iteration `run()`, with shapes)

Each iteration `i`: first run `n_ep=4` real episodes → real REINFORCE (+imagination) updates the actor → periodically refresh the WM / verify.

### 5.1 Collect real episode `collect_episode` (spends VQE, `wm.eval()`, no_grad + detach throughout)
```
env.reset(); hstate = wm.init_state(1) = [K,2,1,D]; feat = zeros(1, D)   # empty-circuit feature
for step in 0..L:
    m_t = illegal_action_new()                     # bool[1,A]
    a   = actor(feat, m_t).sample()                # [1], no_grad
    env.step(gate(a));  vqe_calls += 1             # run VQE
    ε_t = |min_energy − energy| × 1000             # mHa
    store acts[t]=a, errs[t]=ε_t, feats[t]=feat.squeeze(0).detach()(=h_t, [D]), masks[t]=m_t([A])
    feat, hstate, ph = wm.step(a, hstate)          # ph[K,1] → disagree[t]=ph.std(0)
    if done: break
buf.add(acts, errs); n_eps += 1
```
Produces `ep` (see §3.1). `best_err = min(best_err, ε_t)` updated in real time.

### 5.2 Reward `_returns(ep)` (`reward_kind=energy`)
```
y   = log10(clip(errs, ε_min))          # [T], exactly the y_t of §1
r[0]=0; r[t] = y[t-1] − y[t]  (t≥1)      # potential shaping, Φ_real=−y: reduction in log-error
G[t] = r[t] + γ·G[t+1]  (reverse order, γ=0.99)   # discounted return
```
(`staircase` / `curiosity_beta` are ablation branches, not enabled in the main line.)

**[2026-07-27] Oracle-free variant (`oracle_free=1`, default OFF — `escale.py`).** Replaces the *training*
primitive so that no exact ground-state energy E0 is needed anywhere in training. With `F_adopted` = best
real-VQE energy adopted so far (updated only at a refresh boundary, atomically with WM retraining):
```
d    = (E − F_adopted)·1e3                       # mHa above the adopted empirical frontier
S(E) = sgn(d)·log10(1 + |d|/m)                   # symmetric signed-log; m = score_margin_mHa (0.1)
r[t] = S[t-1] − S[t]  (Φ = −S)                   # same telescoping shape as the canonical reward
```
For `d ≥ 0` this equals the canonical log-error form up to a constant → identical step rewards there; for
`d < 0` (a new best) growth is bounded-log instead of a linear blow-up, C1 at `d=0`, strictly monotone.
`S` also becomes the WM label, the DIR-reweighting axis, and the priority/rung axis; E0 survives only as a
diagnostic (`enable_ground_truth_diagnostics`) and in evaluation. **It is a variant, not an equivalence**:
on the hardest non-saturated task (LiH-6q) it costs ~33% final best-of-1 quality because the *reference*
moves — evidence and guardrails in `analysis/outputs/main_results/SUPPLEMENTARY_EXPERIMENTS.md` §1.

### 5.3 Real REINFORCE `real_reinforce(eps)`
```
concat 4 episodes: feats[ΣT,D], masks[ΣT,A], acts[ΣT], rets_raw[ΣT]=concat(G)
adv = (rets_raw − mean) / (std + 1e-6)                      # standardized advantage
dist = actor(feats, masks); logp = dist.log_prob(acts)     # [ΣT]
L_real = −(logp·adv).mean() − c_ent·entropy.mean()         # c_ent=1e-3
if pot_head: L_real += MSE( pot_value(feats), rets_raw )    # value-head auxiliary
```

### 5.4 Imagination `imagine_and_loss` (only `imag_on=True`, `wm.eval()`, actor eats detached features)
```
seeds = buf.imag_seeds(M=64, W=200, strategy)              # M real prefixes (random truncation)
hstate[K,2,M,D], feat[M,D], preds0[K,M] = _encode_seeds(seeds)   # step each prefix to its tip
shadows = _build_shadows(seeds)                            # M legality trackers
Φ_prev = −(preds0.mean(0) + β_pes·preds0.std(0))          # [M]
alive  = ones(M)
for t in 0..H-1:                                           # H=imag_horizon (canonical 15)
    m_t[M,A] = each shadow's illegal mask
    alive &= (~m_t).any(1)                                 # no legal action → dead
    a = actor(feat.detach(), m_t).sample(); logps += logp(a)   # [M]
    for alive rows shadow.commit(a)
    feat, hstate, ph = wm.step(a, hstate)                  # ph[K,M]
    σ  = ph.std(0);  Φ = −(ph.mean(0) + β_pes·σ)           # [M]
    alive &= (σ ≤ τ_σ)                                     # confidence gate (τ_σ=0.60): update first, then record reward
    rewards += (Φ − Φ_prev)·alive.float();  Φ_prev = Φ     # distrusted/dead step → 0
    if alive.sum()==0: break
# λ-return (terminal bootstrap 0)
R=0; for t in reversed: R = rewards[:,t] + γ·λ·R; returns[:,t]=R      # γ=0.99, λ=0.95
adv = returns − returns.mean(0, keepdim)                   # per-step batch-mean baseline
if imag_adv_normalize: adv /= std(adv[alive])              # whiten to O(1) (off by default; on for depth experiments)
L_imag = −(logps · adv.detach() · alive_mask).sum() / alive_mask.sum()
```
`imag_grad_frac = ‖∇_θ L_imag‖ / (‖∇_θ L_real‖ + ‖∇_θ L_imag‖)` (measured once every R steps, the P1 headline metric).

### 5.5 Actor update (merged)
```
L = L_real + [imag_on] · ω_im · L_imag        # ω_im=imag_loss_weight (baseline 1)
L.backward(); clip_grad_norm(actor, 10); actor_opt.step()
```

### 5.6 WM refresh + fidelity gate (every `R=20` iter, and `len(buf)≥8`; `wm.train()`)
```
fid = fidelity_check()                        # see below
imag_on = (imagination==surrogate) and (n_eps ≥ n_warm=500) and (fid ≥ τ_fid=0.70)
# wm_refresh(200 steps):
dir_edges, dir_binw = _dir_bin_weights()       # compute inverse-density weights once over the whole buffer
repeat 200:
    acts[B,T], err[B,T], mask[B,T] = sample_batch()
    y = log10(clip(err, ε_min))
    w = dir_binw[bucketize(y, dir_edges)]       # DIR inverse-density (else legacy tail weight)
    update_popart(y[mask]); y_t = normalize(y)  # popart=0 → identity
    preds = seq_preds(acts)                     # [K,B,T] normalized
    loss = 0
    for k in K:
        bw = (rand[B,T] < 0.8).float() · mask   # 80% bootstrap per head → ensemble diversity
        e  = preds[k] − y_t
        loss += ( huber(e)·w·bw ).sum() / bw.sum()      # huber: |e|≤1→0.5e², else |e|−0.5
    loss = loss/K + kl_scale·kl_loss            # ensemble kl_loss=0
    wm_opt.zero_grad(); loss.backward(); wm_opt.step()
```

### 5.7 fidelity self-check `fidelity_check` (held-out pairwise ranking)
```
acts,err,mask = buf.recent_val(min(200, len(buf)))     # most recent ≤200 episodes
pred,_,_ = wm.predict(acts)                            # ȳ
sc, tr = pred[mask], err[mask]                         # flatten valid steps
sample 50000 pairs (i,j); ok = tr[i]≠tr[j]
fid = mean( sign(sc[i]−sc[j]) == sign(tr[i]−tr[j]) )[ok]     # ∈[0,1], 0.5=random guess
```

### 5.8 DAgger verification `calibrate` (every `C=20` iter, surrogate variant)
```
seeds = buf.imag_seeds(64, 200, strategy)
cands = sample_imagined_circuits(...)                 # per seed, roll out a full circuit + pred_logerr + σ
uniq  = dedup(cands, drop n_imagined==0)
sel   = pick nt=5 with lowest pred_logerr (top) + nd=5 with highest σ (disagree)   # dagger_select=mix
for c in sel:
    tes, applied = _vqe_circuit(c.seq, count_budget=dagger)         # actually run VQE
        # dagger=1: vqe_calls++ per step (counts against budget), best_err may be refreshed, buf.add(applied, tes)
    abs_err = |c.pred_logerr − log10(max(tes[-1], ε_min))|
record calib_MAE, spearman(σ, abs_err)                # WM accuracy & confidence-gate effectiveness diagnostic
```
**DAgger's VQE counts against the budget** → the reported speedup already includes the "stay honest" cost.

### 5.9 Main-loop skeleton
```python
for it in range(n_iterations):
    eps = [collect_episode() for _ in range(4)]        # §5.1
    actor_opt.zero_grad()
    loss = real_reinforce(eps)                         # §5.3
    if imagination=="surrogate" and imag_on:           # §5.4–5.5
        iloss = imagine_and_loss(...);  loss += ω_im * iloss
    loss.backward(); clip_grad_norm(actor,10); actor_opt.step()
    if it % 20 == 0 and len(buf) >= 8:                 # §5.6–5.7
        fid = fidelity_check(); set imag_on; wm_refresh(200)
    if imagination=="surrogate" and it>0 and it%20==0 and len(buf)>=8:
        calibrate(it)                                  # §5.8
    log metrics
```
**Timing**: `n_warm=500` episodes = the first 125 iter (4 ep/iter) with imagination force-off; after that it additionally needs `fid≥0.70` to turn on. The WM trains once every 20 iter starting from iter 0.

---

## 6. Loss summary (three independent objectives)

**① actor (`actor_opt`, parameters θ)**
$$\mathcal L(\theta)=\underbrace{-\mathbb E[\,\log\pi_\theta(a)\,\hat A^{\rm re}\,]-c_{\rm ent}\,\mathbb E[\mathcal H(\pi_\theta)]}_{\mathcal L_{\rm real}}\;+\;\mathbb 1[\mathrm{fid}\ge\tau_{\rm fid}]\;\omega_{\rm im}\underbrace{\Big(-\tfrac{\sum \log\pi_\theta(a)\,\hat A^{\rm im}\,\mathbb 1_{\rm alive}}{\sum \mathbb 1_{\rm alive}}\Big)}_{\mathcal L_{\rm imag}}$$
- $\hat A^{\rm re}=(G-\bar G)/\mathrm{std}(G)$, $G$ uses the true value $y$: $r^{\rm re}_t=y_{t-1}-y_t$.
- $\hat A^{\rm im}=$ λ-return$(\Delta\Phi)$ with per-step baseline removed (optional whitening), $\Phi_t=-(\bar y_t+\beta_{\rm pes}\sigma_t)$.
- with `pot_head`, additionally $+\,\mathrm{MSE}(v_\xi(\bar h),G)$.

**② WM (`wm_opt`, parameters φ)**
$$\mathcal L_{\rm WM}(\phi)=\frac1K\sum_{k}\frac{\sum_{b,t} \mathrm{Huber}\!\big(\hat y^{(k)}_{bt}-\tilde y_{bt}\big)\,w_{bt}\,\mathrm{bw}^{(k)}_{bt}}{\sum \mathrm{bw}^{(k)}}\;(+\,\kappa\,\mathrm{KL}\text{, RSSM only})$$
$\tilde y$ = (optionally PopArt-normalized) target log-error; $w$ = DIR inverse-density weight; $\mathrm{bw}^{(k)}$ = per-head 80% bootstrap mask. **Supervised only, no actor gradient attached.**

---

## 7. Full hyperparameter table

| Group | Name | Default/baseline value | Note |
|---|---|---|---|
| RL | `real_eps_per_iter` | 4 | |
| | `actor_hidden/layers/lr` | 512 / 3 / 3e-5 | |
| | `entropy_coef` `reinforce_gamma` `grad_clip` | 1e-3 / 0.99 / 10 | |
| WM | `wm_embed` `wm_hidden` `wm_gru_layers` | 64 / 256(4q)·384(6q) / 2 | |
| | `wm_ensemble_K` `rpf_beta` | 3 / 3.0 | RPF |
| | `wm_lr` `wm_weight_decay` | 1e-3 / 1e-5 | |
| | `popart`(main line) | **0** | off |
| Training | `warmup_eps` `wm_refresh_every` `wm_refresh_steps` | 500 / 20 / 200 | |
| | `fidelity_tau` | 0.70 | imagination-on threshold |
| | `batch_size` `elite_N` | 64 / 1000 | |
| | frac elite/prior/strat/rand | .25/.35/.20/.20 | |
| DIR | `dir_reweight` `dir_bins` `dir_smooth` `dir_w_max` | 1 / 40 / 2.0 / 50 | |
| Imagination | `imag_horizon` H | **15** (canonical; horizon sweep 0/5/10/15/20) | config default 15 |
| | `imag_n_seeds` M / `imag_seed_window` W | 64 / 200 | |
| | `imag_lambda` `imag_gamma` `imag_conf_tau` | 0.95 / 0.99 / 0.60 | |
| | `pessimism_beta` β_pes | 1.0 | |
| | `imag_adv_normalize` `imag_loss_weight` ω_im | 0 / 1.0 | depth experiments turn on advnorm |
| | `imag_seed_strategy` | frontier | |
| DAgger | `calib_every` C / `calib_n_top` / `calib_n_disagree` | 20 / 5 / 5 | |
| | `dagger` `dagger_select` | 1 / mix | |
| Omitted | `curiosity_beta` `reward_kind=staircase` | 0 / — | ablation, omitted from the main line |
| Oracle-free *(2026-07-27, default OFF)* | `oracle_free` `score_margin_mode` `score_margin_mHa` | **0** / fixed / 0.1 | E0-free training score (§5.2 note); `enable_ground_truth_diagnostics=0` fully severs `env.min_energy` |
| Supplement switches *(default OFF = byte-identical legacy)* | `select_mode` `wm_greedy_eps` | **actor** / 0.1 | `wm_greedy` = use the WM as a 1-step pessimistic candidate selector instead of the actor (reviewer contrast, NOT the method) |
| | `imag_transition_budget` T | **0** | T>0: generate imagined trajectories until ≥T *trusted* transition loss-terms accrue, then randomly downsample to exactly T — matches training-signal quantity across `imag_horizon` |
| | `real_dose_alpha` α | **0.0** | proposed on-policy real-REINFORCE dose control; implemented + smoke-tested, **not launched** |

---

## 8. Tensor-shape cheat-sheet

| Tensor | Shape | Where it appears |
|---|---|---|
| gate embedding | `[A,64]` per member | `embeds[k]` |
| GRU hidden | `[K,2,B,D]`(state) / `[K,B,T,D]`(seq) | WM |
| actor feature $\bar h_t$ | `[B,D]` (collection B=1; REINFORCE B=ΣT) | `feat` |
| actor logits | `[B,A]` | `WMDiscreteActor.forward` |
| WM member prediction | `[K,B,T]`(seq) / `[K,B]`(step) | `seq_preds`/`step` |
| ensemble mean/disagreement | `[B,T]` / `[B,T]` | `predict` |
| training batch | `acts[64,T] err[64,T] mask[64,T]` | `sample_batch` |
| imagination trajectory | `logps/rewards/returns/alive [M,H]` | `imagine_and_loss` |
| imagination hstate | `[K,2,M,D]` | `_encode_seeds` |

---

## 9. Key invariants / pitfalls

1. **Gradient isolation**: the actor input is `.detach()`-ed everywhere; the WM is supervised-trained only by `wm_opt`. When changing things, don't let the actor gradient flow back into the WM.
2. **Known-dynamics consistency**: `Shadow` (legality mask) + `wm.step` (state advance) must be bit-identical to `env` / `encode_seq` (guarded by a test).
3. **Imagination turns on only when all three gating conditions hold**: `surrogate` ∧ `n_eps≥500` ∧ `fid≥0.70`; and `imag_on` updates only once every 20 iter.
4. **Energy target is stationary → keep the buffer in full**; sensitivity comes from elite/prioritized/stratified/DIR, not from dropping old data.
5. **Budget accounting**: real episode +1 per step; DAgger verification +1 per step (counted); in `calibrate` the non-dagger diagnostic VQE is recorded in `calib_vqe_calls` (**not** counted toward the speedup budget).
6. **`best_err` can be refreshed by DAgger** (an imagined circuit verified by real VQE is recorded if it's better) — symmetric: it both spends budget and can discover.
7. **Reward potential shaping**: `Φ_real=−y` (true value), `Φ_imag=−(ȳ+β_pes σ)` (pessimistic estimate) — the two arms are isomorphic, differing only in "true vs. estimated".
8. **per-molecule widening/depth** (`config.WM_HIDDEN` / cfg `[env]`): `wm_hidden` 4q=256 / 6q·8q=384 / 10q=512; `num_layers` 4q=40 / 6q·8q·10q=50; everything else the same.
