
from typing import Optional
import torch
import numpy as np
from utils import *
from sys import stdout
import scipy
import VQE as vc
import os
import copy
import curricula
from collections import Counter
from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import CNOT, RX, RY, RZ


class CircuitEnv():

    # Staircase reward ladder for fn_type = "staircase".
    # Each (threshold, value) gives `value` reward the first time per-episode
    # best_err crosses below `threshold`. Anchored so chem.acc (1.6 mHa) = 5.
    # See phase 2.5 reward redesign discussion for rationale.
    STAIRCASE_LADDER = (
        (5e-2,    2.0),   # 50 mHa
        (2e-2,    2.5),
        (1e-2,    3.0),
        (5e-3,    3.75),
        (1.6e-3,  5.0),   # chem.acc anchor
        (1e-3,    6.0),
        (5e-4,    7.5),
        (2e-4,    9.0),
        (1e-4,   11.0),   # 100 μHa
        (5e-5,   13.5),
        (2e-5,   16.5),
        (1e-5,   20.0),   # 10 μHa
        (5e-6,   24.0),
        (2e-6,   29.0),
        (1e-6,   35.0),   # 1 μHa
        (5e-7,   41.0),
        (2e-7,   49.0),
        (1e-7,   58.0),
    )

    # Aggressive exponential ladder for fn_type in {incremental_with_ladder, logprog_ladder}.
    # ×1.5 per stop, ×3.375 per decade. chem.acc anchored at 5.0.
    # Compared to STAIRCASE_LADDER:
    #   - 5 mHa 以上区间值 << 5 (base ±5 dominates there, by design)
    #   - chem.acc 以下区间指数放大 (1 μHa = 288, 100 nHa = 973)
    # Trade-off: stronger differentiation at high precision; small/large value
    # squashing risk handled by REINFORCE per-batch normalization + symlog reward heads.
    STAIRCASE_LADDER_V2 = (
        (5e-2,    0.66),  # 50 mHa
        (2e-2,    0.99),
        (1e-2,    1.48),
        (5e-3,    2.22),  # 5 mHa
        (2e-3,    3.33),
        (1.6e-3,  5.0),   # chem.acc anchor (unchanged)
        (1e-3,    7.50),  # 1 mHa
        (5e-4,   11.25),
        (2e-4,   16.88),
        (1e-4,   25.31),  # 100 μHa
        (5e-5,   37.97),
        (2e-5,   56.95),
        (1e-5,   85.43),  # 10 μHa
        (5e-6,  128.14),
        (2e-6,  192.22),
        (1e-6,  288.33),  # 1 μHa
        (5e-7,  432.50),
        (2e-7,  648.74),
        (1e-7,  973.12),  # 100 nHa
    )

    def __init__(self, conf, device, shared_bvqe=None):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.random_halt = int(conf['env']['rand_halt'])
        self.n_shots =   conf['env']['n_shots'] 
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        noise_values = conf['env']['noise_values'] 
        self.mol = conf['problem']['ham_type']
        self.map_theta = conf["agent"]["map_theta"]

        if noise_values !=0:
            indx = conf['env']['noise_values'].index(',')
            self.noise_values = [float(noise_values[1:indx]), float(noise_values[indx+1:-1])]
            print(self.noise_values)
        else:
            self.noise_values = []
        self.noise_models = noise_models[0:len(self.noise_values)]
        print(f"self.noise_models: {self.noise_models}")

        if len(self.noise_models) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False

        self.err_mitig = conf['env']['err_mitig']
        
        self.ham_mapping = conf['problem']['mapping']
        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']

        # Reward fn scaling knobs (used by incremental_with_ladder / logprog_ladder).
        # Default 1.0 keeps backward compatibility for existing fn_types.
        self.ladder_coef = float(conf['env']['ladder_coef']) if "ladder_coef" in conf['env'].keys() else 1.0
        self.dense_coef  = float(conf['env']['dense_coef'])  if "dense_coef"  in conf['env'].keys() else 1.0

        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.
        
        
        self.noise_flag = False
        # ── Optional PTM noise backend (default OFF; see code/noise_ptm/) ──────────
        # Left None here and attached from outside — Runner.__init__ / main_baseline —
        # when Config.noise_mode != "off". While it is None every branch below takes
        # the byte-identical legacy path, so the noiseless results are unchanged.
        # When it IS set, `noise_flag` is flipped too, otherwise step() at :292-293
        # would throw the noisy energy away again.
        self._noisy_eval = None
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0
        self.curriculum_dict = {}
        _mol_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "mol_data")
        _ham_path = os.path.join(_mol_data_dir, f"{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        __ham = np.load(_ham_path, allow_pickle=True)

        print(_ham_path)

        _, _, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(eigvals) + energy_shift
        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.min_energy = min(eigvals) + self.energy_shift

        self.max_eig = max(eigvals)+self.energy_shift
        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)
        self.device = device
        # Lazily create a reusable CPU QuantumState only if we actually hit the
        # Qulacs fallback path.  BatchedVQE handles the main GPU path, and
        # eager QuantumStateGpu construction was creating an unwanted CUDA:0
        # context at process startup.
        self.ket = None
        self.done_threshold = conf['env']['accept_err']

        self.op_history = []
        self.delta_mask = np.zeros(self.num_layers, dtype=np.float32)

        stdout.flush()
        self.state_size = self.num_layers * self.num_qubits*(self.num_qubits + 3 + 3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0]*self.num_qubits
        # self.illegal_actions = [[]]*self.num_qubits
        self.illegal_actions = [[] for _ in range(self.num_qubits)]
        self.energy = 0
        self.opt_alg_save = 0

        self.action_size = (self.num_qubits*(self.num_qubits+2))
        self.previous_action = [0, 0, 0, 0]

        if 'non_local_opt' in conf.keys():

            self.external_opt = conf['non_local_opt'].get('external_opt', 0)
            self.opt_scope = conf['non_local_opt'].get('scope', 'all')
            self.opt_k = int(conf['non_local_opt'].get('k', 1))

            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']
            if 'a' in conf['non_local_opt'].keys():
                self.options = {'a': conf['non_local_opt']["a"], 'alpha': conf['non_local_opt']["alpha"],
                            'c': conf['non_local_opt']["c"], 'gamma': conf['non_local_opt']["gamma"],
                            'beta_1': conf['non_local_opt']["beta_1"],
                            'beta_2': conf['non_local_opt']["beta_2"]}

            if 'lamda' in conf['non_local_opt'].keys():
                self.options['lamda'] = conf['non_local_opt']["lamda"]
            if 'maxfev' in conf['non_local_opt'].keys():
                self.maxfev = {}
                self.maxfev['maxfev'] = int(conf['non_local_opt']["maxfev"])
            if 'maxfev1' in conf['non_local_opt'].keys():
                self.maxfevs = {}
                self.maxfevs['maxfev1'] = int( conf['non_local_opt']["maxfev1"] )
                self.maxfevs['maxfev2'] = int( conf['non_local_opt']["maxfev2"] )
                self.maxfevs['maxfev3'] = int( conf['non_local_opt']["maxfev3"] )

            # writeback_coef: how much of the optimizer's result to write back into self.state.
            #   1.0 = full write-back (default route, state tracks x*)
            #   0.0 = no write-back  (ablation route, state keeps agent-initialised params)
            #   (0,1) = blended interpolation
            self.writeback_coef = float(conf['non_local_opt'].get('writeback_coef', 1.0))
            self.rotosolve_sweeps = int(conf['non_local_opt'].get('rotosolve_sweeps', 1))

            # BatchedVQE: GPU-batched energy evaluator used by rotosolve_optim.
            # Only initialised when optim_alg = Rotosolve to avoid loading the
            # full 2^n Hamiltonian matrix onto GPU for COBYLA runs.
            # shared_bvqe: pass an existing BatchedVQE from env[0] so all parallel
            # envs share one H matrix on GPU instead of duplicating it K times.
            if str(self.optim_alg).upper() == 'ROTOSOLVE':
                if shared_bvqe is not None:
                    self._batched_vqe = shared_bvqe
                    self._owns_bvqe   = False
                else:
                    self._batched_vqe = vc.BatchedVQE(
                        self.num_qubits, self.hamiltonian, self.energy_shift, self.device
                    )
                    self._owns_bvqe   = True

        else:
            self.global_iters = 0
            self.optim_method = None
            self.external_opt = 0
            self.writeback_coef = 0.0
            self.rotosolve_sweeps = 1

        # refine_delta_coef: scalar multiplier applied to the RL refinement delta before
        # adding it to existing parameters.  1.0 = full delta (default), 0.0 = no refinement.
        self.refine_delta_coef = float(conf['env'].get('refine_delta_coef', 1.0))
            
        self.start_energy = self.min_eig + self.done_threshold

    def step(self, action, action_params, train_flag: bool = True, refine_delta: Optional[torch.Tensor] = None):

        next_state = self.state.clone()
        self.step_counter += 1

        ctrl, targ, rot_qubit, rot_axis = action[0], (action[0] + action[1]) % self.num_qubits, action[2], action[3]
        self.action = action
        self.current_action = action
        self.illegal_action_new()

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
            self.delta_mask[self.step_counter] = 1.0
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ] )

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1

        
        if rot_qubit < self.num_qubits:
            if self.map_theta:
                theta = map_theta(action_params, self.device)
            else:
                theta = action_params
            next_state[gate_tensor][self.num_qubits + 3 + rot_axis - 1][rot_qubit] = theta
            self.moments[rot_qubit] += 1
            self.op_history.append({'type': 'rot', 'layer': int(gate_tensor), 'q': int(rot_qubit), 'axis': int(rot_axis)})
        elif ctrl < self.num_qubits:
            max_of_two = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two + 1
            self.moments[targ] = max_of_two + 1
            self.op_history.append({'type': 'cnot', 'layer': int(gate_tensor), 'ctrl': int(ctrl), 'targ': int(targ)})

        if refine_delta is not None:
            if isinstance(refine_delta, torch.Tensor):
                delta_vec = refine_delta.detach().cpu().numpy().astype(np.float32)
            else:
                delta_vec = np.asarray(refine_delta, dtype=np.float32)
            upto = len(self.op_history) - 1
            assert len(delta_vec) >= upto, f"delta_vec too short: {len(delta_vec)} < {upto}"
            for t in range(upto):
                rec = self.op_history[t]
                if rec['type'] != 'rot':
                    assert abs(delta_vec[t]) < 1e-12, f"Nonzero delta on CNOT at t={t}: {delta_vec[t]}"
                    continue
                L, q, a = rec['layer'], rec['q'], rec['axis']
                angle_plane = self.num_qubits + 3 + a - 1
                old_theta = float(next_state[L][angle_plane][q].item())
                new_theta = old_theta + self.refine_delta_coef * float(delta_vec[t])
                next_state[L][angle_plane][q] = map_theta(new_theta, self.device) if self.map_theta else new_theta

        self.state = next_state.clone()


        use_external = bool(self.external_opt)
        if use_external and self.optim_method in ["scipy_each_step"]:
            if self.opt_scope == 'all':
                which = None # 全量优化
            elif self.opt_scope == 'last':
                which = 'LAST'
            elif self.opt_scope == 'k_last':
                which = ('K_LAST', int(self.opt_k))
            thetas_for_eval = self._external_optimize(which)
            energy, energy_noiseless = self.get_energy(thetas_for_eval)
        else:
            energy, energy_noiseless = self.get_energy()

        # energy, energy_noiseless = self.get_energy()

        if self.noise_flag == False:
            energy = energy_noiseless
        self.energy = energy
        # Additive: keep the noiseless companion as an ENERGY, not only as
        # `error_noiseless` (which is measured against the FAKE min_eig and so cannot be
        # turned back into an energy without assuming a sign). The noise experiments
        # report the delivered circuit both ways, and this is where the noiseless number
        # comes from. Equals `self.energy` exactly whenever noise is off.
        self.energy_noiseless = energy_noiseless
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)
    
        self.error = float(abs(self.min_eig-energy))
        self.error_noiseless = float(abs(self.min_eig-energy_noiseless))
        rwd = self.reward_fn(energy)
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        done_reason = 1 if energy_done else (-1 if layers_done else 0)
        
        self.previous_action = copy.deepcopy(action)
        

        if self.random_halt and self.step_counter == self.halting_step:
            done = 1
        if done:
            # past_thr = self.curriculum.get_current_threshold()
            self.curriculum.update_threshold(energy_done=energy_done)
            # new_thr = self.curriculum.get_current_threshold()
            # print("\n")
            # print(f"[Env] 结束 episode:lowest_energy={self.curriculum.lowest_energy:.6f}, "
            #       f"done_threshold: {past_thr:.6f} → {new_thr:.6f}")
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return self.state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done, done_reason
        else:
            state_no_angles = self.state[:, :self.num_qubits+3]
            return state_no_angles.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done, done_reason

    def reset(self):   

        self.delta_mask = np.zeros(self.num_layers, dtype=np.float32)
        self.op_history = []

        state = torch.zeros((self.num_layers, self.num_qubits + 3 + 3, self.num_qubits))
        self.state = state
        
        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=self.num_layers, p=0.573, size=1), 25, self.num_layers)[0]  
            self.halting_step = statistics_generated
        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*4
        # self.illegal_actions = [[]]*self.num_qubits
        self.illegal_actions = [[] for _ in range(self.num_qubits)]
        # print("*********************************************")
        # print(f"[DEBUG] self.halting_step : {self.halting_step}")
        # print(f"[DEBUG] self.current_action : {self.current_action}")
        # print(f"[DEBUG] self.illegal_actions : {self.illegal_actions}")
        # print("*********************************************")

        self.make_circuit(state)
        self.step_counter = -1
        self.moments = [0]*self.num_qubits
        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        
        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)
        _mol_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "mol_data")
        __ham = np.load(os.path.join(_mol_data_dir, f"{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz"))
        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals)+self.energy_shift

        # Sync BatchedVQE Hamiltonian when geometry changes between episodes.
        # Only the owner calls set_problem; shared envs trust the owner's copy.
        if hasattr(self, '_batched_vqe') and getattr(self, '_owns_bvqe', True):
            self._batched_vqe.set_problem(self.hamiltonian, self.energy_shift)

        self.prev_energy = self.get_energy()[1]
        # Per-episode best error tracker for staircase reward (fn_type=staircase).
        # Reset to inf so the first ladder crossing fires correctly.
        self.best_err = float("inf")
        # Per-episode previous true_err tracker for logprog_ladder (dense per-step).
        # Initialized to the true error of the initial (pre-action) circuit so the
        # first action step also gets a dense log-progress reward, matching the
        # per-step semantics. (Reviewer fix 2026-07-01.)
        self.prev_true_err = (float(abs(self.min_energy - self.prev_energy))
                              if self.min_energy is not None else None)   # None when E0 severed

        return state.reshape(-1).to(self.device)
      
        
    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+3:]
        
        circuit = ParametricQuantumCircuit(self.num_qubits)
        
        for i in range(self.num_layers):
            
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(ctrl[r], targ[r]))
                    
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)
            
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(rot_qubit, thetas[i][0][rot_qubit]) 
                    elif r == 1:
                        circuit.add_parametric_RY_gate(rot_qubit, thetas[i][1][rot_qubit])
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(rot_qubit, thetas[i][2][rot_qubit])
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        
                        assert r >2
        
        return circuit

    def R_gate(self, qubit, axis, angle):
        if axis == 'X' or axis == 'x' or axis == 1:
            return RX(qubit, angle)
        elif axis == 'Y' or axis == 'y' or axis == 2:
            return RY(qubit, angle)
        elif axis == 'Z' or axis == 'z' or axis == 3:
            return RZ(qubit, angle)
        else:
            print("Wrong gate")
            return 1


    def _energy_via_bvqe(self):
        """Single-circuit noiseless energy via BatchedVQE (GPU).

        Extracts rotation angles from self.state via vectorised indexing
        (no Python loop) and calls eval_batch with B=1.

        Thread-safe: eval_batch only reads self._batched_vqe.H (immutable
        after init) and creates local tensors; self.state is per-env.
        """
        n    = self.num_qubits
        st   = self.state          # on CPU
        bvqe = self._batched_vqe
        rot_pos  = (st[:, n:n+3] == 1).nonzero(as_tuple=True)
        angle_t  = st[:, n+3:][rot_pos].float().unsqueeze(0).to(bvqe.device)  # (1, n_params)
        return float(bvqe.eval_batch(st, angle_t)[0])

    def _ensure_ket(self):
        """Create a reusable CPU QuantumState only when the Qulacs path is used."""
        if self.ket is None:
            self.ket = QuantumState(self.num_qubits)
        return self.ket

    def attach_noisy_evaluator(self, evaluator):
        """Turn the PTM noise backend ON for this env.

        Sets both the evaluator and ``noise_flag`` together — they are one switch.
        Without the flag, step() at :292-293 would still overwrite the noisy energy
        with the noiseless one, which is why the pre-existing `noise_values` cfg keys
        never did anything.
        """
        self._noisy_eval = evaluator
        self.noise_flag = evaluator is not None

    def get_energy(self, thetas=None):
        ev = self._noisy_eval
        if ev is not None:
            # ── PTM noise backend ────────────────────────────────────────────────
            # Noisy <H> comes from the PTM path (exact density-matrix expectation, no
            # shots, deterministic). The noiseless companion is still the qulacs
            # statevector — it is what the paper reports the delivered circuit against,
            # and at 4 qubits it is nearly free.
            circ = self.make_circuit(thetas)
            if thetas is not None:
                tmp_state = self.state.clone()
                tmp_state[:, self.num_qubits + 3:] = thetas
            else:
                tmp_state = self.state
            ket = self._ensure_ket()
            energy_noiseless = vc.get_exp_val(
                self.num_qubits, circ, self.hamiltonian, state=ket) + self.energy_shift
            return ev.energy_from_state(tmp_state), energy_noiseless

        # Fast GPU path: noiseless, no angle override, BatchedVQE available.
        if thetas is None and not self.phys_noise and hasattr(self, '_batched_vqe'):
            energy = self._energy_via_bvqe()
            return energy, energy

        # Qulacs fallback path (noisy circuits or explicit thetas override).
        circ = self.make_circuit(thetas)
        if thetas is not None:
            tmp_state = self.state.clone()
            tmp_state[:, self.num_qubits+3:] = thetas
            qulacs_inst = vc.Parametric_Circuit(
                n_qubits=self.num_qubits,
                noise_models=self.noise_models,
                noise_values=self.noise_values)
            noisy_circ = qulacs_inst.construct_ansatz(tmp_state)
        else:
            qulacs_inst = vc.Parametric_Circuit(
                n_qubits=self.num_qubits,
                noise_models=self.noise_models,
                noise_values=self.noise_values)
            noisy_circ = qulacs_inst.construct_ansatz(self.state)

        ket = self._ensure_ket()
        expval_noisy = vc.get_exp_val(
            self.num_qubits, noisy_circ, self.hamiltonian,
            self.phys_noise, self.err_mitig, state=ket)
        expval_noiseless = vc.get_exp_val(
            self.num_qubits, circ, self.hamiltonian, state=ket)
        shot_noise = vc.get_shot_noise(self.weights, self.n_shots)
        energy = expval_noisy + shot_noise + self.energy_shift
        energy_noiseless = expval_noiseless + self.energy_shift
        return energy, energy_noiseless

    def reference_energy(self):
        """complex128 Qulacs recompute of the CURRENT circuit at its current (optimized)
        angles. Passing explicit thetas forces the Qulacs statevector path, bypassing the
        GPU complex64 BatchedVQE fast path — used for frozen-eval energy on the rotosolve
        (8q/10q) path so the 1.6 mHa success label never flips on complex64 search error."""
        thetas = self.state[:, self.num_qubits + 3:]
        return self.get_energy(thetas)[0]

    def get_ket(self, thetas=None):
        state = self.state.clone()
        circ = self.make_circuit(thetas)
        ket = self._ensure_ket()
        ket.set_zero_state()
        circ.update_quantum_state(ket)
        v = ket.get_vector()
        v_real = torch.tensor(np.real(v),device=self.device,dtype=torch.float)
        v_imag = torch.tensor(np.imag(v),device=self.device,dtype=torch.float)
        return torch.cat((v_real,v_imag)).view(1,-1)

    def get_angles(self, update_idx):
        state = self.state.clone()
        thetas = state[-1]
        thetas[update_idx] = 0.0
        theta1, theta2, theta3 = thetas.clone(), thetas.clone(), thetas.clone()
        theta1[update_idx] -= np.pi/2
        theta3[update_idx] += np.pi/2

        e1 = self.get_energy(theta1)
        e2 = self.get_energy(theta2)
        e3 = self.get_energy(theta3)
        
        C = 0.5*(e1+e3)
        B = np.arctan2((e2-C), (e3-C))

        thetas[update_idx] = -B-np.pi/2
        return thetas

        
    def scipy_optim(self, which_angles=None):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)  
        angles = thetas[rot_pos]

        qulacs_inst = vc.Parametric_Circuit(
            n_qubits=self.num_qubits,
            noise_models=self.noise_models,
            noise_values = self.noise_values
            )
        qulacs_circuit = qulacs_inst.construct_ansatz(state)
        
        x0_full = np.asarray(angles.detach().cpu()) 
        param_count = len(x0_full)

        if which_angles is None:
            idx = np.arange(param_count)
        else:
            idx = np.asarray(which_angles, dtype=int)

        x0_sub = x0_full[idx].copy()

        # PTM noise backend: bind the circuit ONCE per step, so each of the (up to
        # global_iters=1000) objective evaluations below only ships the angle vector
        # to the device instead of re-parsing and re-uploading the slot arrays.
        ev = self._noisy_eval
        if ev is not None:
            n_rot = ev.rebind(state)
            if n_rot != param_count:
                raise RuntimeError(
                    f"noise backend parsed {n_rot} rotations but the optimizer has "
                    f"{param_count} parameters — state-tensor parsing disagrees"
                )

        def cost(x_sub):
            x_all = x0_full.copy()
            x_all[idx] = x_sub
            if self.map_theta:
                x_all = map_theta(torch.tensor(x_all, dtype=torch.float32, device=self.device), self.device).detach().cpu().numpy()
            if ev is not None:
                return ev.energy(x_all)
            return vc.get_energy_qulacs(
                x_all,
                observable = self.hamiltonian,
                weights = self.weights, 
                circuit = qulacs_circuit,
                n_qubits = self.num_qubits, 
                energy_shift = self.energy_shift,
                n_shots = int(self.n_shots),
                phys_noise = self.phys_noise,
                which_angles=np.arange(param_count)
                )

        result = scipy.optimize.minimize(
            cost,
            x0=x0_sub,
            method= self.optim_alg,
            options={'maxiter': self.global_iters}
        )
        x_best_sub = result.x
        x_best_full = x0_full.copy()
        x_best_full[idx] = x_best_sub

        thetas_opt_full = thetas.clone().to('cpu')
        thetas_opt_full[rot_pos] = torch.from_numpy(x_best_full).to(thetas_opt_full.dtype)

        return thetas_opt_full, int(result.nfev), x_best_sub


    # ── deferred-energy interface (used by CUDA-stream parallel training) ──────

    def step_deferred_energy(self, action, action_params,
                             train_flag: bool = True,
                             refine_delta=None):
        """Apply gate to circuit — energy/reward deferred for external batched eval.

        Identical to the gate-application half of step().  After this call
        self.state carries the new gate (with agent-initialised angles).
        The caller is responsible for running _rotosolve_envs_batched() and
        _eval_env_energies_batched(), then calling finalize_step_with_energy().
        """
        next_state = self.state.clone()
        self.step_counter += 1

        ctrl      = action[0]
        targ      = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis  = action[3]
        self.action         = action
        self.current_action = action
        self.illegal_action_new()

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
            self.delta_mask[self.step_counter] = 1.0
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits + rot_axis - 1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            theta = map_theta(action_params, self.device) if self.map_theta else action_params
            next_state[gate_tensor][self.num_qubits + 3 + rot_axis - 1][rot_qubit] = theta
            self.moments[rot_qubit] += 1
            self.op_history.append({'type': 'rot',  'layer': int(gate_tensor),
                                    'q': int(rot_qubit), 'axis': int(rot_axis)})
        elif ctrl < self.num_qubits:
            max_of_two = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two + 1
            self.moments[targ] = max_of_two + 1
            self.op_history.append({'type': 'cnot', 'layer': int(gate_tensor),
                                    'ctrl': int(ctrl), 'targ': int(targ)})

        if refine_delta is not None:
            if isinstance(refine_delta, torch.Tensor):
                delta_vec = refine_delta.detach().cpu().numpy().astype(np.float32)
            else:
                delta_vec = np.asarray(refine_delta, dtype=np.float32)
            upto = len(self.op_history) - 1
            assert len(delta_vec) >= upto, f"delta_vec too short: {len(delta_vec)} < {upto}"
            for t in range(upto):
                rec = self.op_history[t]
                if rec['type'] != 'rot':
                    assert abs(delta_vec[t]) < 1e-12, \
                        f"Nonzero delta on CNOT at t={t}: {delta_vec[t]}"
                    continue
                L, q, a   = rec['layer'], rec['q'], rec['axis']
                angle_plane = self.num_qubits + 3 + a - 1
                old_theta   = float(next_state[L][angle_plane][q].item())
                new_theta   = old_theta + self.refine_delta_coef * float(delta_vec[t])
                next_state[L][angle_plane][q] = (map_theta(new_theta, self.device)
                                                  if self.map_theta else new_theta)

        self.state = next_state.clone()
        self._deferred_train_flag = train_flag

    def finalize_step_with_energy(self, energy, energy_noiseless):
        """Compute reward/done from externally provided energy, update curriculum.

        Call after _rotosolve_envs_batched() and _eval_env_energies_batched()
        have updated self.state and provided energy values.
        Returns (next_state, reward, done, done_reason) — same as step().
        """
        train_flag = getattr(self, '_deferred_train_flag', True)

        if self.noise_flag == False:
            energy = energy_noiseless
        self.energy = energy
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)

        self.error           = float(abs(self.min_eig - energy))
        self.error_noiseless = float(abs(self.min_eig - energy_noiseless))
        rwd                  = self.reward_fn(energy)
        self.prev_energy     = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done        = int(energy_done or layers_done)
        done_reason = 1 if energy_done else (-1 if layers_done else 0)

        self.previous_action = copy.deepcopy(self.action)

        if self.random_halt and self.step_counter == self.halting_step:
            done = 1
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)

        if self.state_with_angles:
            return (self.state.view(-1).to(self.device),
                    torch.tensor(rwd, dtype=torch.float32, device=self.device),
                    done, done_reason)
        else:
            state_no_angles = self.state[:, :self.num_qubits + 3]
            return (state_no_angles.reshape(-1).to(self.device),
                    torch.tensor(rwd, dtype=torch.float32, device=self.device),
                    done, done_reason)

    # ─────────────────────────────────────────────────────────────────────────

    def rotosolve_optim(self, which_angles=None):
        """Rotosolve: GPU-batched analytical angle optimisation.

        For each rotation parameter θ_i (or a subset given by which_angles),
        evaluates E at {0, π/2, π} with all others fixed, fits the model
            E(θ_i) = A·cos(θ_i) + B·sin(θ_i) + C
        and sets θ_i to the analytical minimum atan2(−B, −A).
        Repeats for self.rotosolve_sweeps full sweeps.

        All 3·|subset| probe circuits are evaluated in a single GPU kernel per
        sweep via BatchedVQE.eval_batch — no sequential Qulacs calls needed.
        All probe-matrix construction and angle fitting run on GPU (zero numpy).

        Returns (thetas_opt_full, nfev, opt_angles) — same interface as scipy_optim.
        """
        bvqe  = self._batched_vqe
        bdev  = bvqe.device
        st    = self.state          # lives on CPU (created in reset())

        # Rotation positions and current angles — extracted on CPU (no GPU sync).
        rot_pos  = (st[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles_t = st[:, self.num_qubits + 3:][rot_pos].float().to(bdev)   # (n_params,) on GPU
        n_params = angles_t.shape[0]

        if n_params == 0:
            return st[:, self.num_qubits + 3:].clone(), 0, angles_t

        idx_t = (torch.arange(n_params, device=bdev)
                 if which_angles is None
                 else torch.tensor(which_angles, dtype=torch.long, device=bdev))
        n_opt = idx_t.shape[0]

        probe_vals_t = torch.tensor([0.0, torch.pi / 2.0, torch.pi],
                                    dtype=torch.float32, device=bdev)
        nfev = 0

        for _ in range(self.rotosolve_sweeps):
            n_probes = 3 * n_opt

            # Build probe matrix entirely on GPU — no numpy, no CPU loop.
            probe_matrix = angles_t.unsqueeze(0).expand(n_probes, -1).contiguous()
            col_idx = idx_t.repeat_interleave(3)
            probe_matrix[torch.arange(n_probes, device=bdev), col_idx] = \
                probe_vals_t.repeat(n_opt)

            energies = bvqe.eval_batch(st, probe_matrix)   # (n_probes,) on GPU
            nfev    += n_probes

            # Fit E(θ) = A cosθ + B sinθ + C, find minimum — all on GPU.
            e_mat = energies.reshape(n_opt, 3)
            A_t   = (e_mat[:, 0] - e_mat[:, 2]) * 0.5
            C_t   = (e_mat[:, 0] + e_mat[:, 2]) * 0.5
            B_t   = e_mat[:, 1] - C_t
            angles_t = angles_t.clone()
            angles_t[idx_t] = torch.arctan2(-B_t, -A_t)

        # Write back to thetas_opt (same shape as st[:, n+3:], on CPU).
        thetas_opt = st[:, self.num_qubits + 3:].clone()
        thetas_opt[rot_pos] = angles_t.cpu()
        return thetas_opt, nfev, angles_t


    def _external_optimize(self, which):
        """
        Translate scope to parameter indices, call scipy_optim, then optionally write
        the optimised parameters back into self.state according to writeback_coef:

          writeback_coef = 1.0  (default): full write-back — self.state carries x*_{t+1}.
                                Next step's agent and refine head both see optimised params.
          writeback_coef = 0.0  (ablation): no write-back — self.state always carries
                                agent-initialised params; refine acts on those.
          0 < writeback_coef < 1: blended interpolation between the two.

        Returns thetas_opt_full for energy evaluation regardless of write-back setting.
        """
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)
        param_count = rot_pos[0].numel()
        self.last_nfev = 0       # A3: inner-optimizer feval count (0 for no-op / CNOT steps)

        if which is None:
            which_angles = None  # all params
        elif which == 'LAST':
            rot_qubit = self.action[2]
            if not (rot_qubit < self.num_qubits):
                return thetas  # CNOT step — nothing to optimise
            which_angles = [param_count - 1]
        elif isinstance(which, tuple) and which[0] == 'K_LAST':
            k = int(which[1])
            k = max(1, min(k, param_count))
            which_angles = list(range(param_count - k, param_count))
        else:
            which_angles = None

        if str(getattr(self, 'optim_alg', 'COBYLA')).upper() == 'ROTOSOLVE':
            thetas_opt_full, self.last_nfev, _ = self.rotosolve_optim(which_angles=which_angles)
        else:
            thetas_opt_full, self.last_nfev, _ = self.scipy_optim(which_angles=which_angles)

        if self.writeback_coef > 0.0:
            # Both thetas_opt_full (from scipy) and self.state are CPU tensors — no device transfer needed.
            thetas_current = self.state[:, self.num_qubits+3:].clone()
            thetas_wb = (self.writeback_coef * thetas_opt_full
                         + (1.0 - self.writeback_coef) * thetas_current)
            if self.map_theta:
                thetas_wb = map_theta(thetas_wb, thetas_wb.device)
            self.state[:, self.num_qubits+3:] = thetas_wb

        return thetas_opt_full

    def reward_fn(self, energy):
        if self.min_energy is None and self.fn_type in ("staircase", "incremental_with_ladder", "logprog_ladder"):
            # These ladder rewards consume the TRUE E0; incompatible with severed diagnostics
            # (the phase2 runner guards this at startup — this is the fail-loud backstop).
            raise RuntimeError(f"fn_type={self.fn_type} requires the true ground-state energy (E0)")
        if self.fn_type == "staircase_curriculum":
            # Original curriculum-coupled staircase (legacy, do not use for new exps)
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "staircase":
            # Q1 reward redesign: event-driven sparse reward keyed on absolute
            # best_err thresholds (no curriculum coupling, no -5 max_depth penalty).
            # Uses TRUE error (against real FCI), not self.error (which is against
            # fake_min_energy and would be 2-3 Ha offset — never crosses any rung).
            true_err = float(abs(self.min_energy - energy))
            prev_best = self.best_err
            new_best = min(prev_best, true_err)
            self.best_err = new_best
            if new_best >= prev_best:
                return 0.0
            rwd = 0.0
            for thresh, val in self.STAIRCASE_LADDER:
                if prev_best > thresh >= new_best:
                    rwd += val
            return rwd
        elif self.fn_type == "incremental_with_ladder":
            # User's design: v2 incremental base UNCHANGED + ladder bonus on best_err crossings.
            # Base uses self.error (vs fake_min_energy, curriculum-driven) — preserves v2 semantics.
            # Ladder uses true_err (vs real FCI) — milestone events on absolute precision.
            # ladder values from STAIRCASE_LADDER_V2 (×1.5/stop, chem.acc anchor=5.0).
            max_depth = self.step_counter == (self.num_layers - 1)
            if self.error < self.done_threshold:
                base_r = 5.0
            elif max_depth:
                base_r = -5.0
            else:
                base_r = float(np.clip(
                    (self.prev_energy - energy) / abs(self.start_energy - self.min_eig),
                    -1, 1,
                ))

            true_err = float(abs(self.min_energy - energy))
            prev_best = self.best_err
            new_best = min(prev_best, true_err)
            self.best_err = new_best
            bonus = 0.0
            if new_best < prev_best:
                for thresh, val in self.STAIRCASE_LADDER_V2:
                    if prev_best > thresh >= new_best:
                        bonus += val
            return base_r + self.ladder_coef * bonus

        elif self.fn_type == "logprog_ladder":
            # Design A: dense per-step log-progress (positive only) + ladder bonus.
            # No v2 base. All signals come from true_err vs real FCI.
            true_err = float(abs(self.min_energy - energy))
            prev_err = self.prev_true_err
            self.prev_true_err = true_err
            dense_r = 0.0
            # Guards: first step (prev=None) / regression (prev<=now) / true_err=0 (perfect, would log-explode)
            if prev_err is not None and true_err > 0 and prev_err > true_err:
                dense_r = self.dense_coef * float(np.log10(prev_err / true_err))

            prev_best = self.best_err
            new_best = min(prev_best, true_err)
            self.best_err = new_best
            event_r = 0.0
            if new_best < prev_best:
                for thresh, val in self.STAIRCASE_LADDER_V2:
                    if prev_best > thresh >= new_best:
                        event_r += val
            return dense_r + self.ladder_coef * event_r

        elif self.fn_type == "incremental_with_fixed_ends":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return rwd
        elif self.fn_type == "incremental_with_fixed_start":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                # 用固定的 start_energy 作为分母参考
                rwd = np.clip((self.prev_energy - energy) / abs(self.start_energy - self.min_eig), -1, 1)
            return rwd

        
    def illegal_action_new(self):

        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]


        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        seen = set()
        for i in range(len(illegal_action)):
            if len(illegal_action[i]) == 0:
                continue
            key = to_tuple4(illegal_action[i])
            if key in seen:
                illegal_action[i] = []          
            else:
                seen.add(key)

       
        write = 0
        for read in range(len(illegal_action)):
            if len(illegal_action[read]) != 0:
                if write != read:
                    illegal_action[write] = list(illegal_action[read])
                    illegal_action[read] = []
                write += 1
        for k in range(write, len(illegal_action)):
            illegal_action[k] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        
        return illegal_action_decode


if __name__ == "__main__":
    pass
