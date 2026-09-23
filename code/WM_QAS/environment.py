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

    STAIRCASE_LADDER = (
        (5e-2,    2.0),
        (2e-2,    2.5),
        (1e-2,    3.0),
        (5e-3,    3.75),
        (1.6e-3,  5.0),
        (1e-3,    6.0),
        (5e-4,    7.5),
        (2e-4,    9.0),
        (1e-4,   11.0),
        (5e-5,   13.5),
        (2e-5,   16.5),
        (1e-5,   20.0),
        (5e-6,   24.0),
        (2e-6,   29.0),
        (1e-6,   35.0),
        (5e-7,   41.0),
        (2e-7,   49.0),
        (1e-7,   58.0),
    )

    STAIRCASE_LADDER_V2 = (
        (5e-2,    0.66),
        (2e-2,    0.99),
        (1e-2,    1.48),
        (5e-3,    2.22),
        (2e-3,    3.33),
        (1.6e-3,  5.0),
        (1e-3,    7.50),
        (5e-4,   11.25),
        (2e-4,   16.88),
        (1e-4,   25.31),
        (5e-5,   37.97),
        (2e-5,   56.95),
        (1e-5,   85.43),
        (5e-6,  128.14),
        (2e-6,  192.22),
        (1e-6,  288.33),
        (5e-7,  432.50),
        (2e-7,  648.74),
        (1e-7,  973.12),
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

        self.ladder_coef = float(conf['env']['ladder_coef']) if "ladder_coef" in conf['env'].keys() else 1.0
        self.dense_coef  = float(conf['env']['dense_coef'])  if "dense_coef"  in conf['env'].keys() else 1.0

        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.


        self.noise_flag = False
        self._noisy_eval = None
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0
        self.curriculum_dict = {}
        _mol_data_dir = os.path.abspath(os.path.expanduser(os.environ.get("DREAMQAS_MOL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "mol_data"))))
        _ham_path = os.path.join(_mol_data_dir, f"{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        if not os.path.isfile(_ham_path):
            raise FileNotFoundError(f"Hamiltonian not found: {_ham_path}. See data/README.md; set DREAMQAS_MOL_DATA for external data.")
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
        self.ket = None
        self.done_threshold = conf['env']['accept_err']

        self.op_history = []
        self.delta_mask = np.zeros(self.num_layers, dtype=np.float32)

        stdout.flush()
        self.state_size = self.num_layers * self.num_qubits*(self.num_qubits + 3 + 3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0]*self.num_qubits
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

            self.writeback_coef = float(conf['non_local_opt'].get('writeback_coef', 1.0))
            self.rotosolve_sweeps = int(conf['non_local_opt'].get('rotosolve_sweeps', 1))

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
                which = None
            elif self.opt_scope == 'last':
                which = 'LAST'
            elif self.opt_scope == 'k_last':
                which = ('K_LAST', int(self.opt_k))
            thetas_for_eval = self._external_optimize(which)
            energy, energy_noiseless = self.get_energy(thetas_for_eval)
        else:
            energy, energy_noiseless = self.get_energy()


        if self.noise_flag == False:
            energy = energy_noiseless
        self.energy = energy
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
            self.curriculum.update_threshold(energy_done=energy_done)
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
        self.illegal_actions = [[] for _ in range(self.num_qubits)]

        self.make_circuit(state)
        self.step_counter = -1
        self.moments = [0]*self.num_qubits
        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())

        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)
        _mol_data_dir = os.path.abspath(os.path.expanduser(os.environ.get("DREAMQAS_MOL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "mol_data"))))
        __ham = np.load(os.path.join(_mol_data_dir, f"{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz"))
        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals)+self.energy_shift

        if hasattr(self, '_batched_vqe') and getattr(self, '_owns_bvqe', True):
            self._batched_vqe.set_problem(self.hamiltonian, self.energy_shift)

        self.prev_energy = self.get_energy()[1]
        self.best_err = float("inf")
        self.prev_true_err = (float(abs(self.min_energy - self.prev_energy))
                              if self.min_energy is not None else None)

        return state.reshape(-1).to(self.device)


    def make_circuit(self, thetas=None):
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
        n    = self.num_qubits
        st   = self.state
        bvqe = self._batched_vqe
        rot_pos  = (st[:, n:n+3] == 1).nonzero(as_tuple=True)
        angle_t  = st[:, n+3:][rot_pos].float().unsqueeze(0).to(bvqe.device)
        return float(bvqe.eval_batch(st, angle_t)[0])

    def _ensure_ket(self):
        if self.ket is None:
            self.ket = QuantumState(self.num_qubits)
        return self.ket

    def attach_noisy_evaluator(self, evaluator):
        self._noisy_eval = evaluator
        self.noise_flag = evaluator is not None

    def get_energy(self, thetas=None):
        ev = self._noisy_eval
        if ev is not None:
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

        if thetas is None and not self.phys_noise and hasattr(self, '_batched_vqe'):
            energy = self._energy_via_bvqe()
            return energy, energy

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


    def step_deferred_energy(self, action, action_params,
                             train_flag: bool = True,
                             refine_delta=None):
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


    def rotosolve_optim(self, which_angles=None):
        bvqe  = self._batched_vqe
        bdev  = bvqe.device
        st    = self.state

        rot_pos  = (st[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles_t = st[:, self.num_qubits + 3:][rot_pos].float().to(bdev)
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

            probe_matrix = angles_t.unsqueeze(0).expand(n_probes, -1).contiguous()
            col_idx = idx_t.repeat_interleave(3)
            probe_matrix[torch.arange(n_probes, device=bdev), col_idx] = \
                probe_vals_t.repeat(n_opt)

            energies = bvqe.eval_batch(st, probe_matrix)
            nfev    += n_probes

            e_mat = energies.reshape(n_opt, 3)
            A_t   = (e_mat[:, 0] - e_mat[:, 2]) * 0.5
            C_t   = (e_mat[:, 0] + e_mat[:, 2]) * 0.5
            B_t   = e_mat[:, 1] - C_t
            angles_t = angles_t.clone()
            angles_t[idx_t] = torch.arctan2(-B_t, -A_t)

        thetas_opt = st[:, self.num_qubits + 3:].clone()
        thetas_opt[rot_pos] = angles_t.cpu()
        return thetas_opt, nfev, angles_t


    def _external_optimize(self, which):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)
        param_count = rot_pos[0].numel()
        self.last_nfev = 0

        if which is None:
            which_angles = None
        elif which == 'LAST':
            rot_qubit = self.action[2]
            if not (rot_qubit < self.num_qubits):
                return thetas
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
            thetas_current = self.state[:, self.num_qubits+3:].clone()
            thetas_wb = (self.writeback_coef * thetas_opt_full
                         + (1.0 - self.writeback_coef) * thetas_current)
            if self.map_theta:
                thetas_wb = map_theta(thetas_wb, thetas_wb.device)
            self.state[:, self.num_qubits+3:] = thetas_wb

        return thetas_opt_full

    def reward_fn(self, energy):
        if self.min_energy is None and self.fn_type in ("staircase", "incremental_with_ladder", "logprog_ladder"):
            raise RuntimeError(f"fn_type={self.fn_type} requires the true ground-state energy (E0)")
        if self.fn_type == "staircase_curriculum":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "staircase":
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
            true_err = float(abs(self.min_energy - energy))
            prev_err = self.prev_true_err
            self.prev_true_err = true_err
            dense_r = 0.0
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
