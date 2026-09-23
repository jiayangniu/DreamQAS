from __future__ import annotations

import numpy as np

from utils import dictionary_of_actions, to_tuple4

_DICT_CACHE = {}
_REV_CACHE = {}


def _dict_actions(num_qubits):
    d = _DICT_CACHE.get(num_qubits)
    if d is None:
        d = dictionary_of_actions(num_qubits)
        _DICT_CACHE[num_qubits] = d
    return d


def _rev_actions(num_qubits):
    r = _REV_CACHE.get(num_qubits)
    if r is None:
        r = {tuple(v): k for k, v in _dict_actions(num_qubits).items()}
        _REV_CACHE[num_qubits] = r
    return r


def compute_illegal(illegal_action, current_action, num_qubits):
    action = current_action
    ctrl, targ = action[0], (action[0] + action[1]) % num_qubits
    rot_qubit, rot_axis = action[2], action[3]

    if ctrl < num_qubits:
        are_you_empty = sum([sum(l) for l in illegal_action])
        if are_you_empty != 0:
            for ill_ac_no, ill_ac in enumerate(illegal_action):
                if len(ill_ac) != 0:
                    ill_ac_targ = (ill_ac[0] + ill_ac[1]) % num_qubits

                    if ill_ac[2] == num_qubits:

                        if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        elif targ == ill_ac[0] or targ == ill_ac_targ:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        else:
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break
                    else:
                        if ctrl == ill_ac[2]:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        elif targ == ill_ac[2]:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break
                        else:
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break
        else:
            illegal_action[0] = action

    if rot_qubit < num_qubits:
        are_you_empty = sum([sum(l) for l in illegal_action])

        if are_you_empty != 0:
            for ill_ac_no, ill_ac in enumerate(illegal_action):

                if len(ill_ac) != 0:
                    ill_ac_targ = (ill_ac[0] + ill_ac[1]) % num_qubits

                    if ill_ac[0] == num_qubits:

                        if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        elif rot_qubit != ill_ac[2]:
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break
                    else:
                        if rot_qubit == ill_ac[0]:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        elif rot_qubit == ill_ac_targ:
                            illegal_action[ill_ac_no] = []
                            for i in range(1, num_qubits):
                                if len(illegal_action[i]) == 0:
                                    illegal_action[i] = action
                                    break

                        else:
                            for i in range(1, num_qubits):
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

    rev = _rev_actions(num_qubits)
    illegal_action_decode = []
    for s in illegal_action:
        if s:
            k = rev.get(tuple(s))
            if k is not None:
                illegal_action_decode.append(k)

    return illegal_action, illegal_action_decode


def apply_structure(struct, moments, action, num_qubits):
    ctrl = action[0]
    targ = (action[0] + action[1]) % num_qubits
    rot_qubit = action[2]
    rot_axis = action[3]

    if rot_qubit < num_qubits:
        gate_layer = moments[rot_qubit]
    elif ctrl < num_qubits:
        gate_layer = max(moments[ctrl], moments[targ])
    else:
        return struct, moments

    if ctrl < num_qubits:
        struct[gate_layer, targ, ctrl] = 1
    elif rot_qubit < num_qubits:
        struct[gate_layer, num_qubits + rot_axis - 1, rot_qubit] = 1

    if rot_qubit < num_qubits:
        moments[rot_qubit] += 1
    elif ctrl < num_qubits:
        m = max(moments[ctrl], moments[targ])
        moments[ctrl] = m + 1
        moments[targ] = m + 1

    return struct, moments


def fresh_illegal(num_qubits):
    return [[] for _ in range(num_qubits)]


def empty_structure(num_layers, num_qubits):
    return np.zeros((num_layers, num_qubits + 3, num_qubits), dtype=np.float32)


class Shadow:

    __slots__ = ("num_qubits", "illegal", "current_action")

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.illegal = fresh_illegal(num_qubits)
        self.current_action = [num_qubits] * 4

    def copy(self) -> "Shadow":
        s = Shadow.__new__(Shadow)
        s.num_qubits = self.num_qubits
        s.illegal = [list(x) for x in self.illegal]
        s.current_action = list(self.current_action)
        return s

    def mask_indices(self):
        _, decoded = compute_illegal(self.illegal, self.current_action, self.num_qubits)
        return decoded

    def commit(self, action):
        self.current_action = list(action)
        compute_illegal(self.illegal, self.current_action, self.num_qubits)
