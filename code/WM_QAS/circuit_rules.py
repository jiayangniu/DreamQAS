"""Pure, env-free circuit-structure + illegal-action rules.

Used by imagination (imagination/imagine.py) so that latent-space rollouts sample
ONLY env-legal actions, exactly as real episodes do — closing the gap where
imagination previously reinforced illegal actions on the shared actor.

SINGLE-SOURCE-OF-TRUTH NOTE
---------------------------
`compute_illegal` is a verbatim port of ``CircuitEnv.illegal_action_new``
(environment.py) and `apply_structure` is a structure-only port of the
gate-placement block in ``CircuitEnv.step``.  Both are guarded by
``tests/test_circuit_rules.py``, which drives the REAL ``CircuitEnv`` on random
rollouts and asserts bit-exact agreement of (a) the decoded illegal-action set
and (b) the structure tensor ``state[:, :N+3]``.  If you edit the env logic,
re-run that test and update this file.
"""

from __future__ import annotations

import numpy as np

from utils import dictionary_of_actions, to_tuple4

_DICT_CACHE = {}
_REV_CACHE = {}


def _dict_actions(num_qubits):
    """Cached dictionary_of_actions (rebuilt per call is wasteful in the hot loop)."""
    d = _DICT_CACHE.get(num_qubits)
    if d is None:
        d = dictionary_of_actions(num_qubits)
        _DICT_CACHE[num_qubits] = d
    return d


def _rev_actions(num_qubits):
    """Cached reverse map {action-tuple: index} for O(1) decode of illegal slots."""
    r = _REV_CACHE.get(num_qubits)
    if r is None:
        r = {tuple(v): k for k, v in _dict_actions(num_qubits).items()}
        _REV_CACHE[num_qubits] = r
    return r


def compute_illegal(illegal_action, current_action, num_qubits):
    """Verbatim port of ``CircuitEnv.illegal_action_new``.

    Stateful recurrence: mutates ``illegal_action`` IN PLACE (a list of up to
    ``num_qubits`` slots, each a 4-element action list or ``[]``) using
    ``current_action`` (a 4-element ``[ctrl, offset, rot_qubit, rot_axis]`` list,
    with ``num_qubits`` in the unused fields, exactly as in ``dictionary_of_actions``).

    Returns ``(illegal_action, decoded_indices)`` where ``decoded_indices`` is the
    list of illegal action indices (same value the env method returns).
    """
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

    # Decode occupied slots to action indices via a reverse lookup — O(active
    # slots) instead of the env's O(A x N) double loop. Order differs (slot order
    # vs dictionary order) but consumers use this as a SET (mask building), and the
    # equivalence test compares sets, so this stays bit-exact for masking.
    rev = _rev_actions(num_qubits)
    illegal_action_decode = []
    for s in illegal_action:
        if s:
            k = rev.get(tuple(s))
            if k is not None:
                illegal_action_decode.append(k)

    return illegal_action, illegal_action_decode


def apply_structure(struct, moments, action, num_qubits):
    """Structure-only port of the gate-placement block in ``CircuitEnv.step``.

    Mutates ``struct`` (a ``[num_layers, num_qubits+3, num_qubits]`` array — the
    structure rows the WM sees, i.e. ``state[:, :N+3]``) and ``moments`` (a list of
    per-qubit occupied-depth counters) IN PLACE, and returns ``(struct, moments)``.

    Matches env row/col conventions exactly:
      - CNOT  → struct[gate_layer, targ, ctrl] = 1              (rows 0..N-1)
      - Rot   → struct[gate_layer, N + axis - 1, rot_qubit] = 1 (rows N..N+2)
    where gate_layer = moments[rot_qubit] for a rotation, or
    max(moments[ctrl], moments[targ]) for a CNOT.
    """
    ctrl = action[0]
    targ = (action[0] + action[1]) % num_qubits
    rot_qubit = action[2]
    rot_axis = action[3]

    # Layer (moment) the gate lands on — mirrors env.step lines 230-234.
    if rot_qubit < num_qubits:
        gate_layer = moments[rot_qubit]
    elif ctrl < num_qubits:
        gate_layer = max(moments[ctrl], moments[targ])
    else:
        return struct, moments  # no-op action (should not occur)

    # Structure write — mirrors env.step lines 236-239.
    if ctrl < num_qubits:
        struct[gate_layer, targ, ctrl] = 1
    elif rot_qubit < num_qubits:
        struct[gate_layer, num_qubits + rot_axis - 1, rot_qubit] = 1

    # Moment update — mirrors env.step lines 242-253.
    if rot_qubit < num_qubits:
        moments[rot_qubit] += 1
    elif ctrl < num_qubits:
        m = max(moments[ctrl], moments[targ])
        moments[ctrl] = m + 1
        moments[targ] = m + 1

    return struct, moments


def fresh_illegal(num_qubits):
    """Initial illegal-action slot list after reset (matches env.reset)."""
    return [[] for _ in range(num_qubits)]


def empty_structure(num_layers, num_qubits):
    """Empty-circuit structure tensor (obs rows only)."""
    return np.zeros((num_layers, num_qubits + 3, num_qubits), dtype=np.float32)


class Shadow:
    """Per-trajectory illegal-action tracker mirroring CircuitEnv bookkeeping.

    The env's action mask is a pure function of ``illegal_actions`` +
    ``current_action`` (NOT of the circuit structure tensor), so this deliberately
    tracks ONLY the illegal-action recurrence — no structure tensor — keeping the
    imagination hot loop light. The structure port ``apply_structure`` remains
    available and is still covered by tests/test_circuit_rules.py for any consumer
    that needs it.

    Call pattern mirrors the env exactly (validated bit-exact in that test):
        mask_indices()  ↔  runner's `ill = env.illegal_action_new()` (selection)
        commit(action)  ↔  env.step()'s internal illegal_action_new update
    """

    __slots__ = ("num_qubits", "illegal", "current_action")

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.illegal = fresh_illegal(num_qubits)
        self.current_action = [num_qubits] * 4      # matches env.reset current_action

    def copy(self) -> "Shadow":
        s = Shadow.__new__(Shadow)
        s.num_qubits = self.num_qubits
        s.illegal = [list(x) for x in self.illegal]
        s.current_action = list(self.current_action)
        return s

    def mask_indices(self):
        """Illegal action indices for the NEXT action (mutates illegal, like env's ill= call)."""
        _, decoded = compute_illegal(self.illegal, self.current_action, self.num_qubits)
        return decoded

    def commit(self, action):
        """Apply a chosen action: mirror env.step's internal illegal-action update."""
        self.current_action = list(action)
        compute_illegal(self.illegal, self.current_action, self.num_qubits)
