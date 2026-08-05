import configparser
import numpy as np 
import json
from itertools import product
import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    设置随机种子，确保实验可复现。
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   # seeds the current device only (set_device must be called first)

def get_config(config_name, experiment_name, path='configuration_files', verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path, config_name, experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'refine_dropout', 'alpha',
                      'beta', 'beta_incr', 'a', 'gamma', 'c',
                      'maxfev', 'lamda', 'beta_1', 'beta_2',
                      'maxfev1', 'maxfev2', 'maxfev3','entropy_coef', 'grad_clip',
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en","n_shots", "err_mitig", "rand_halt",
                      "writeback_coef", "refine_delta_coef",
                      "ppo_clip", "gae_lambda", "vf_coef"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type", "angle_activation"]
            lists = ['noise_values','episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes","refine_neurons"]  
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict

def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        
def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def to_tuple4(x):
    """统一成4元组(int,int,int,int)，兼容 list/tuple/np.ndarray。"""
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    # x 可能是 [], 这里仅在非空时调用
    return tuple(int(v) for v in x)

def modify_state(state: torch.Tensor, env, conf: dict, device) -> torch.Tensor:
    """Append prev_energy / done_threshold to agent observation if configured.

    Args:
        state: flat observation tensor
        env: CircuitEnv instance (reads prev_energy, done_threshold)
        conf: config dict (reads agent.en_state, agent.threshold_in_state)
        device: torch device
    """
    if conf["agent"].get("en_state", 0):
        state = torch.cat(
            (state, torch.tensor(env.prev_energy, dtype=torch.float, device=device).view(1))
        )
    if conf["agent"].get("threshold_in_state", 0):
        state = torch.cat(
            (state, torch.tensor(env.done_threshold, dtype=torch.float, device=device).view(1))
        )
    return state


def agent_gradient_step(agent, batch_trajs: list) -> float:
    """Dispatch gradient update to the correct agent method.

    Returns policy loss as float.
    """
    if hasattr(agent, "act_with_refine"):
        return agent.gradient_update_batch_refine(
            batch_trajs, agent.entropy_coef, agent.grad_clip)
    else:
        return agent.gradient_update_batch(
            batch_trajs, agent.entropy_coef, agent.grad_clip)


def map_theta(theta: torch.Tensor, device) -> torch.Tensor:
    """Wrap angle to [0, 2π). Uses modulo — the most natural periodic mapping."""
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float32, device=device)
    else:
        theta = theta.to(device).to(torch.float32)

    two_pi = torch.tensor(2.0 * np.pi, dtype=torch.float32, device=device)
    theta = theta % two_pi
    return theta
