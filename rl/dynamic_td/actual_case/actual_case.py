import numpy as np
import time
from tqdm import tqdm
from actual_case_env import Charge_env
from utils import *
import os
import multiprocessing

if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    charge_env = Charge_env()
    # policy0 = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))
    iter = 30
    iter_state_eval = 50

    policy = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))

    # policy = np.load('medium_greedy_policy.npy')
    for i in tqdm(range(iter)):
        values, sync_iteration = charge_env.compute_state_value_parallel(max_iter=iter_state_eval, discount = 1, policy = policy)
        policy = charge_env.greedy_policy_parallel(values)
        # np.save(f'medium_greedy_policy_iter{i}', policy)

    np.save(f'greedy_policy_iter{iter}x{iter_state_eval}', policy)