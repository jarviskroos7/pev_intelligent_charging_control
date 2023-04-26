import numpy as np
import time
from tqdm import tqdm
from env import actual_env
from utils import *
import os
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()
    env = actual_env()
    # policy0 = env.actions_prob * np.ones((env.state_size_delta_soc, env.state_size_delta_time, env.state_size_time, env.action_size))
    iter = 30
    iter_state_eval = 50

    policy = env.actions_prob * np.ones((env.state_size_delta_soc, env.state_size_delta_time, env.state_size_time, env.action_size))

    # policy = np.load('medium_greedy_policy.npy')
    for i in tqdm(range(iter)):
        values, sync_iteration = env.compute_state_value_parallel(max_iter=iter_state_eval, discount = 1, policy = policy)
        policy = env.greedy_policy_parallel(values)
        # np.save(f'medium_greedy_policy_iter{i}', policy)

    np.save(f'greedy_policy_iter{iter}x{iter_state_eval}', policy)