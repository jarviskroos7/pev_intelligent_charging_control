import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



from normal_case_env import Normal_env
charge_env = Normal_env()

policy0 = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))
num_iteration = 20
for i in tqdm(range(num_iteration)):
    values, sync_iteration = charge_env.compute_state_value(max_iter=120, discount = 1,policy = policy0)
    policy = charge_env.greedy_Policy(values)

np.save(f'normal_greedy_policy_iter{num_iteration}', policy)
