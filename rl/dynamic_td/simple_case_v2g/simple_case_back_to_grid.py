import numpy as np
from tqdm import tqdm
from simple_charge_back_to_grid_env import Simple_charge_back_to_grid_env
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()
    charge_env = Simple_charge_back_to_grid_env()
    iter = 20
    iter_state_eval = 40

    policy = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))
    # policy = np.load('simple_greedy_policy_iter20x40.npy')

    for i in tqdm(range(iter)):

        # evaluation
        values, sync_iteration = charge_env.compute_state_value_parallel(
            max_iter=iter_state_eval, 
            discount=1, 
            policy=policy
            )
        
        # greedy improvement
        policy = charge_env.greedy_policy_parallel(values)

    np.save(f'simple_greedy_policy_iter{iter}x{iter_state_eval}', policy)
