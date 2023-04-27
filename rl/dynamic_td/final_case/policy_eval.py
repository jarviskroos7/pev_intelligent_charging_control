import numpy as np
from tqdm import tqdm
from env import final_env
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()
    env = final_env()
    iter = 5
    iter_state_eval = 120

    policy = env.actions_prob * np.ones((env.state_size_delta_soc, env.state_size_delta_time, env.state_size_time, env.action_size))
    # policy = np.load('simple_greedy_policy_iter20x40.npy')

    for i in tqdm(range(iter)):

        # evaluation
        values, sync_iteration = env.compute_state_value(
            max_iter=iter_state_eval, 
            discount=1, 
            policy=policy
            )
        
        # greedy improvement
        policy = env.greedy_Policy(values)
        if i % 2 == 0:
            np.save(f'final_greedy_policy_charging_limit_iter{i}x{iter_state_eval}', policy)

    np.save(f'final_greedy_policy_12min_charging_limit_iter{iter}x{iter_state_eval}', policy)
