import numpy as np
from tqdm import tqdm
from env import final_env
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()
    env = final_env()
    iter = 20
    iter_state_eval = 40

    policy = env.actions_prob * np.ones((env.state_size_delta_soc, env.state_size_delta_time, env.state_size_time, env.action_size))
    # policy = np.load('simple_greedy_policy_iter20x40.npy')

    for i in tqdm(range(iter)):

        # evaluation
        values, sync_iteration = env.compute_state_value_parallel(
            max_iter=iter_state_eval, 
            discount=1, 
            policy=policy
            )
        
        # greedy improvement
        policy = env.greedy_policy_parallel(values)

    np.save(f'final_greedy_policy_terminal_cond_iter{iter}x{iter_state_eval}', policy)
