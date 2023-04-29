import numpy as np
from tqdm import tqdm
from env import final_env
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()
    env = final_env()
    iter = 5
    iter_state_eval = 120

    policy = env.actions_prob * np.ones(
        (env.state_size_delta_soc, 
         env.state_size_delta_time, 
         env.state_size_time, env.action_size)
        )
    
    # continue q value evaluation
    # policy = np.load('policy/iter5x120-highSOC_penalty_large.npy')

    for i in tqdm(np.arange(0, 0+iter)):

        # evaluation
        values, sync_iteration = env.compute_state_value(
            max_iter=iter_state_eval, 
            discount=1, 
            policy=policy
            )
        
        # greedy improvement
        policy = env.greedy_Policy(values)
        np.save(f'policy/iter{i}x{iter_state_eval}-highSOC_penalty_large', policy)

    np.save(f'policy/iter{i}x{iter_state_eval}-highSOC_penalty_large', policy)
