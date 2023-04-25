import numpy as np
from simple_charge_back_to_grid_env import Simple_charge_back_to_grid_env
policy = np.load('simple_greedy_policy_iter10x40.npy')


# state = [0.8, 47,0]
# state = [0.28, 47,0]
state = [0.8, 47, 0]
# state = [0.28, 47,0]
done = False
charge_env = Simple_charge_back_to_grid_env()
action_history = [None]*charge_env.state_size_time
while not done:
    state_0_index = charge_env.get_index(state[0])
    action_prob = policy[state_0_index,state[1],state[2],:]
    action = np.argmax(action_prob)
    # if action !=0:
    action_history[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    state = new_state
print(action_history)
