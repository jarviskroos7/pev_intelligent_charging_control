import numpy as np
from charge_env import Charge_env
policy = np.load('medium_greedy_policy_iter7.npy')


state = [0.8, 47,0]
state = [0.28, 47,0]
done = False
charge_env = Charge_env()
action_history = [0]*charge_env.state_size_time
while not done:
    state_0_index = charge_env.get_index(state[0])
    action_prob = policy[state_0_index,state[1],state[2],:]
    action = np.argmax(action_prob)
    if action !=0:
        action_history[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    state = new_state
print(action_history)
