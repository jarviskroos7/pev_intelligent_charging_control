import numpy as np
from env import final_env
policy = np.load('final_greedy_policy_12min_charging_limit_iter5x120.npy')


# state = [0.8, 47,0]
# state = [0.28, 47,0]
state = [0.8, 119, 0]
# state = [0.28, 47,0]
done = False
charge_env = final_env()

action_history = [None] * charge_env.state_size_time
reward_history = [None] * charge_env.state_size_time

while not done:
    state_0_index = charge_env.get_index(state[0])
    action_prob = policy[state_0_index, state[1], state[2], :]
    action = np.argmax(action_prob)
    # if action !=0:
    action_history[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    reward_history[state[2]] = reward
    state = new_state

for idx, action in enumerate(action_history):
    if action == 0:
        # discharging
        action_history[idx] = -1
    elif action == 1:
        # do nothing
        action_history[idx] = 0
    elif action == 2:
        # charging
        action_history[idx] = 1
    else:
        pass

print(action_history)
action_history = [0 if a is None else a for a in action_history]

print("Total SOC charged =", sum(action_history) * charge_env.delta_soc_interval)
print("Total state value =", sum(filter(None, reward_history)))
print("Total energyCharged cost = $", np.array(action_history) @ np.array(charge_env.price_curve) \
      * charge_env.delta_soc_interval)