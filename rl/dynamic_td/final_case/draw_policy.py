import numpy as np
from env import final_env
policy = np.load('final_greedy_policy_12min_charging_limit_iter5x120.npy')

# state = [0.8, 47,0]
# state = [0.28, 47,0]
state = [0.8, 119, 0]
# state = [0.28, 47,0]
done = False
charge_env = final_env()

baseline_action = [None] * charge_env.state_size_time 
opt_action = baseline_action.copy()
baseline_reward = baseline_action.copy()
opt_reward = baseline_action.copy()

# step through optimal policy
while not done:
    state_0_index = charge_env.get_index(state[0])
    action_prob = policy[state_0_index, state[1], state[2], :]
    action = np.argmax(action_prob)
    opt_action[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    opt_reward[state[2]] = reward
    state = new_state

# step through baseline
state = [0.8, 119, 0]
done = False

while not done:
    action = 2
    baseline_action[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    baseline_reward[state[2]] = reward
    state = new_state

for idx, action in enumerate(opt_action):
    if action == 0:
        # discharging
        opt_action[idx] = -1
    elif action == 1:
        # do nothing
        opt_action[idx] = 0
    elif action == 2:
        # charging
        opt_action[idx] = 1
    else:
        pass

baseline_action = [0 if a is None else a for a in baseline_action]
baseline_action = np.array(baseline_action) / 2

print(opt_action)
opt_action = [0 if a is None else a for a in opt_action]
opt_cost = np.array(opt_action) @ np.array(charge_env.price_curve) \
      * charge_env.delta_soc_interval
baseline_cost = np.array(baseline_action) @ np.array(charge_env.price_curve) \
      * charge_env.delta_soc_interval

print("======== Optimal Policy: ========")
print("Total SOC charged =", sum(opt_action) * charge_env.delta_soc_interval)
print("Total state value =", sum(filter(None, opt_reward)))
print("Total energyCharged cost = $", opt_cost)

print()
print("======= Baseline Policy: ========")
print("Total energyCharged cost = $", baseline_cost)

print()
print("========= Evaluation: ===========")
print("Charging Cost saved = $", round(opt_cost - baseline_cost, 3), ",", round((opt_cost - baseline_cost) / baseline_cost * 100, 3), "%")