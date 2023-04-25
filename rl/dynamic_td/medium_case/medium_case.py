import numpy as np
import time
# voltage = 240
# battery_volume = 60 * 1000 / voltage #amp
# resistance = 0
# power_boundary = 9.6 * 1000
# power_boundary_decrease_point = 0.8
# I_max = power_boundary/voltage
#
# action_size = 2
# actions_prob = 1/action_size
# actions = [i for i in range(action_size)]
# action_current_list = np.linspace(0, I_max ,action_size)
# charge_interval = 30
#
# delta_soc_interval = 0.1
# state_size_delta_soc = int(1/delta_soc_interval) + 1
# state_size_delta_time = int(1440/charge_interval)
# state_size_time = int(1440/charge_interval * 2)
#
#
# price_max_value = 1
# x = np.linspace(0, int(state_size_delta_time)-1, int(state_size_delta_time))
# price_curve = price_max_value/((state_size_delta_time/2)**2) * (x-(state_size_delta_time/2))**2
# price_curve = np.concatenate((price_curve, price_curve), axis =0)
#
#
# def is_terminal(state):
#     if ((state[0] <= 0) or(state[1] <= 0) or (state[2] >= state_size_time-1)):
#         return True
#     return False
#
#
# def step(state, action):
#     if is_terminal(state):
#         # if state[0] == 0:
#         #     return state, 0, True
#         return state, -state[0]*10*price_max_value, True
#     new_state = [0, 0, 0]
#
#     # new_state[0] = state[0] - action_current_list[action]* charge_interval/60 / battery_volume
#     #
#     if action == 1:
#         new_state[0] = state[0] - delta_soc_interval
#     else:
#         new_state[0] = state[0]
#     new_state[1] = state[1] - 1
#     new_state[2] = state[2] + 1
#
#     #     new_state[0] = np.round(state[0]*(1+increase_rate_tumor/100),1)
#     #     new_state[1] = np.round(state[1]*(1+increase_rate_bad_feeling/100),1)
#     #     new_state[0] = min(10, new_state[0])
#     #     new_state[1] = min(10, new_state[1])
#     #     new_state[0] = max(1, new_state[0])
#     #     new_state[1] = max(1, new_state[1])
#     #     new_state[2] = max(drugB_usage, state[2])
#     new_state[1] = max(new_state[1], 0)
#     new_state[2] = min(new_state[2], state_size_time - 1)
#
#     if new_state[1] < 0:
#         raise Exception("delta time out of bound")
#     if new_state[2] >= state_size_time:
#         raise Exception("current time out of bound")
#
#     reward = - action * price_curve[state[2]]
#
#     done = False
#     if ((new_state[0] <= 0) or (new_state[1] <= 0) or (new_state[2] >= state_size_time)):
#         done = True
#         reward += - new_state[0] * 10 *price_max_value
#     return new_state, reward, done
#
# def get_index(state):
#     index = int(np.round(state/delta_soc_interval))
#     return index
#
#
#
# def compute_state_value(max_iter=9, discount=1, policy=actions_prob * np.ones(
#     (state_size_delta_soc, state_size_delta_time, state_size_time, action_size))):
#     new_state_values = np.zeros((state_size_delta_soc, state_size_delta_time, state_size_time))
#     iteration = 0
#
#     while iteration <= max_iter:
#         t1 = time.time()
#         state_values = new_state_values.copy()
#         old_state_values = state_values.copy()
#
#         for i in np.linspace(0, 1, state_size_delta_soc):
#             # print(i)
#             for j in range(int(state_size_delta_time)):
#                 for m in range(int(state_size_time) - j):
#                     i = np.round(i, 2)
#                     # print(i)
#                     index_i = get_index(i)
#                     # print(index_i)
#                     value = 0
#                     for k, a in enumerate(actions):
#                         (next_i, next_j, next_m), reward, done = step([i, j, m], a)
#                         next_index_i = get_index(next_i)
#                         value += policy[index_i, j, m, k] * (
#                                     reward + discount * state_values[next_index_i, next_j, next_m])
#                     new_state_values[index_i, j, m] = value
#
#         iteration += 1
#         t2 = time.time()
#         print(t2 - t1)
#     return new_state_values, iteration
#
#
# def greedy_Policy(values,discount = 1):
#     new_state_values = values
#     policy = np.zeros((state_size_delta_soc, state_size_delta_time, state_size_time, action_size))
#
#     state_values = new_state_values.copy()
#
#     for i in np.linspace(0, 1, state_size_delta_soc):
#         for j in range((int(state_size_delta_time))):
#             for m in range(int(state_size_time) - j):
#                 i = np.round(i, 2)
#                 index_i = get_index(i)
#                 value = np.min(values);
#                 for k,a in enumerate(actions):
#                     (next_i, next_j, next_m), reward, done = step([i, j, m], a)
#                     next_index_i = get_index(next_i)
#                     valtemp = reward + discount*state_values[next_index_i, next_j, next_m]
#                     if valtemp > value:
#                         value = valtemp
#                         actionind = k
#
#
#                 policy[index_i,j,m,actionind] = 1
#
#     return policy

# policy0 = actions_prob * np.ones((state_size_delta_soc, state_size_delta_time, state_size_time, action_size))
# values, sync_iteration = compute_state_value(max_iter=3, policy = policy0)
# greedy_policy = greedy_Policy(values)
#
# values, sync_iteration = compute_state_value(max_iter=3, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# values, sync_iteration = compute_state_value(max_iter=3, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# np.save("policy.npy", greedy_policy)




import numpy as np
import time
from tqdm import tqdm
from charge_env import Charge_env
charge_env = Charge_env()
policy0 = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))


num_iteration = 30
# policy = charge_env.actions_prob * np.ones((charge_env.state_size_delta_soc, charge_env.state_size_delta_time, charge_env.state_size_time, charge_env.action_size))
policy = np.load('medium_greedy_policy.npy')
for i in tqdm(range(num_iteration)):
    values, sync_iteration = charge_env.compute_state_value(max_iter=5, discount = 1,policy = policy)
    policy = charge_env.greedy_Policy(values)
    np.save(f'medium_greedy_policy_iter{i}', policy)
# np.save(f'greedy_policy_iter{num_iteration}', policy)

# values, sync_iteration = charge_env.compute_state_value(max_iter=3, discount = 1,policy = policy0)
# greedy_policy = charge_env.greedy_Policy(values)
#
# values, sync_iteration = charge_env.compute_state_value(max_iter=3,discount = 1, policy = greedy_policy)
# greedy_policy = charge_env.greedy_Policy(values)
# values, sync_iteration = charge_env.compute_state_value(max_iter=3, discount = 1,policy = greedy_policy)
# greedy_policy = charge_env.greedy_Policy(values)
#
#
#
# values, sync_iteration = charge_env.compute_state_value(max_iter=3, discount = 1,policy = policy0)
# greedy_policy = charge_env.greedy_Policy(values)
#
# values, sync_iteration = charge_env.compute_state_value(max_iter=3, discount = 1,policy = greedy_policy)
# greedy_policy = charge_env.greedy_Policy(values)
# values, sync_iteration = charge_env.compute_state_value(max_iter=3, discount = 1,policy = greedy_policy)
# greedy_policy = charge_env.greedy_Policy(values)

# values, sync_iteration = compute_state_value(max_iter=3, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
#
# values, sync_iteration = compute_state_value(max_iter=3, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
#
# values, sync_iteration = compute_state_value(max_iter=3, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# np.save("policy.npy", greedy_policy)
print(123)