import numpy as np
import time
import pandas as pd
import multiprocessing as mp

"""
modified based on NORMAL case

action space: [0, 1, 2] 
    0: discharge
    1: do nothing
    2: charge

state space: [delta_soc, delta_t, current_t]
delta_soc interval:     0.5%
delta_t interval:       12min

price signal: data/price_day_idx_12min.csv
battery discharge soc_limit = 20%

est. convergence and optimal iterations =       ?
est. time used for convergence =                ?
"""


class final_env():

    def __init__(self):

        self.voltage = 240                                                                          # V
        self.battery_capacity = 80                                                                  # kWh
        self.battery_volume = self.battery_capacity * 1000 / self.voltage                           # Ah
        self.resistance = 0                                                                         # R
        self.power_boundary = 9.6 * 1000                                                            # W
        self.power_boundary_decrease_point = 0.8
        self.I_max = self.power_boundary / self.voltage                                             # 40A

        self.action_size = 3
        self.actions_prob = 1 / self.action_size
        self.actions = [i for i in range(self.action_size)]                                         # V2G
        
        # negative_current_list  = np.linspace(-self.I_max, 0, int((self.action_size+1)/2))
        # positive_current_list =  np.linspace( 0, self.I_max, int((self.action_size+1)/2))
        # self.action_current_list = np.concatenate((negative_current_list[:-1], positive_current_list))

        self.charge_interval = 12                                                                   # min
        self.delta_soc_interval = 0.02                                                              # 40*12/60*0.85/333.333 = 0.0204 * 100 ~= 2%
        self.state_size_delta_soc = int(1 / self.delta_soc_interval) + 1
        self.state_size_delta_time = int(1440 / self.charge_interval)
        self.state_size_time = int(1440 / self.charge_interval * 2)

        self.price_curve = pd.read_csv('../../../data/price_day_idx_12min.csv')['price'].values     # $/kWh
        self.price_curve = np.concatenate((self.price_curve, self.price_curve), axis=0)
        self.price_max_value = max(self.price_curve)
        self.loss_coefficient = 0.85
        self.v2g_discount = 0.8
        self.soc_limit = 0.2

        # self.price_max_value = 1
        # x = np.linspace(0, int(self.state_size_delta_time) - 1, int(self.state_size_delta_time))
        # price_curve = self.price_max_value / ((self.state_size_delta_time / 2) ** 2) * (x - (self.state_size_delta_time / 2)) ** 2
        # self.price_curve = np.concatenate((price_curve, price_curve), axis=0)
        # self.loss_coefficient = 0.85

    def is_terminal(self, state):
        # if ((state[0] <= 0) or (state[1] <= 0) or (state[2] >= self.state_size_time - 1)):
        #     return True
        # return False
        if ((state[1] <= 0) or (state[2] >= self.state_size_time - 1)):
            return True
        return False

    def at_soc_limit(self, state):
        if state[0] >= 1 - self.soc_limit:
            # deltaSoc >= 0.8
            # battery is at or below a set SOC limit
            return True
        else:
            return False
        
    def step(self,state, action):
        if self.is_terminal(state):
            # if state[0] == 0:
            #     return state, 0, True
            return state, -state[0] * 10 * self.price_max_value, True
        
        new_state = [0, 0, 0]

        # for variable current
        # new_state[0] = state[0] - self.action_current_list[action] * self.charge_interval / 60 / self.battery_volume

        # do not discharge if SOC is at or below the SOC limit
        if self.at_soc_limit(state) and action == -1:
            action = 1 # DO NOTHING

        # CHARGING
        if action == 2:
            new_state[0] = state[0] - self.delta_soc_interval
        # DISCHARGING
        elif action == 0:
            new_state[0] = state[0] + self.delta_soc_interval
        # DO NOTHING
        elif action == 1:
            new_state[0] = state[0]
        else:
            raise Exception('Invaid action!')

        new_state[1] = state[1] - 1
        new_state[2] = state[2] + 1
        new_state[0] = min(new_state[0], 1)
        new_state[1] = max(new_state[1], 0)
        new_state[2] = min(new_state[2], self.state_size_time - 1)
        new_state[0] = max(new_state[0], 0)

        if new_state[1] < 0:
            raise Exception("delta time out of bound")
        if new_state[2] >= self.state_size_time:
            raise Exception("current time out of bound")

        if np.round(new_state[0], 2) == np.round(state[0], 2):
            action = 1

        # DISCHARGING
        if action == 0:
            reward = action * self.price_curve[state[2]] * 0.8
        # DO NOTHING
        elif action == 1:
            reward = 0
        # CHARGING
        else:
            reward = - action * self.price_curve[state[2]]

        new_state[0] = np.round(new_state[0], 2)
        done = False
        if ((new_state[1] <= 0) or (new_state[2] >= self.state_size_time)):
            done = True
        return new_state, reward, done

    def get_index(self, state):
        index = int(np.round(state / self.delta_soc_interval))
        return index

    def compute_state_value(self,max_iter, discount, policy):
        new_state_values = np.zeros((self.state_size_delta_soc, self.state_size_delta_time, self.state_size_time))
        iteration = 0

        while iteration <= max_iter:
            t1 = time.time()
            state_values = new_state_values.copy()
            old_state_values = state_values.copy()

            for i in np.linspace(0, 1, self.state_size_delta_soc):
                # print(i)
                for j in range(int(self.state_size_delta_time)):
                    for m in range(int(self.state_size_time) - j):
                        i = np.round(i, 2)
                        # print(i)
                        index_i = self.get_index(i)
                        # print(index_i)
                        value = 0
                        for k, a in enumerate(self.actions):
                            (next_i, next_j, next_m), reward, done = self.step([i, j, m], a)
                            next_index_i = self.get_index(next_i)
                            value += policy[index_i, j, m, k] * (
                                    reward + discount * state_values[next_index_i, next_j, next_m])
                        new_state_values[index_i, j, m] = value

            iteration += 1
            t2 = time.time()
            # print(t2 - t1)
        return new_state_values, iteration

    def greedy_Policy(self,values,discount = 1):
        new_state_values = values
        policy = np.zeros((self.state_size_delta_soc, self.state_size_delta_time, self.state_size_time, self.action_size))

        state_values = new_state_values.copy()

        for i in np.linspace(0, 1, self.state_size_delta_soc):
            for j in range((int(self.state_size_delta_time))):
                for m in range(int(self.state_size_time) - j):
                    i = np.round(i, 2)
                    index_i = self.get_index(i)
                    value = np.min(values);
                    for k,a in enumerate(self.actions):
                        (next_i, next_j, next_m), reward, done = self.step([i, j, m], a)
                        next_index_i = self.get_index(next_i)
                        valtemp = reward + discount*state_values[next_index_i, next_j, next_m]
                        if valtemp > value:
                            value = valtemp
                            actionind = k


                    policy[index_i,j,m,actionind] = 1

        return policy
    
    def state_value_parallel_loop(self, args):

        i, j, m, policy, actions, discount, state_values, new_state_values = args
        i = np.round(i, 2)
        index_i = self.get_index(i)
        value = 0
        for k, a in enumerate(actions):
            (next_i, next_j, next_m), reward, done = self.step([i, j, m], a)
            next_index_i = self.get_index(next_i)
            value += policy[index_i, j, m, k] * (reward + discount * state_values[next_index_i, next_j, next_m])
        new_state_values[index_i, j, m] = value
        
    def compute_state_value_parallel(self, max_iter, discount, policy):
        new_state_values = np.zeros((self.state_size_delta_soc, self.state_size_delta_time, self.state_size_time))
        iteration = 0

        while iteration <= max_iter:
            t1 = time.time()
            state_values = new_state_values.copy()
            old_state_values = state_values.copy()

            args_list = []
            for i in np.linspace(0, 1, self.state_size_delta_soc):
                for j in range(int(self.state_size_delta_time)):
                    for m in range(int(self.state_size_time) - j):
                        args = (i, j, m, policy, self.actions, discount, state_values, new_state_values)
                        args_list.append(args)

            with mp.Pool() as pool:
                pool.map(self.state_value_parallel_loop, args_list)

            iteration += 1
            t2 = time.time()
            print(f'state-value iteration {iteration} time =', round(t2 - t1, 2), 's')
        
        return new_state_values, iteration
    
    def greedy_policy_process(self, i, j, m, values, discount, actions):
        value = np.min(values)
        actionind = 0
        for k, a in enumerate(actions):
            (next_i, next_j, next_m), reward, done = self.step([i, j, m], a)
            next_index_i = self.get_index(next_i)
            valtemp = reward + discount * values[next_index_i, next_j, next_m]
            if valtemp >= value:
                value = valtemp
                actionind = k

        return (i, j, m, actionind)

    def greedy_policy_parallel(self, values, discount=1):

        t1 = time.time()
        new_state_values = values
        policy = np.zeros((self.state_size_delta_soc, self.state_size_delta_time, self.state_size_time, self.action_size))
        state_values = new_state_values.copy()

        pool = mp.Pool()
        tasks = []

        for i in np.linspace(0, 1, self.state_size_delta_soc):
            for j in range(int(self.state_size_delta_time)):
                for m in range(int(self.state_size_time) - j):
                    i = np.round(i, 2)
                    tasks.append((i, j, m, values, discount, self.actions))

        # map the tasks to the worker processes in parallel
        results = pool.starmap(self.greedy_policy_process, tasks)

        # update the policy based on the results
        for i, j, m, actionind in results:
            index_i = self.get_index(i)
            policy[index_i, j, m, actionind] = 1

        pool.close()        
        t2 = time.time()
        print(f'greedy policy evaluation time =', round(t2 - t1, 2), 's')

        return policy