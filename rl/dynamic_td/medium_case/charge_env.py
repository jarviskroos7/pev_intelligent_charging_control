import numpy as np
import time
class Charge_env():
    def __init__(self):
        self.voltage = 240
        self.battery_volume = 60 * 1000 / self.voltage  # amp
        self.resistance = 0
        self.power_boundary = 9.6 * 1000
        self.power_boundary_decrease_point = 0.8
        self.I_max = self.power_boundary / self.voltage

        self.action_size = 2
        self.actions_prob = 1 / self.action_size
        self.actions = [i for i in range(self.action_size)]
        self.action_current_list = np.linspace(0, self.I_max, self.action_size)
        self.charge_interval = 15

        self.delta_soc_interval = 0.04
        self.state_size_delta_soc = int(1 / self.delta_soc_interval) + 1
        self.state_size_delta_time = int(1440 / self.charge_interval)
        self.state_size_time = int(1440 / self.charge_interval * 2)

        self.price_max_value = 1
        x = np.linspace(0, int(self.state_size_delta_time) - 1, int(self.state_size_delta_time))
        price_curve = self.price_max_value / ((self.state_size_delta_time / 2) ** 2) * (x - (self.state_size_delta_time / 2)) ** 2
        self.price_curve = np.concatenate((price_curve, price_curve), axis=0)

    def is_terminal(self,state):
        if ((state[0] <= 0) or (state[1] <= 0) or (state[2] >= self.state_size_time - 1)):
            return True
        return False

    def step(self,state, action):
        if self.is_terminal(state):
            # if state[0] == 0:
            #     return state, 0, True
            return state, -state[0] * 10 * self.price_max_value, True
        new_state = [0, 0, 0]

        # new_state[0] = state[0] - self.action_current_list[action] * self.charge_interval / 60 / self.battery_volume

        if action == 1:
            new_state[0] = state[0] - self.delta_soc_interval
        else:
            new_state[0] = state[0]
        new_state[1] = state[1] - 1
        new_state[2] = state[2] + 1

        #     new_state[0] = np.round(state[0]*(1+increase_rate_tumor/100),1)
        #     new_state[1] = np.round(state[1]*(1+increase_rate_bad_feeling/100),1)
        #     new_state[0] = min(10, new_state[0])
        #     new_state[1] = min(10, new_state[1])
        #     new_state[0] = max(1, new_state[0])
        #     new_state[1] = max(1, new_state[1])
        #     new_state[2] = max(drugB_usage, state[2])
        new_state[1] = max(new_state[1], 0)
        new_state[2] = min(new_state[2], self.state_size_time - 1)

        if new_state[1] < 0:
            raise Exception("delta time out of bound")
        if new_state[2] >= self.state_size_time:
            raise Exception("current time out of bound")

        reward = - action * self.price_curve[state[2]]
        new_state[0] = np.round(new_state[0],2)
        done = False
        if ((new_state[0] <= 0) or (new_state[1] <= 0) or (new_state[2] >= self.state_size_time)):
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
            print(t2 - t1)
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