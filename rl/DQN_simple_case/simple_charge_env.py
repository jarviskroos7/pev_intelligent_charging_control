
import numpy as np

import gym
from gym import logger, spaces



'''
    observation_space: 
                        high        low
        current_soc:     1           0
        target_soc:      1           0
        start_time:     1440         0
        end_time:       2880         0
        current_time:   end_time   start_time
            P:         10000(kw)     0
          I_max:        1e5          0
    action_space: 0 - 100 amp ( with the interval of 0.5A)      
'''

max_current = 25
min_current = 0
current_interval = 0.5


#time interval between change the current (I)
step = 10

total_time = 1440

start_time_max = total_time / step
end_time_max = total_time / step * 2

min_charge_intervals = 60/step

battery_volume = 60 * 1000 / 400 #amp
resistance = 1
power_boundary = 10* 1000
power_boundary_decrease_point= 0.8
voltage = 400
class Simple_charge_env:
    def __init__(self):

        self.current_soc = np.random.uniform(0, 0.8)
        self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)

        self.start_time = np.random.randint(0, start_time_max)
        self.end_time = np.random.randint(start_time_max + self.start_time, start_time_max + self.start_time +  min_charge_intervals)

        self.action_space = spaces.Discrete(int((max_current-min_current)/current_interval))

        # self.observation_space = spaces.Box(-np.array([0, 0, 0, 0, 0, 0, 0]),
        #                                     np.array([1, 1, 1e5, 1e5, 1e5, 1e5, 1e5]), dtype=np.float32)

        self.observation_space = spaces.Box(-np.array([0, 0, 0, 0,0]),
                                            np.array([ 1e5, 1e5, 1e5, 1e5,1e5]), dtype=np.float32)


        # self.voltage = 0.4

        self.current_list = np.linspace(0, max_current, int((max_current-min_current)/current_interval))

        self.resistance = float(resistance)

        self.power_boundary = float(power_boundary)

        self.power_boundary_decrease_point = float(power_boundary_decrease_point)

        self.current_time = self.start_time

        self.charge_interval = step

        self.battery_volume = float(battery_volume)
        self.voltage = voltage


    def get_power_limit(self):
        if self.current_soc < self.power_boundary_decrease_point:
            return self.power_boundary
        else:
            return self.power_boundary - (self.power_boundary)/ (1.0-self.power_boundary_decrease_point)*(self.current_soc-self.power_boundary_decrease_point)

    def get_voltage(self):
        return self.voltage

    def get_I_limit(self):
        # I = (-U + sqrt(U**2+8*R*P))/(4R)
        return (-self.voltage + np.sqrt(self.voltage**2 + 8 * self.resistance * self.current_power_limit))/(4*self.resistance)

    def reset(self):
        self.current_soc = np.random.uniform(0, 0.8)
        self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)

        self.start_time = np.random.randint(0, start_time_max)
        self.end_time = np.random.randint(start_time_max + self.start_time, start_time_max + self.start_time +  min_charge_intervals)
        self.current_time = self.start_time

        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()


        return np.array([self.current_soc,
                         self.target_soc,
                         self.current_time,
                         self.end_time,
                         self.I_max
                         ])


        # return np.array([self.current_soc,
        #                  self.target_soc,
        #                  self.start_time,
        #                  self.end_time,
        #                  self.current_time,
        #                  self.current_power_limit,
        #                  self.I_max
        #                  ])

    def reset_with_values(self,
                          current_soc,
                          target_soc,
                          start_time,
                          end_time,):
        self.current_soc = current_soc
        self.target_soc = target_soc

        self.start_time = start_time
        self.end_time = end_time
        self.current_time = self.start_time

        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()


        return np.array([self.current_soc,
                         self.target_soc,
                         self.current_time,
                         self.end_time,
                         self.I_max
                         ])


        # return np.array([self.current_soc,
        #                  self.target_soc,
        #                  self.start_time,
        #                  self.end_time,
        #                  self.current_time,
        #                  self.current_power_limit,
        #                  self.I_max
        #                  ])

    def step(self, action):
        current = self.current_list[action]
        if current > self.I_max:
            current = self.I_max
        self.current_soc += self.charge_interval * current /self.battery_volume / 60
        self.current_time += 1
        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()

        observation = np.array([self.current_soc,
                         self.target_soc,
                         self.current_time,
                         self.end_time,
                         self.I_max
                         ])

        # observation = np.array([self.current_soc,
        #                          self.target_soc,
        #                          self.start_time,
        #                          self.end_time,
        #                          self.current_time,
        #                          self.current_power_limit,
        #                          self.I_max
        #                          ])
        reward = 0
        terminated = False
        if ((self.current_soc >=  self.target_soc) or (self.end_time == self.current_time)):

            terminated = True

        info = {}

        return observation, reward, terminated, False, info



