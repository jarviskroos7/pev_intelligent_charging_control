import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from Basic_charging_agent import Basic_charging_agent

class Navie_charing_agent(Basic_charging_agent):
    def __init__(self):
        super().__init__()

    def check_validation(self,start_time, end_time, start_soc, end_soc):
        target_charge_volumn = (end_soc - start_soc) * self.battery_volumn
        if target_charge_volumn > self.I_max * (end_time - start_time)*self.step:
            return False
        return True

    def get_total_emission_value(self, start_time, end_time, start_soc, end_soc,season):
        validation = self.check_validation(start_time, end_time, start_soc, end_soc)
        if not validation:
            return -1
        emission_volume = 0
        current_soc = start_soc
        current_time = start_time
        charging_history = []
        for i in range(self.maximum_steps):
            charging_history.append(0)
        while current_soc < end_soc and current_time < end_time:
            power_limit = self.get_power_limit(current_soc)

            if self.R == 0:
                current = power_limit / self.voltage
            else:
                current = (-self.voltage + np.sqrt(self.voltage*self.voltage + 8*self.R*power_limit))/(4*self.R)

            if (current_soc + current * self.step / self.battery_volumn)> end_soc:
                current = (end_soc - current_soc)*self.battery_volumn/self.step

            temp_power = current * self.voltage + current**2*self.R
            current_soc += current * self.step / self.battery_volumn
            if season == "summer":
                emission_volume += temp_power * self.summer_emission_array[current_time]
            if season == "winter":
                emission_volume += temp_power * self.winter_emission_array[current_time]
            # print(temp_power* self.emission_array[current_time])
            # print(self.Power)
            charging_history[current_time] = temp_power
            current_time += 1
        # print(current_time)
        # print(current_soc)
        # print(emission_volume - self.Power * self.emission_array[current_time])
        return emission_volume, charging_history





'''
    modify_emission_array:
        example:
            n = Navie_charing_agent()
            maximum_steps = 576
            x = np.linspace(0, int(maximum_steps / 2), int(maximum_steps / 2) + 1)
            emission_array = 1 / ((maximum_steps / 2 / 2) ** 2) * (x - (maximum_steps / 2 / 2)) ** 2
            emission_array = np.concatenate((emission_array[:-1], emission_array[:-1]), axis=0)
            n.modify_emission_array(emission_array)
'''


# n = Navie_charing_agent()
# print(n.get_total_emission_value(144, 288, 0.216, 0.99))

# n = Navie_charing_agent()
# print(n.get_total_emission_value(225, 253, 0.891, 0.99))

