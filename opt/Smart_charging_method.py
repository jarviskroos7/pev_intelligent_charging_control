import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from Basic_charging_agent import Basic_charging_agent


class Smart_charing_agent(Basic_charging_agent):
    def __init__(self):
        super().__init__()

    def check_validation(self,start_time, end_time, start_soc, end_soc):
        target_charge_volumn = (end_soc - start_soc) * self.battery_volumn
        if target_charge_volumn > self.I_max * (end_time - start_time)*self.step:
            return False
        return True

    def get_total_emission_value(self, start_time, end_time, start_soc, end_soc):

        target_charge_volumn = (end_soc - start_soc) * self.battery_volumn
        current_state = cp.Variable(self.maximum_steps, 'current at each step')
        P = cp.Variable(self.maximum_steps, 'power of the charger at each step')
        soc = cp.Variable(self.maximum_steps, "state of charge")
        voltage = cp.Variable(self.maximum_steps, "voltage")
        objective = cp.Minimize(cp.sum(P * self.emission_array))
        constraints = []

        for i in range(0, self.maximum_steps):
            constraints += [voltage[i] == 400]

        for i in range(0, self.maximum_steps):
            constraints += [P[i] == 2 * self.R * current_state[i] + 400 * current_state[i]]

        for i in range(start_time):
            constraints += [current_state[i] == 0]
            constraints += [soc[i] == 0]

        for i in range(end_time, self.maximum_steps):
            constraints += [current_state[i] == 0]

        constraints += [soc[start_time] == start_soc]

        for i in range(start_time + 1, end_time + 1):
            constraints += [soc[i] == soc[i - 1] + current_state[i - 1] * self.step / self.battery_volumn]

        for i in range(self.maximum_steps):
            constraints += [P[i] <= self.Power_limit_slope_line_Intercept * (1 - soc[i])]

        constraints += [cp.sum(current_state) * self.step >= (target_charge_volumn)]
        constraints += [P <= self.Power]
        constraints += [P >= 0]

        problem = cp.Problem(objective, constraints)
        emission_volume = problem.solve()
        # print(soc.value)
        P_values = P.value
        P_values = [value if value >= 1 else 0 for value in P_values]
        # print(np.count_nonzero(P_values))
        return emission_volume



# s = Smart_charing_agent()
# print(s.get_total_emission_value(144, 144+287, 0.2, 0.7))
#
# s = Smart_charing_agent()
# print(s.get_total_emission_value(260, 320, 0.4, 0.7))

# s = Smart_charing_agent()
# print(s.get_total_emission_value(225, 253, 0.891, 0.98))