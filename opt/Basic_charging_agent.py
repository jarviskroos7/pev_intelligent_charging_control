import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

class Basic_charging_agent():
    def __init__(self):
        self.voltage = 240  # nominal_voltage
        self.battery_volumn = 60 * 1000 / 400  # Q = kWh / v
        self.emission_max_value = 100
        self.Power = 9.6 * 1000  # power of the charger
        self.I_max = self.Power / self.voltage
        self.R = 0.00001  # resistance
        self.Power_limit = 10 * 1000  # simple assumption to the limit of the power: 100 kw
        self.Power_limit_slope_line_Intercept = 50 * 1000  # simple assumption to the limit of the power(the sloped line): 100 kw
        self.action_interval = 5
        self.step = self.action_interval / 60  # 10(min)/ 60(min)
        self.maximum_steps = int(24 / self.step * 2)  # 24 hours divided by step
        x = np.linspace(0, int(self.maximum_steps / 2), int(self.maximum_steps / 2) + 1)
        emission_array = self.emission_max_value / ((self.maximum_steps / 2 / 2) ** 2) * (x - (self.maximum_steps / 2 / 2)) ** 2
        self.emission_array = np.concatenate((emission_array[:-1], emission_array[:-1]), axis=0)

        winter_emission_data = pd.read_csv("pred_feb.csv")
        winter_emission_array = np.array(winter_emission_data['pred'].to_list())
        self.winter_emission_array = np.concatenate((winter_emission_array,winter_emission_array), axis=0)

        summer_emission_data = pd.read_csv("pred_may.csv")
        summer_emission_array = np.array(summer_emission_data['pred'].to_list())
        self.summer_emission_array = np.concatenate((summer_emission_array, summer_emission_array), axis=0)

    def get_power_limit(self,soc):
        power_limit_1 = self.Power_limit_slope_line_Intercept*(1- soc)
        self.Power_limit = min(self.Power, power_limit_1)
        return self.Power_limit
        # return min(self.Power, power_limit_1)

    def modify_emission_array(self, new_emission_array):
        if (new_emission_array.shape != self.emission_array.shape):
            raise Exception("new_emission_array has the wrong shape, the shape should be %s"%str(self.emission_array.shape))
        self.emission_array = new_emission_array