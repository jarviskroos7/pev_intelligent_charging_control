import numpy as np
import matplotlib.pyplot as plt
from Charging_data import Charging_data


back_to_grid_charging =Charging_data()
back_to_grid_charging.get_charging_history("back_to_grid_charge_history_36.txt")
back_to_grid_charging.get_emission_volume_list("back_to_grid_charge_volume_36.txt")


plt.plot()

count = 0
for num in back_to_grid_charging.emission_volume_list:
    if num < 0:
        count += 1
percentage = (count / len(back_to_grid_charging.emission_volume_list)) * 100
print(percentage)

CO2_emission  = sum(back_to_grid_charging.emission_volume_list) /12/10**6
print(CO2_emission)

each_session_saved = CO2_emission/len(back_to_grid_charging.emission_volume_list)
print(each_session_saved)