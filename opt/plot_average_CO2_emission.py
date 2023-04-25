import numpy as np
import matplotlib.pyplot as plt
from Charging_data import Charging_data
back_to_grid_charging =Charging_data()
smart_charging =Charging_data()
naive_charging = Charging_data()

plt.rcParams['font.size'] = 14


def get_CO2_emission_per_session(charging_agent,history_txt,volume_txt):
    charging_agent.get_charging_history(history_txt)
    charging_agent.get_emission_volume_list(volume_txt)
    average_emission = charging_agent.get_CO2_emission_per_session()
    return average_emission
back_to_grid_value = get_CO2_emission_per_session(back_to_grid_charging,
                                                  "back_to_grid_charge_history_36.txt",
                                                  "back_to_grid_charge_volume_36.txt")
smart_charging_value = get_CO2_emission_per_session(smart_charging,
                                                  "smart_charge_history_36.txt",
                                                  "smart_charge_volume_36.txt")
naive_charging_value = get_CO2_emission_per_session(naive_charging,
                                                  "naive_charge_history_36.txt",
                                                  "naive_charge_volume_36.txt")

# print(back_to_grid_value)
# print(smart_charging_value)
# print(naive_charging_value)

charging_types = ['Baseline Charging', 'Shift Charging', 'Back to Grid ']
charging_values = [naive_charging_value, smart_charging_value, back_to_grid_value]
# charging_types = ['Baseline Charging', 'Shift Charging']
# charging_values = [naive_charging_value, smart_charging_value]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# colors = ['#1f77b4', '#ff7f0e']
# Create a bar chart
fig, ax = plt.subplots()
ax.bar(charging_types, charging_values, color=colors)
for i, v in enumerate(charging_values):
    if v >0:
        ax.text(i, v+0.2, str(np.round(v,1)), ha='center')
    else:
        ax.text(i, v-0.7, str(np.round(v,1)), ha='center')
# Add titles and labels
# ax.set_title('Average CO2 Emission Per Charging Session vs Charging Type')
ax.set_ylabel('Average Marginal CO2 Emission(lbs)')
ax.axhline(y=0, color='r', linestyle='--')
ax.set_ylim(-2,9)
# Display the chart
plt.savefig("CO2_emission_plot.png", dpi=800)
plt.show()
