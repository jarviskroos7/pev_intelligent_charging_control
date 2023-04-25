import numpy as np
import matplotlib.pyplot as plt
from Charging_data import Charging_data

plt.rcParams['font.size'] = 14
def get_percentage_of_saving(naive_volume, smart_volume):
    return (np.sum(naive_volume) - np.sum(smart_volume))/np.sum(naive_volume)


smart_charging =Charging_data()
smart_charging.get_charging_history("smart_charge_history_36.txt")
smart_charging.get_emission_volume_list("smart_charge_volume_36.txt")

naive_charging =Charging_data()
naive_charging.get_charging_history("naive_charge_history_36.txt")
naive_charging.get_emission_volume_list("naive_charge_volume_36.txt")

smart_charging_group_by_step = smart_charging.get_charging_history_group_by_step()
naive_charging_group_by_step = naive_charging.get_charging_history_group_by_step()
smart_charging_group_by_hour = smart_charging.get_charging_history_group_by_hour()
naive_charging_group_by_hour = naive_charging.get_charging_history_group_by_hour()




fig, ax = plt.subplots()
ax.bar(range(len(smart_charging_group_by_hour)), naive_charging_group_by_hour*smart_charging.charge_interval/60/1000, alpha=0.8, label="Baseline Charging")
ax.bar(range(len(smart_charging_group_by_hour)), -smart_charging_group_by_hour*naive_charging.charge_interval/60/1000, alpha=0.8,label="Shift Charging")
# plt.ylim(-1.5e6,1e6)
ax.set_xlabel("Day Time(hour)")
ax.set_ylabel("Power Demand (kWh)")
ax.set_xticks(np.arange(0, 26, 4))

# y_max = np.max(smart_charging_group_by_hour)
# y_ticks_position = np.linspace(-y_max*10**order,y_max*10**order,10,dtype=np.float32)
# print(y_ticks_position)
# y_ticks = np.linspace(-y_max,y_max,10,dtype=np.float32)
# plt.yticks(y_ticks_position, y_ticks)

ax.legend()
ax.set_title("Power Demand With Different Charging Strategies")

plt.show()
fig.savefig("Power_Demand.png", dpi=800)

# sum1 = np.sum(naive_charging_group_by_hour[16:])
# sum2 = np.sum(smart_charging_group_by_hour[16:])
# print((sum1-sum2)/sum1)
# print("percentage of saving(energy):  %f"%get_percentage_of_saving(naive_charging.emission_volume_list, smart_charging.emission_volume_list))


# print(np.sum(naive_charging.emission_volume_list)/10**6)
# print(np.sum(smart_charging.emission_volume_list)/10**6)



# import matplotlib.pyplot as plt
# import numpy as np
#
# # Generate some sample data
# x = np.arange(0, 10, 0.1)
# y1 = np.sin(x)
# y2 = np.exp(x)
#
# # Create the plot
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# # Plot the first dataset on the left y-axis
# ax1.plot(x, y1, 'b-')
# ax1.set_ylabel('sin(x)', color='b')
# ax1.tick_params('y', colors='b')
#
# # Plot the second dataset on the right y-axis
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('exp(x)', color='r')
# ax2.tick_params('y', colors='r')
#
# # Set the y-axis limits to start at 0
# ax1.set_ylim(bottom=0)
# ax2.set_ylim(bottom=0)
#
# # Display the plot
# plt.show()



