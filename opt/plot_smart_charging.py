import numpy as np
import matplotlib.pyplot as plt


class Charging_data():
    def __init__(self):
        self.charging_history = []
        self.emission_volume_list = []
        self.charge_interval = 5
        self.step_one_day = 24*(60/self.charge_interval)

    def read_original_charging_history(self, txt_name):
        result = []
        with open(txt_name, "r") as file:
            for line in file:
                if len(line) > 1:
                    my_list = eval(line)
                    result.append(my_list)
        return result

    def get_emission_volume_list(self, txt_name):
        with open(txt_name, "r") as file:
            lines = file.readlines()  # Read all the lines of the file into a list
            numbers = []
            for line in lines:
                line = line.strip()  # Remove any leading or trailing whitespace
                if line:  # Check if the line is not empty
                    numbers.append(float(line))  # Convert the line to a float and append to the list
        self.emission_volume_list = numbers
        return numbers

    def get_charging_history(self, txt_name):
        original_charging_history = self.read_original_charging_history(txt_name)
        charging_history = []
        for i in range(len(original_charging_history)):
            temp_history = [0]*int(self.step_one_day)
            for j in range(int(self.step_one_day*2)):
                temp_history[j%int(self.step_one_day)] += original_charging_history[i][j]
            charging_history.append(temp_history)
        self.charging_history = np.array(charging_history)

    def get_charging_history_group_by_step(self):
        return np.sum(self.charging_history, axis=0)

    def get_charging_history_group_by_hour(self):
        arr = self.get_charging_history_group_by_step()
        grouped = arr.reshape((int(len(arr)/12), 12))  # group elements into groups of 4
        new_arr = np.sum(grouped, axis=1)  # sum the elements in each group to create a new array with 24 elements
        return new_arr


def get_percentage_of_saving(naive_volume, smart_volume):
    return (np.sum(naive_volume) - np.sum(smart_volume))/np.sum(naive_volume)


smart_charging =Charging_data()
smart_charging.get_charging_history("smart_charge_history.txt")
smart_charging.get_emission_volume_list("smart_charge_volume.txt")

naive_charging =Charging_data()
naive_charging.get_charging_history("naive_charge_history.txt")
naive_charging.get_emission_volume_list("naive_charge_volume.txt")

smart_charging_group_by_step = smart_charging.get_charging_history_group_by_step()
naive_charging_group_by_step = naive_charging.get_charging_history_group_by_step()
smart_charging_group_by_hour = smart_charging.get_charging_history_group_by_hour()
naive_charging_group_by_hour = naive_charging.get_charging_history_group_by_hour()

# print(smart_charging_group_by_step)
plt.bar(range(len(smart_charging_group_by_hour)), naive_charging_group_by_hour, alpha=0.8, label="Naive Charging")
plt.bar(range(len(smart_charging_group_by_hour)), -smart_charging_group_by_hour, alpha=0.8,label="Smart Charging")
# plt.ylim(-1.5e6,1e6)



# y_max = np.max(smart_charging_group_by_hour)
# y_ticks_position = np.linspace(-y_max*10**order,y_max*10**order,10,dtype=np.float32)
# print(y_ticks_position)
# y_ticks = np.linspace(-y_max,y_max,10,dtype=np.float32)
# plt.yticks(y_ticks_position, y_ticks)

plt.legend()
plt.title("Power Demand With Different Charging Strategies")

plt.show()
print("percentage of saveing(energy):  %f"%get_percentage_of_saving(naive_charging.emission_volume_list, smart_charging.emission_volume_list))







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



