import numpy as np

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

    def get_CO2_emission_per_session(self):
        CO2_emission_total = sum(self.emission_volume_list) * self.charge_interval/60 / 10 ** 6
        self.each_session_saved = CO2_emission_total/len(self.emission_volume_list)
        return self.each_session_saved