import pybamm
import matplotlib.pyplot as plt


class cell_2170():
    # nom_cap = 4.8  # Ah
    # Nom_vol = 3.6  # V
    cut_off_lower = 2.6  # V
    cut_off_upper = 4.1  # V

    holding_voltage = 3.9  # V

    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values

    # Change width or height to change capacity
    # since original nominal capacity is 0.680616, use / 0.680616 * 3.9 to change to capacity of around 4 Ah
    # by default 0.207
    param['Electrode width [m]'] = 0.207 / 0.680616 * 4

    # Cut-off voltage
    # 2170 cell has lower bound of 2.5 volts, and upper bound of 4.2 volts
    # (searched on ChatGPT, official source needed)
    param['Lower voltage cut-off [V]'] = cut_off_lower
    param['Upper voltage cut-off [V]'] = cut_off_upper

    # Discharge and charge both use nominal capacity as reference for C-rate
    # by default, 0.680616
    param['Nominal cell capacity [A.h]'] = 4
    # for definition of C-rate, see https://web.mit.edu/evt/summary_battery_specifications.pdf

    def __init__(self, initial_voltage=3.3):

        """
        Initial Voltage at the time when the EV is plugged onto the wall
        We can suggest several initial_voltage values as different SOCs when start charging.
        :param initial_voltage:
        """
        self.init_V = initial_voltage

        # calibrate to find the <total capacity> and <partial capacity> and <capacity offset>
        experiment_cali = pybamm.Experiment(
            [
                "Discharge at 1.16 A until " + str(self.cut_off_lower) + " V",
                "Rest for 30 minutes",
                "Charge at 0.58 A until " + str(self.holding_voltage) + " V",
                "Hold at " + str(self.holding_voltage) + " V until 0 A",
                "Rest for 30 minutes"
            ] * 2
        )

        self.sim_cali = pybamm.Simulation(self.model, experiment=experiment_cali, parameter_values=self.param)
        self.sim_cali.solve()

        """
        cali as calibration.
        To calculate the total capacity and other relevant info
        """
        solution = self.sim_cali.solution
        cali_discharge_capacity_array = solution["Discharge capacity [A.h]"].data

        # for SOC calculation and true capacity calculation during charge and discharge
        self.total_capacity = max(cali_discharge_capacity_array) - min(cali_discharge_capacity_array)
        self.partial_capacity = max(cali_discharge_capacity_array)
        self.cap_offset = - min(cali_discharge_capacity_array)

        # for record keeping and plotting of voltage vs capacity
        self.cali_capacity_array = self.total_capacity - self.cap_offset - cali_discharge_capacity_array
        self.cali_v_array = solution['Terminal voltage [V]'].data
        self.cali_i_array = solution['Current [A]'].data

    def discharge_n_charge(self, elec_constraint, charging_current_or_power, end_constraint, duration_or_endV):
        """
        All the cell charging and discharging processes use this method.
        :param elec_constraint: "current" or "power"
        :param charging_current_or_power: a number in A or W
        :param end_constraint:  "duration" or "endV"
        :param duration_or_endV: a number in minutes or V
        :return:
        capacity_array: capacity records within the time step above (probably only the last element of array is needed)
        v_array: terminal voltage records within the time step above (probably only the last element of array is needed)
        """

        if elec_constraint == "current":
            experiment_statement_2 = "Charge at " + str(charging_current_or_power) + " A"
        elif elec_constraint == "power":
            experiment_statement_2 = "Charge at " + str(charging_current_or_power) + " W"

        if end_constraint == "duration" or "time":
            experiment_statement_2 += " for " + str(duration_or_endV) + " minutes"
        elif end_constraint == "voltage":
            experiment_statement_2 += " until " + str(duration_or_endV) + " V"

        experiment = pybamm.Experiment(
            [
                # discharge the battery cell until it reaches the initial Voltage condition (SOC condition) needed
                # 0.5 C = 0.5 * nominal capacity per hour = 2 Amp
                "Discharge at 0.5 C until " + str(self.init_V) + " V",
                experiment_statement_2
            ] * 1
        )

        model = pybamm.lithium_ion.SPM()

        self.sim_charge_n_discharge = pybamm.Simulation(model, experiment=experiment, parameter_values=self.param)
        self.sim_charge_n_discharge.solve()

        solution = self.sim_charge_n_discharge.solution
        # note that discharge_capacity_array is how much capacity is "DISCHARGED"
        discharge_capacity_array = solution["Discharge capacity [A.h]"].data
        v_array = solution['Terminal voltage [V]'].data
        i_array = solution['Current [A]'].data

        capacity_status_array = self.total_capacity - self.cap_offset - discharge_capacity_array

        return capacity_status_array, v_array, i_array


if __name__ == '__main__':
    cell_model = cell_2170()
    print("Total capacity counts from upper cut-off voltage to lower cut-off voltage:", cell_model.total_capacity)
    print("Partial capacity counts from default initial voltage to lower cut-off voltage:", cell_model.partial_capacity)

    experiment_cap_array, experiment_v_array, experiment_i_array = cell_model.discharge_n_charge("current", 0.5, "duration", 60)
    # print(experiment_cap_array)
    # print(experiment_v_array)

    output_variables = [
        "Terminal voltage [V]",
        "Current [A]",
        # "Electrolyte concentration [mol.m-3]",
        # "Negative particle surface concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        "Discharge capacity [A.h]"
    ]
    cell_model.sim_charge_n_discharge.plot(output_variables)

    fig, ax = plt.subplots()
    ax.plot(cell_model.cali_capacity_array, cell_model.cali_v_array)
    ax.set_title("Voltage vs Capacity curve during calibration")
    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Terminal Voltage (V)")
    plt.show()
