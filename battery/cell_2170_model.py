import pybamm
import matplotlib.pyplot as plt


class cell_2170():
    # nom_cap = 4.8  # Ah
    # Nom_vol = 3.6  # V
    cut_off_lower = 3.1  # V
    cut_off_upper = 4.1  # V

    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values

    # Change width or height to change capacity
    # since original nominal capacity is 0.680616, use / 0.680616 * 3.9 to change to capacity of around 4 Ah
    # by default 0.207
    param['Electrode width [m]'] = 0.207 / 0.680616 * 3.9

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

        # init for the following methods
        self.total_capacity = None
        self.partial_capacity = None
        self.sim_charge = None
        self.sim_cali = None

    def find_capacity(self):
        experiment_cali = pybamm.Experiment(
            [
                "Discharge at 0.5 C until " + str(self.cut_off_lower) + " V",
                "Charge at 0.5 C until " + str(self.cut_off_upper) + " V"
            ] * 2
        )

        self.sim_cali = pybamm.Simulation(self.model, experiment=experiment_cali, parameter_values=self.param)
        self.sim_cali.solve()

        solution = self.sim_cali.solution

        dcap = solution["Discharge capacity [A.h]"].data
        V = solution['Terminal voltage [V]'].data

        self.total_capacity = max(dcap) - min(dcap)
        self.partial_capacity = max(dcap)

        return self.total_capacity, self.partial_capacity, dcap, V

    def charge_cell(self, charging_current, duration):
        experiment = pybamm.Experiment(
            [
                # discharge the battery cell until it reaches the initial Voltage condition (SOC condition) needed
                "Discharge at 2C until " + str(self.init_V) + " V",
                "Charge at " + str(charging_current) + " A for " + str(duration) + " minutes"
            ] * 1
        )

        self.sim_charge = pybamm.Simulation(self.model, experiment=experiment, parameter_values=self.param)
        self.sim_charge.solve()

        solution = self.sim_charge.solution
        ending_disch_cap = solution["Discharge capacity [A.h]"].data[-1]
        ending_v = solution['Terminal voltage [V]'].data[-1]

        total_cap, _, _, _ = self.find_capacity()

        ending_soc = (total_cap - ending_disch_cap) / total_cap

        return ending_soc, ending_v


if __name__ == '__main__':
    cell_model = cell_2170()
    v_after_charge, cap_after_charge = cell_model.charge_cell(0.5, 60)
    print(v_after_charge, cap_after_charge)

    output_variables = [
        "Terminal voltage [V]",
        "Current [A]",
        # "Electrolyte concentration [mol.m-3]",
        # "Negative particle surface concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        "Discharge capacity [A.h]"
    ]

    cell_model.sim_charge.plot(output_variables)

    # total capacity counts from
    total_capacity, partial_capacity, dcap, V = cell_model.find_capacity()
    print("Total capacity counts from upper cut-off voltage to lower cut-off voltage = ", total_capacity)
    print("Partial capacity counts from default initial voltage to lower cut-off voltage = ", partial_capacity)

    fig, ax = plt.subplots()
    ax.plot(dcap, V)
    plt.show()