import pandas as pd

from Smart_charging_method import Smart_charing_agent
from naive_charing_method import Naive_charing_agent

from Back_to_grid_method import Back_to_grid_charing_agent

naive_charging_agent = Naive_charing_agent()
naive_charging_agent.set_Resistance(36)
# naive_charging_agent.set_Resistance(36)
data = pd.read_csv("sessions_5_min_step.csv", index_col=0)


print(len(data))
session_pair = []
naive_charging_agent_emission_list = []


month_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

for i in range(len(data)):
    start_soc = float(data.loc[i, "start_soc"])
    # end_soc = data.loc[i, "end_soc"]
    end_soc = 0.95
    start_step = data.loc[i, "start_time_step_5_interval"]
    end_step = data.loc[i, "end_time_step_5_interval"]
    month = month_dict[data.loc[i, "month_plugin"]]

    if month >= 10 or month <= 3:
        season = "winter"
    else:
        season = "summer"
    naive_charging_emission, naive_charging_history = naive_charging_agent.get_total_emission_value(start_step, end_step, start_soc, end_soc, season)
    naive_charging_agent_emission_list.append(naive_charging_emission)

    with open('naive_charge_history_%s.txt'% naive_charging_agent.R, 'a') as f:
        f.write("%s\n "%naive_charging_history)

    with open('naive_charge_volume_%s.txt'% naive_charging_agent.R, 'a') as f:
        f.write("%s\n "%naive_charging_emission)