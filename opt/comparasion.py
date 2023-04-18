import pandas as pd

from Smart_charging_method import Smart_charing_agent
from naive_charing_method import Navie_charing_agent

smart_agent = Smart_charing_agent()
naive_agent = Navie_charing_agent()

data = pd.read_csv("sessions_5_min_step.csv", index_col=0)


print(len(data))
session_pair = []
smart_agent_emission_list = []
naive_agent_emission_list = []
for i in range(123,126):
    start_soc = data.loc[i, "start_soc"]
    # end_soc = data.loc[i, "end_soc"]
    end_soc = 0.95
    start_step = data.loc[i, "start_time_step_5_interval"]
    end_step = data.loc[i, "end_time_step_5_interval"]
    smart_emission = smart_agent.get_total_emission_value(start_step, end_step, start_soc, end_soc)
    naive_emission = naive_agent.get_total_emission_value(start_step, end_step, start_soc, end_soc)
    smart_agent_emission_list.append(smart_emission)
    naive_agent_emission_list.append(naive_emission)

# for i in range(len(naive_agent_emission_list)):
#     print(naive_agent_emission_list[i], smart_agent_emission_list[i])
print((sum(naive_agent_emission_list)-sum(smart_agent_emission_list))/sum(naive_agent_emission_list))


