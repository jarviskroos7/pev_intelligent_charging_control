import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib.colors import ListedColormap
from simple_charge_env import current_interval


emission_max_value = 100
start_time_max = 144

height = 10
pkl_file = './predict_output/predict_11.pkl'

with open(pkl_file, 'rb') as f:
    result = pickle.load(f)

print(result)
# create a simple 2D array
data = np.array(result['action_history'])
start_soc = result['start_soc']
target_soc = result['target_soc']
end_soc = result['current_soc']

start_time = int(result['start_time'])
current_time = int(result['current_time'])
end_time = int(result['end_time'])

filling_zeros_front = list(np.zeros(start_time))
filling_zeros_back = list(np.zeros(end_time-current_time))
# print(filling_zeros_front + list(result['action_history']))
action_history = [ action * current_interval for action in result['action_history']]
data = filling_zeros_front + action_history + filling_zeros_back

data = np.array(data*height)
# data = np.array(result['action_history']*height)
data = data.reshape(int(height), int(len(data)/height))


# plot the data with the colormap

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))

fig.suptitle("start_soc {:.2f} target_soc {:.2f} end_soc {:.2f} \n start_time {:} end_time {:}".format(np.round(start_soc,2), np.round(target_soc,2),
                                                                        np.round(end_soc,2), start_time, end_time))

# fig.suptitle("start_soc %f end_soc % f"%(np.round(start_soc,1), np.round(target_soc,1)))
ax1.get_yaxis().set_visible(False)
im = ax1.imshow(data,cmap='Greens')
# ax1.set_xticks(np.linspace(start_time,end_time,int((end_time-start_time)/10)+1))
ax1.set_xlabel("Step")
cbar = fig.colorbar(im, ax=ax1, shrink=0.6)
x = np.linspace(0, int(start_time_max), int(start_time_max + 1))
pollution = emission_max_value / ((start_time_max / 2) ** 2) * (x - (start_time_max / 2)) ** 2
ax2.plot(x, pollution,label = "emission_curve")
ax2.set_xlabel("Step")
ax2.set_ylabel("Emission Value")
ax2.legend()



# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def my_app(cfg : DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))
# my_app()

plt.show()
