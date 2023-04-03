import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from simple_charge_env import Simple_charge_env
from simple_charge_env import max_current,current_interval, step, start_time_max, step
import dill as pickle
import glob
import os
from pathlib import Path

env = Simple_charge_env()
N_actions = env.action_space.n
N_states = env.observation_space.shape[0]
output_dir = "predict_output"

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_states, 32)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.fc2 = nn.Linear(32, 64)
#         self.fc2.weight.data.normal_(0, 0.1)
#         self.fc3 = nn.Linear(64, 128)
#         self.fc3.weight.data.normal_(0, 0.1)
#         self.out = nn.Linear(128, N_actions)
#         self.out.weight.data.normal_(0, 0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_states, 32)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.fc2 = nn.Linear(32, 64)
#         self.fc2.weight.data.normal_(0, 0.1)
#         self.fc3 = nn.Linear(64, 128)
#         self.fc3.weight.data.normal_(0, 0.1)
#         self.fc4 = nn.Linear(128, 128)
#         self.fc4.weight.data.normal_(0, 0.1)
#         self.out = nn.Linear(128, N_actions)
#         self.out.weight.data.normal_(0, 0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, 128)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(128, 256)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(256, 256)
        self.fc5.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, N_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value




predict_net = Net()
predict_net.load_state_dict(torch.load("./trained_model.pt"))
my_file = Path("./trained_model.pt")
if my_file.is_file():
    predict_net.load_state_dict(torch.load("./trained_model.pt"))
    print("successfully loaded")

def count_pkl_file_number(dir_path=output_dir):
    pkl_files = glob.glob(os.path.join(dir_path, '*.pkl'))
    num_pkl_files = len(pkl_files)
    return num_pkl_files


# def get_reward(time, a, I_max, emission_max_value):
#     time = time % start_time_max
#     x = np.linspace(0, int(start_time_max), int(start_time_max+1))
#     y = emission_max_value/((start_time_max/2)**2) * (x-(start_time_max/2))**2
#     max_y = y[0]
#     current_list = np.linspace(0, max_current, int(max_current/current_interval) + 1)
#
#     current = min(I_max, current_list[a])
#     reward = (max_y - y[int(time)])/max_y * current * step/60
#     print("_____________")
#     print("reward:", reward)
#     print("time:", time)
#     print("emission:", y[int(time)])
#     print("max_y:", max_y)
#     print("current:", current)
#     print("_____________")
#     return reward


for i_episode in range(1):
    # s = env.reset()
    # s = env.reset_with_values(0.2159713063120908,
    #                           0.6871365930818221,
    #                           143,
    #                           287)
    # s = env.reset_with_values(0.3,
    #                           0.9,
    #                           143,
    #                           270)

    s = env.reset_with_values(0.22,
                              0.7,
                              0,
                              144)
    # s = env.reset_with_values(0.2159713063120908,
    #                           0.6871365930818221,
    #                           143,
    #                           287)
    current_soc, target_soc, current_time, end_time,I_max = s
    start_soc = current_soc
    start_time = current_time
    print(i_episode)
    action_history = []
    ep_r = 0
    while True:
        # env.render(mode = "human")
        x = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))
        action_value = predict_net.forward(x)
        print(action_value)
        action = torch.max(action_value, 1)[1].data.numpy()
        print(action)
        action = action[0]
        action_history.append(action)
        # a = dqn.choose_action(s)

        # take action
        s_, r, done, tru, info = env.step(action)
        # modify the reward
        current_soc, target_soc, current_time, end_time, I_max = s_
        # x, x_dot, theta, theta_dat = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = get_reward(current_time, action, I_max, 100)
        # r = r1 + r2

        # dqn.store_transition(s, a, r, s_)
        ep_r += r
        # if dqn.memory_counter > Memory_capacity:
        #     dqn.learn()
        #     if done:
        #         print('Ep: ', i_episode,
        #               '| Ep_r: ', round(ep_r, 2))
        # print("start_soc: ", start_soc)
        # print("current_soc: ", current_soc)
        # print("target_soc: ", target_soc)
        # print("start_time: ", start_time)
        # print("current_time: ", current_time)
        # print("end_time: ", end_time)
        print("current_soc: ", current_soc)
        if done:
            print("ep_r:", ep_r)
            # print("start_soc: ", start_soc)
            # print("current_soc: ", current_soc)
            # print("target_soc: ", target_soc)
            # print("start_time: ", start_time)
            # print("current_time: ", current_time)
            # print("end_time: ", end_time)
            my_object = {"start_soc":start_soc,
                        "current_soc": current_soc,
                        "target_soc": target_soc,
                        "start_time":start_time,
                        "current_time":current_time,
                        "end_time":end_time,
                         "action_history":action_history}

            num_pkl_files = count_pkl_file_number()
            pickle_name = "predict_%s.pkl" %num_pkl_files
            file_path = os.path.join(os.path.dirname(__file__),output_dir , pickle_name)
            with open(file_path, 'wb') as f:
                pickle.dump(my_object, f)
            break
        s = s_
