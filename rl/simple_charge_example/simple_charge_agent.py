import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym
import hydra
from omegaconf import DictConfig, OmegaConf
from  simple_charge_env import Simple_charge_env
from simple_charge_env import max_current,current_interval,  start_time_max, step,battery_volume
from pathlib import Path
import random

# parameters
Batch_size = 64
Lr = 0.00001
Epsilon = 0.995 # greedy policy
Gamma = 1  # reward discount
Target_replace_iter = 50   # target update frequency
Memory_capacity = 3000
episode_num = 50000
epsilion_increase_value = (1-Epsilon)/episode_num
emission_max_value = 1


# @hydra.main(config_path="conf", config_name="env")
# def generate_env(cfg):
#     global env
#     env = Simple_charge_env(**cfg)
env = Simple_charge_env()
# generate_env()

# env = gym.make('CartPole-v0')
# env = env.unwrapped
N_actions = env.action_space.n
N_states = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

voltage = 400
resistance = 0

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


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        my_file = Path("./trained_model.pt")
        if my_file.is_file():
            self.eval_net.load_state_dict(torch.load("./trained_model.pt"))
            self.target_net.load_state_dict(torch.load("./trained_model.pt"))
            print("successfully loaded")
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, cur_episode_num=0):
        # print("x:", x[0])
        # print(len(x[0]))
        if type(x) == tuple:
            x = x[0]

        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < Epsilon + epsilion_increase_value * cur_episode_num:
            action_value = self.eval_net.forward(x)
            # print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()
            # print(action)
            action = action[0]
        else:
            action = np.random.randint(0, N_actions)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        if type(s) == tuple:
            s = s[0]
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))


        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



x = np.linspace(0, int(start_time_max)-1, int(start_time_max))
y = emission_max_value/((start_time_max/2)**2) * (x-(start_time_max/2))**2

def get_reward(time, a, I_max, emission_max_value):
    time = time % start_time_max
    # x = np.linspace(0, int(start_time_max), int(start_time_max+1))
    # y = emission_max_value/((start_time_max/2)**2) * (x-(start_time_max/2))**2
    # max_y = y[0]
    current_list = np.linspace(0, max_current, int(max_current/current_interval) + 1)

    current = min(I_max, current_list[a])
    # reward = (max_y - y[int(time)])/max_y * current * step / 60
    reward = -y[int(time)] * (current + 2*current*resistance/voltage) * step / 60
    # reward = -y[int(time)] * current * step / 60
    # reward = -(max_y - y[int(time)])/max_y * current
    # print("_____________")
    # print("reward:", reward)
    # print("time:", time)
    # print("emission:", y[int(time)])
    # print("max_y:", max_y)
    # print("current:", current)
    # print("_____________")
    return reward



def run_experiment(save_model= True):

    possible_starting_state = [(0.2159713063120908,
                                0.6871365930818221,
                                0,
                                144),
                               (0.2159713063120908,
                                0.6871365930818221,
                                90,
                                254),
                               (0.6,
                                0.9,
                                90,
                                180),
                               (0.2,
                                0.6,
                                50,
                                90),
                               (0.2,
                                0.7,
                                50,
                                70),
                               (0.32,
                                0.5,
                                32,
                                176
                               ),
                               (0.12,
                                  0.49,
                                  142,
                                  291),
                               (0.2,
                                0.81,
                                91,
                                238),
                               ]
    dqn = DQN()
    print('\nCollecting experience...')
    current_history_lists = []
    for i_episode in range(episode_num):
        random_s = random.choice(possible_starting_state)
        current_history_list = []
        # s = env.reset()
        s = env.reset_with_values(*random_s)
        # s = env.reset_with_values(0.2159713063120908,
        #                           0.6871365930818221,
        #                           143,
        #                           287)
        # s = env.reset_with_values(0.3,
        #                           0.9,
        #                           144,
        #                           270)
        # s = env.reset_with_values(0.2159713063120908,
        #                           0.6871365930818221,
        #                           144,
        #                           287)
        # s = env.reset_with_values(0.22,
        #                           0.7,
        #                           0,
        #                           144)
        # s = env.reset_with_values(0.2159713063120908,
        #                           0.6871365930818221,
        #                           143,
        #                           287)
        # current_soc, target_soc, start_time, end_time, current_time, current_power_limit, I_max = s
        current_soc, target_soc, current_time, end_time, I_max = s
        start_soc = current_soc
        ep_r = 0
        while True:
            # env.render(mode = "human")
            a = dqn.choose_action(s, i_episode)
            current_history_list.append(a)
            # take action
            s_, r, done, tru, info = env.step(a)
            # modify the reward
            current_soc, target_soc, current_time, end_time, I_max = s_
            # current_soc, target_soc, start_time, end_time, current_time, current_power_limit, I_max = s_
            # x, x_dot, theta, theta_dat = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = get_reward(current_time, a, I_max, emission_max_value)
            # if done:
            #     if current_soc < target_soc:
            #         r += abs(target_soc - current_soc) * emission_max_value * battery_volume * -1
            if done:
                if current_soc < target_soc:
                    r += -abs(target_soc - current_soc) * emission_max_value * battery_volume *(1 + max_current * resistance/voltage)

            # r = r1 + r2

            dqn.store_transition(s, a, r, s_)
            ep_r += r
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                if done and i_episode % 500 ==0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))
            if done and i_episode % 500 ==0:
                print("start_soc: ", start_soc)
                print("current_soc: ", current_soc)
                print("target_soc: ", target_soc)
                # print("start_time: ", start_time)
                print("current_time: ", current_time)
                print("end_time: ", end_time)
                print("current_history_list:",current_history_list)
                break
            if done:
                break
            s = s_
        if i_episode % 5000 == 0:
            torch.save(dqn.target_net.state_dict(), "./trained_model.pt")
            print("save successfully")
    if save_model:
        torch.save(dqn.target_net.state_dict(), "./trained_model.pt")




run_experiment()


# @hydra.main(config_path="conf", config_name="env")
# def run(cfg):
#     run_experiment()
# env.close()

