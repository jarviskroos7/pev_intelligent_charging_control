import numpy as np
from env import final_env
import matplotlib.pyplot as plt

def draw(state, policy, draw=True, supress_print=False):

    if not supress_print:
        print(f'session, startSOC={(1-state[0])*100}%, deltaT={state[1]*12/60}hrs, t={(state[2]%120*12/60)}')
        print()
    
    charge_env = final_env()
    state_baseline = state.copy()

    baseline_action = [None] * charge_env.state_size_time
    opt_action = baseline_action.copy()
    baseline_reward = baseline_action.copy()
    opt_reward = baseline_action.copy()

    # step through optimal policy
    done = False
    while not done:
        state_0_index = charge_env.get_index(state[0])
        action_prob = policy[state_0_index, state[1], state[2], :]
        action = np.argmax(action_prob)
        opt_action[state[2]] = action
        new_state, reward, done = charge_env.step(state, action)
        opt_reward[state[2]] = reward
        state = new_state

    # step through baseline
    done = False
    while not done:
        action = 2
        baseline_action[state_baseline[2]] = action
        new_state, reward, done = charge_env.step(state_baseline, action)

        # baseline terminates when deltaSOC --> 0
        if new_state[0] <= 0:
            done = True

        baseline_reward[state[2]] = reward
        state_baseline = new_state

    for idx, action in enumerate(opt_action):
        if action == 0:
            # discharging
            opt_action[idx] = -1
        elif action == 1:
            # do nothing
            opt_action[idx] = 0
        elif action == 2:
            # charging
            opt_action[idx] = 1
        else:
            pass

    baseline_action = [0 if a is None else a for a in baseline_action]
    baseline_action = np.array(baseline_action) / 2

    if draw:
        print(opt_action)

    # make copy for plotting use
    opt_action_none = opt_action.copy()
    baseline_action_none = baseline_action.copy()

    opt_action = [0 if a is None else a for a in opt_action]
    opt_cost = np.array(opt_action) @ np.array(charge_env.price_curve) \
        * charge_env.delta_soc_interval * 80 #kWh
    baseline_cost = np.array(baseline_action) @ np.array(charge_env.price_curve) \
        * charge_env.delta_soc_interval * 80 #kWh

    if not supress_print:
        print("======== Optimal Policy: ========")
        print("Total SOC charged =", sum(opt_action) * charge_env.delta_soc_interval)
        print("Total state value =", sum(filter(None, opt_reward)))
        print("Total energyCharged cost = $", round(opt_cost, 3))

        print()
        print("======= Baseline Policy: ========")
        print("Total energyCharged cost = $", baseline_cost)

        print()
        print("========= Evaluation: ===========")
        print("Charging Cost saved = $", round(opt_cost - baseline_cost, 3), \
            ",", round((opt_cost - baseline_cost) / baseline_cost * 100, 3), "%")

    return opt_cost, baseline_cost, opt_action_none, baseline_action_none
    
def plot_soc_traj(action_list, state, price_curve):

    action_filter = [False if a is None else True for a in action_list]
    action_list = np.array(action_list)[action_filter]
    session_price_curve = np.array(price_curve)[action_filter]

    fig, ax1 = plt.subplots(figsize=(7, 3))
    action_list = np.array([0 if a is None else a for a in action_list]) * 2
    ax1.plot(np.cumsum(action_list) + (1-state[0])*100, label='SOC')
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('SOC [%]')
    tick_pos = np.arange(0, sum(action_filter), 5)
    tick_label = np.append(np.arange(0, 24), np.arange(0, 24))[action_filter[::5]]
    ax1.set_xticks(tick_pos, tick_label)
    ax1.legend(loc=3, bbox_to_anchor=(0, 0.1))

    ax2 = ax1.twinx()
    ax2.plot(session_price_curve, '--', c='orange', label='price_signal')
    ax2.set_ylabel('Cost [$/kWh]')
    ax2.legend(loc=3)

    plt.show()

def main(state, policy):
    return draw(state, policy, draw=True)

if __name__ == "__main__":

    # state = [0.8, 47,0]
    # state = [0.28, 47,0]
    state = [0.4, 80, 90]
    # state = [0.28, 47,0]

    policy = np.load('policy/iter10x120_charging_limit.npy')
    # policy = np.load('iter2x120-highSOC_penalty_small.npy')
    # policy = np.load('policy/iter2x120-highSOC_penalty_small_45-75.npy')
    main(state, policy)