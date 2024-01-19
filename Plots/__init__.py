import matplotlib.pyplot as plt
from TradingEnvironment import TradingEnvironment, ACTION, STATE

"""
STATE_ACTION_PROFIT_PERCENT = 0 # TradingEnvironment.STATE_ACTION_PROFIT_PERCENT
STATE_PERCENT_PROFIT = TradingEnvironment.STATE_PERCENT_PROFIT

ACTION_SELL = TradingEnvironment.ACTION_SELL
ACTION_HOLD = TradingEnvironment.ACTION_HOLD
ACTION_BUY = TradingEnvironment.ACTION_BUY
"""

def plot_value_action(states, dones, start_time=0, end_time=-1):
    value = []
    action = []
    
    i = start_time
    while i < len(states):
        if end_time != -1 and i >= end_time:
            break
        if dones[i]:
            break

        value.append(states[i][2])
        if states[i][9] == ACTION.HOLD.value:
            action.append(states[i][2])
        elif states[i][9] == ACTION.LONG.value:
            action.append(states[i][2] + 5)
        elif states[i][9] == ACTION.SHORT.value:
            action.append(states[i][2] - 5)

        i += 1
    
    plt.plot(value, label="Value")
    plt.plot(action, label="Prediction")
    plt.legend()

def plot_value_action_v1(states, dones, start_time=0, end_time=-1):
    value = []
    action = []
    
    i = start_time
    while i < len(states):
        if end_time != -1 and i >= end_time:
            break
        if dones[i]:
            break

        value.append(states[i][2])
        if states[i][5] == ACTION.HOLD.value:
            action.append(states[i][2])
        elif states[i][5] == ACTION.LONG.value:
            action.append(states[i][2] + 5)
        elif states[i][5] == ACTION.SHORT.value:
            action.append(states[i][2] - 5)

        i += 1

    plt.plot(value, label="Value")
    plt.plot(action, label="Prediction")
    plt.legend()

def plot_action_profit(states, dones, start_time=0):
    profit = []
    position = []

    i = start_time
    while i < len(states):
        if dones[i]:
            break

        profit.append(1 + states[i][9])

        agent_position = states[i][8]
        if agent_position == ACTION.HOLD.value:
            position.append(1)
        elif agent_position == ACTION.LONG.value:
            position.append(1.1)
        elif agent_position == ACTION.SHORT.value:
            position.append(0.9)
        else:
            position.append(1)

        i += 1

    plt.plot(profit, label="Profit")
    plt.plot(position, label="Position")
    plt.legend()

def plot_action_profit_percent(states, action, reward, dones, start_time=0, end_time=-1):
    percent_profit = []
    action_profit_percent = []
    actions = []
    rewards = []

    if end_time == -1:
        end_time = len(states)

    i = start_time
    last_action = ACTION_HOLD
    min_val = 2e9
    while i < end_time:
        if dones[i]:
            percent_profit.append(0)
            action_profit_percent.append(0)
            actions.append(ACTION_HOLD)
            rewards.append(0)

            last_action = ACTION_HOLD
            i += 1

            break
        
        current_action = action[i] - 1
        current_action_state = last_action
        if last_action == ACTION_HOLD:
            current_action_state = current_action
        elif last_action == ACTION_BUY and current_action == ACTION_SELL:
            current_action_state = ACTION_HOLD
        elif last_action == ACTION_SELL and current_action == ACTION_BUY:
            current_action_state = ACTION_HOLD

        percent_profit.append(states[i][-1])
        action_profit_percent.append(states[i][STATE_ACTION_PROFIT_PERCENT])
        actions.append(action[i] - 1)
        rewards.append(reward[i])

        min_val = min(min_val, states[i][STATE_ACTION_PROFIT_PERCENT])
        last_action = current_action_state
        i += 1

    for i in range(len(action_profit_percent)):
        action_profit_percent[i] = 0
        rewards[i] = max(-1, rewards[i])

    plt.plot(percent_profit, label="Percent Profit")
    # plt.plot(action_profit_percent, label="Action Profit Percent")
    plt.plot(actions, label="Actions")
    plt.plot(rewards, label="Rewards")
    plt.legend()
    plt.show(True)

