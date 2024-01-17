import matplotlib.pyplot as plt
from TradingEnvironment import TradingEnvironment

STATE_ACTION_PROFIT_PERCENT = 0 # TradingEnvironment.STATE_ACTION_PROFIT_PERCENT
STATE_PERCENT_PROFIT = TradingEnvironment.STATE_PERCENT_PROFIT

ACTION_SELL = TradingEnvironment.ACTION_SELL
ACTION_HOLD = TradingEnvironment.ACTION_HOLD
ACTION_BUY = TradingEnvironment.ACTION_BUY


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

