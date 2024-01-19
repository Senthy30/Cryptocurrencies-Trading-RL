import numpy as np
import gym
from enum import Enum
from Dataset import BitcoinData

class ACTION(Enum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

def get_legal_actions(position):
    if position == ACTION.HOLD:
        return [ACTION.HOLD.value, ACTION.LONG.value, ACTION.SHORT.value]
    else:
        return [ACTION.HOLD.value, ACTION.CLOSE.value]

class TradingEnvironmentV2(gym.Env):
    NUM_OBSERVATIONS = 1000

    def __init__(self, bitcoin_data, num_observations=NUM_OBSERVATIONS, train_data = 0.8, start_time=0, end_time=-1, profit_end=0.8):
        super(TradingEnvironmentV2, self).__init__()

        self.bitcoin_data = bitcoin_data
        self.num_observations = num_observations
        self.train_data = train_data
        self.start_time = start_time
        self.end_time = end_time if end_time != -1 else self.bitcoin_data.length
        self.current_time = self.start_time
        self.profit_end = profit_end

        self.position_open_price = self.current_price()
        self.position = ACTION.HOLD
        self.profit_gain = 0.0
        self.observation_space = self.get_current_state().shape[0]

        self.action_space = gym.spaces.Discrete(n=len(ACTION))
        self.legal_actions = list(range(len(ACTION)))

    def reset(self):
        self.profit = 1.0
        self.min_profit = self.profit
        self.max_profit = self.profit
        self.reward = 0.0
        self.step_profit_gain = 0.0
        self.profit_gain = 0.0

        self.position = ACTION.HOLD
        self.position_open_price = self.current_price()
        self.legal_actions = get_legal_actions(self.position)
        self.action = None

        self.start_time = self.current_time
        self.end_time = self.start_time + self.num_observations
        self.current_time += 1

        return self.get_current_state()

    def step(self, action):
        current_price = self.current_price()
        next_price = self.next_price()

        if action == ACTION.HOLD.value:
            if self.position == ACTION.HOLD or self.position == ACTION.CLOSE:
                self.step_profit_gain = 0.0
            elif self.position == ACTION.LONG:
                self.step_profit_gain = min(self.profit, 1.0) * (next_price / current_price - 1)
            elif self.position == ACTION.SHORT:
                self.step_profit_gain = min(self.profit, 1.0) * (current_price / next_price - 1)
            
            self.reward = self.step_profit_gain
            self.profit_gain += self.step_profit_gain

            if self.position == ACTION.HOLD or self.position == ACTION.CLOSE:
                self.reward = -0.05

        elif action == ACTION.LONG.value:
            wrong_input = False
            if self.position == ACTION.HOLD:
                self.position = ACTION.LONG
                self.position_open_price = current_price
            else:
                wrong_input = True

            self.step_profit_gain = min(self.profit, 1.0) * (next_price / current_price - 1)
            self.reward = self.step_profit_gain if not wrong_input else min(0, self.step_profit_gain)
            self.profit_gain += self.step_profit_gain

        elif action == ACTION.SHORT.value:
            wrong_input = False
            if self.position == ACTION.HOLD:
                self.position = ACTION.SHORT
                self.position_open_price = current_price
            else:
                wrong_input = True

            self.step_profit_gain = min(self.profit, 1.0) * (current_price / next_price - 1)
            self.reward = self.step_profit_gain if not wrong_input else min(0, self.step_profit_gain)
            self.profit_gain += self.step_profit_gain

        elif action == ACTION.CLOSE.value:
            wrong_input = False
            self.reward = 0.0
            if self.position == ACTION.LONG:
                self.position = ACTION.HOLD
                profit_close_gain = min(self.profit, 1.0) * (current_price / self.position_open_price - 1)
                self.reward = min(self.profit, 1.0) * (current_price / next_price - 1)
                self.profit += profit_close_gain

            elif self.position == ACTION.SHORT:
                self.position = ACTION.HOLD
                profit_close_gain = min(self.profit, 1.0) * (self.position_open_price / current_price - 1)
                self.reward = min(self.profit, 1.0) * (next_price / current_price - 1)
                self.profit += profit_close_gain

            else:
                self.position = ACTION.HOLD
                wrong_input = True

            self.step_profit_gain = 0.0
            self.profit_gain += self.step_profit_gain

        self.reward *= 100
        self.legal_actions = get_legal_actions(self.position)

        if self.current_time >= self.end_time:
            if self.position == ACTION.LONG:
                self.position = ACTION.HOLD
                self.profit += min(self.profit, 1.0) * (current_price / self.position_open_price - 1)
            
            elif self.position == ACTION.SHORT:
                self.position = ACTION.HOLD
                self.profit += min(self.profit, 1.0) * (self.position_open_price / current_price - 1)

            self.reward += (self.profit - 1) * 20
            done = True
        else:
            self.current_time += 1
            done = False 

        self.min_profit = min(self.min_profit, self.profit)
        self.max_profit = max(self.max_profit, self.profit)

        return self.get_current_state(), self.reward, done, None

    def current_price(self):
        return self.bitcoin_data.get_price_at(self.current_time)
    
    def next_price(self):
        return self.bitcoin_data.get_price_at(self.current_time + 1)
    
    def get_observation_space(self):
        return self.observation_space
    
    def get_action_space(self):
        return self.action_space.n

    def get_current_state(self):
        return np.array([
            self.current_time,
            self.position_open_price, # added
            self.bitcoin_data.get_price_at(self.current_time),
            self.bitcoin_data.get_open_at(self.current_time),
            self.bitcoin_data.get_high_at(self.current_time),
            self.bitcoin_data.get_low_at(self.current_time),
            self.bitcoin_data.get_close_at(self.current_time),
            self.bitcoin_data.get_volume_btc_at(self.current_time),
            self.bitcoin_data.get_volume_usd_at(self.current_time),
            self.position.value,
            self.profit_gain
        ])
    
class STATE(Enum):
    TIME = 0
    PRICE = 1
    VOLUME_BTC = 2
    VOLUME_USD = 3
    POSITION = 4
    PROFIT_GAIN = 5

class TradingEnvironment():

    NUM_OBSERVATIONS = 8500
    #NUM_OBSERVATIONS = 100

    # ACTIONS
    ACTION_SELL = -1
    ACTION_HOLD = 0
    ACTION_BUY = 1

    MULTIPLIER_HOLD = 90
    MULTIPLIER_BUY_SELL = 100

    # REWARDS FUNCTIONS
    REWARD_PROFIT = 0
    REWARD_SHARPES_RATIO = 1

    # TERMINATION CONDITIONS
    TERMINATION_PROFIT_LOSS = 0
    TERMINATION_ENDED_EPOCH = 1

    # STATES
    STATE_TIME = 0
    STATE_LAST_ACTION_TIME = 1
    STATE_OPEN = 2
    STATE_HIGH = 3
    STATE_LOW = 4
    STATE_CLOSE = 5
    STATE_VOLUME = 6
    STATE_QUOTE_ASSET_VOLUME = 7
    STATE_PRICE = 8
    STATE_ACTION_PROFIT_PERCENT = 9
    STATE_WALLET_ACTION = 10
    STATE_PERCENT_PROFIT = 11

    def __init__(
                    self, bitcoin_data, num_observations=NUM_OBSERVATIONS, parcent_train=0.8,
                    start_time=0, end_time=-1, 
                    stop_loss=0.05, take_profit=0.12, 
                    percent_termination = 0.85, holding_penalty = 0.02, allowed_holding_steps = 0,
                    holding_action_penalty = 0.02, allowed_holding_action_steps = 25,
                    same_action_penalty = 0.02, reward_open_action = 0.0, reward_transaction = 0.2,
                    REWARD_FUNCTION=REWARD_PROFIT
                ):
        self.bitcoin_data = bitcoin_data

        self.percent_train = parcent_train
        self.length_train = int(self.bitcoin_data.length * self.percent_train)

        self.num_observations = num_observations
        self.start_time = start_time
        self.end_time = end_time if end_time != -1 else self.bitcoin_data.length
        self.current_time = self.start_time

        self.wallet = (self.ACTION_HOLD, 0)
        self.min_percent_profit = 1.0
        self.percent_profit = 1.0
        self.max_percent_profit = 1.0
        self.percent_termination = percent_termination
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.last_time_action = self.current_time
        self.last_buy_sell_time_action = self.current_time

        self.holding_action_penalty = holding_action_penalty
        self.holding_penalty = holding_penalty
        self.same_action_penalty = same_action_penalty
        self.reward_open_action = reward_open_action
        self.reward_transaction = reward_transaction

        self.allowed_holding_action_steps = allowed_holding_action_steps
        self.allowed_holding_steps = allowed_holding_steps

        self.observation_space = self.get_current_state()

        if REWARD_FUNCTION == self.REWARD_PROFIT:
            self.reward_function = self.profit_reward
        elif REWARD_FUNCTION == self.REWARD_SHARPES_RATIO:
            self.reward_function = self.sharpes_ratio_reward

    def reset(self):
        self.start_time = self.current_time
        self.end_time = self.start_time + self.num_observations

        self.min_percent_profit = 1.0
        self.percent_profit = 1.0
        self.max_percent_profit = 1.0

        self.last_buy_sell_time_action = self.current_time
        self.last_time_action = self.current_time
        self.wallet = (self.ACTION_HOLD, 0)
        self.current_time += 1

        return self.get_current_state()

    def step(self, action):
        if self.current_time >= self.length_train:
            self.current_time = 0
            return self.reset(), self.get_termination_reward(), True, None
        
        if self.current_time >= self.end_time:
            if self.wallet[0] == self.ACTION_SELL:
                _, self.wallet = self.update_wallet(self.ACTION_BUY)
            elif self.wallet[0] == self.ACTION_BUY:
                _, self.wallet = self.update_wallet(self.ACTION_SELL)

            return self.get_current_state(), self.get_termination_reward(), True, None

        if self.wallet[0] != self.ACTION_HOLD or action != self.ACTION_HOLD:
            self.last_buy_sell_time_action = self.current_time
        if self.wallet[0] != self.ACTION_HOLD and action != self.ACTION_HOLD and action != self.wallet[0]:
            self.last_time_action = self.current_time
        if self.wallet[0] == self.ACTION_HOLD and action != self.ACTION_HOLD:
            self.last_time_action = self.current_time

        reward = self.get_reward(action)
        allowed, self.wallet = self.update_wallet(action)

        self.current_time += 1

        done = self.check_termination()
        if done:
            reward -= 6

        if not allowed:
            reward -= self.same_action_penalty

        if self.check_predefined_stop_action():
            self.update_percent_profit(self.percent_profit + self.get_partial_profit_percent())
            self.wallet = (self.ACTION_HOLD, 0)

        return self.get_current_state(), reward, done, None

    def profit_reward(self):
        profit = self.get_profit()
        partial_percent_profit = self.get_partial_profit_percent()

        if profit > 0:
            return 1.5 + partial_percent_profit * 15
        elif profit < 0:
            return -1.5 + partial_percent_profit * 15
        else:
            return 0

    def sharpes_ratio_reward(self):
        print("SHARPES RATIO REWARD")
        pass

    def keep_holding_penalty(self):
        penalty = self.holding_penalty * ((self.current_time - self.allowed_holding_steps) - self.last_buy_sell_time_action)

        return -max(0, penalty)
    
    def keep_holding_action_penalty(self):
        penalty = self.holding_action_penalty * ((self.current_time - self.allowed_holding_action_steps) - self.last_time_action)

        return -max(0, penalty)

    def check_termination(self):
        if self.percent_profit <= self.percent_termination:
            return True
        return False
    
    def check_predefined_stop_action(self):
        if self.get_partial_profit_percent() >= self.take_profit:
            return True
        
        if self.get_partial_profit_percent() <= -self.stop_loss:
            return True
        
        return False
    
    def get_reward(self, action):
        reward = self.keep_holding_penalty() + self.keep_holding_action_penalty() + self.get_transaction_reward(action)
        
        """
        if action == self.ACTION_BUY and self.wallet[0] == self.ACTION_SELL:
            reward += self.reward_function()
        elif action == self.ACTION_SELL and self.wallet[0] == self.ACTION_BUY:
            reward += self.reward_function()
        elif action != self.ACTION_HOLD and self.wallet[0] == self.ACTION_HOLD:
            reward += self.reward_open_action
        """
        
        return reward
    
    def get_termination_reward(self):
        x_profit_percent = self.percent_profit - 1
        reward = x_profit_percent * 30
        if x_profit_percent > 0:
            reward += 1.5
        elif x_profit_percent < 0:
            reward -= 1.5

        return reward

    def get_transaction_reward(self, action):
        reward = 0
        current_percent_profit = self.percent_profit + self.get_partial_profit_percent()
        
        if self.wallet[0] == self.ACTION_HOLD:
            if action != self.ACTION_HOLD:
                next_profit_value = (self.next_price() - self.current_price()) * action
                next_percent_profit = current_percent_profit + next_profit_value / self.current_price()

                reward = (next_percent_profit - current_percent_profit) * self.MULTIPLIER_HOLD

        else:
            next_profit_value = (self.next_price() - self.current_price()) * self.wallet[0]
            next_percent_profit = current_percent_profit + next_profit_value / self.current_price()

            if self.wallet[0] == self.ACTION_BUY:
                if action == self.ACTION_HOLD or action == self.ACTION_BUY:
                    reward = (next_percent_profit - current_percent_profit) * self.MULTIPLIER_HOLD
                elif action == self.ACTION_SELL:
                    reward = self.get_partial_profit_percent() * self.MULTIPLIER_BUY_SELL

            elif self.wallet[0] == self.ACTION_SELL:
                if action == self.ACTION_HOLD or action == self.ACTION_SELL:
                    reward = (next_percent_profit - current_percent_profit) * self.MULTIPLIER_HOLD
                elif action == self.ACTION_BUY:
                    reward = self.get_partial_profit_percent() * self.MULTIPLIER_BUY_SELL
 
        return reward

        """
        current_price = self.current_price()
        next_price = self.next_price()
        
        if self.wallet[0] == self.ACTION_HOLD:
            next_profit = (next_price - current_price) * action
        else:
            next_profit = (next_price - current_price) * self.wallet[0]

        if next_profit > 0:
            return self.reward_transaction
        elif next_profit < 0:
            return -self.reward_transaction
        return 0
        """

    def update_wallet(self, action):
        if action == self.ACTION_BUY:
            if self.wallet[0] == self.ACTION_SELL:
                self.update_percent_profit(self.percent_profit + self.get_partial_profit_percent())
                return True, (self.ACTION_HOLD, 0)
            elif self.wallet[0] == self.ACTION_HOLD:
                return True, (self.ACTION_BUY, self.current_price())
            
            return False, self.wallet
        elif action == self.ACTION_SELL:
            if self.wallet[0] == self.ACTION_BUY:
                self.update_percent_profit(self.percent_profit + self.get_partial_profit_percent())
                return True, (self.ACTION_HOLD, 0)
            elif self.wallet[0] == self.ACTION_HOLD:
                return True, (self.ACTION_SELL, self.current_price())
            
            return False, self.wallet
            
        return True, self.wallet
    
    def update_percent_profit(self, value):
        self.percent_profit = value

        if self.percent_profit < self.min_percent_profit:
            self.min_percent_profit = self.percent_profit
        if self.percent_profit > self.max_percent_profit:
            self.max_percent_profit = self.percent_profit

    def current_price(self):
        return self.bitcoin_data.get_price_at(self.current_time)
    
    def next_price(self):
        return self.bitcoin_data.get_price_at(self.current_time + 1)

    def get_action_profit_percent(self):
        return self.get_partial_profit_percent() * self.percent_profit

    def get_profit(self):
        action = self.wallet[0]
        price = self.wallet[1]
        profit = action * (self.current_price() - price)

        return profit
    
    def get_partial_profit_percent(self):
        if self.wallet[0] == self.ACTION_HOLD:
            return 0

        action = self.wallet[0]
        price = self.wallet[1]
        profit = action * (self.current_price() - price)

        return profit / price
    
    def get_current_state(self):
        """
        return np.array([
            self.current_time,
            self.last_time_action,
            self.bitcoin_data.get_open_at(self.current_time),
            self.bitcoin_data.get_high_at(self.current_time),
            self.bitcoin_data.get_low_at(self.current_time),
            self.bitcoin_data.get_close_at(self.current_time),
            self.bitcoin_data.get_volume_btc_at(self.current_time),
            self.bitcoin_data.get_volume_usd_at(self.current_time),
            self.bitcoin_data.get_price_at(self.current_time),
            self.wallet[1],
            self.wallet[0],
            self.percent_profit
        ])
        """

        return np.array([
            self.current_time,
            self.allowed_holding_action_steps - (self.current_time - self.last_time_action),
            self.bitcoin_data.get_price_at(self.current_time),
            self.bitcoin_data.get_volume_btc_at(self.current_time),
            self.bitcoin_data.get_volume_usd_at(self.current_time),
            self.wallet[0],
            self.wallet[1],
            self.percent_profit + self.get_partial_profit_percent()
        ])
    
    def get_action_space(self):
        return np.array([
            self.ACTION_SELL, 
            self.ACTION_HOLD, 
            self.ACTION_BUY
        ]).shape[0]
    
    def get_observation_space(self):
        return self.observation_space.shape[0]