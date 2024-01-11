import numpy as np
from Dataset import BitcoinData

class TradingEnvironment():

    NUM_OBSERVATIONS = 25000

    # ACTIONS
    ACTION_SELL = -1
    ACTION_HOLD = 0
    ACTION_BUY = 1

    # REWARDS FUNCTIONS
    REWARD_PROFIT = 0
    REWARD_SHARPES_RATIO = 1

    # TERMINATION CONDITIONS
    TERMINATION_PROFIT_LOSS = 0
    TERMINATION_ENDED_EPOCH = 1

    def __init__(
                    self, bitcoin_data, num_observations=NUM_OBSERVATIONS,
                    start_time=0, end_time=-1, 
                    stop_loss=0.05, take_profit=0.12, 
                    percent_termination = 0.4, holding_penalty = 0.01, allowed_holding_steps = 10,
                    same_action_penalty = 0.05,
                    REWARD_FUNCTION=REWARD_PROFIT
                ):
        self.bitcoin_data = bitcoin_data

        self.num_observations = num_observations
        self.start_time = start_time
        self.end_time = end_time if end_time != -1 else self.bitcoin_data.length
        self.current_time = self.start_time

        self.wallet = (self.ACTION_HOLD, 0)
        self.min_percent_profit = 1
        self.percent_profit = 1
        self.max_percent_profit = 1
        self.percent_termination = percent_termination
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.last_buy_sell_time_action = 0
        self.holding_penalty = holding_penalty
        self.same_action_penalty = same_action_penalty
        self.allowed_holding_steps = allowed_holding_steps

        self.observation_space = self.get_current_state()

        if REWARD_FUNCTION == self.REWARD_PROFIT:
            self.reward_function = self.profit_reward
        elif REWARD_FUNCTION == self.REWARD_SHARPES_RATIO:
            self.reward_function = self.sharpes_ratio_reward

    def reset(self):
        self.start_time = self.current_time
        self.end_time = self.start_time + self.num_observations

        self.min_percent_profit = 1
        self.percent_profit = 1
        self.max_percent_profit = 1

        self.last_buy_sell_time_action = self.current_time
        self.wallet = (self.ACTION_HOLD, 0)

        return self.get_current_state()

    def step(self, action):
        if self.current_time >= self.end_time:
            return None, None, True, self.TERMINATION_ENDED_EPOCH

        allowed, self.wallet = self.update_wallet(action)
        if self.wallet[0] != self.ACTION_HOLD:
            self.last_buy_sell_time_action = self.current_time

        self.current_time += 1

        reward = self.reward_function() + self.keep_holding_penalty()
        done = self.check_termination()

        if not allowed:
            reward -= self.same_action_penalty

        if self.check_predefined_stop_action():
            self.update_percent_profit(self.percent_profit + self.get_partial_profit_percent())
            self.wallet = (self.ACTION_HOLD, 0)

        return self.get_current_state(), reward, done, None

    def profit_reward(self):
        profit = self.get_profit()

        if profit > 0:
            return 1
        elif profit < 0:
            return -1
        else:
            return 0

    def sharpes_ratio_reward(self):
        print("SHARPES RATIO REWARD")
        pass

    def keep_holding_penalty(self):
        penalty = self.holding_penalty * ((self.current_time - self.allowed_holding_steps) - self.last_buy_sell_time_action)

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
        return np.array([
            self.current_time,
            self.bitcoin_data.get_open_at(self.current_time),
            self.bitcoin_data.get_high_at(self.current_time),
            self.bitcoin_data.get_low_at(self.current_time),
            self.bitcoin_data.get_close_at(self.current_time),
            self.bitcoin_data.get_volume_at(self.current_time),
            self.bitcoin_data.get_quote_asset_volume_at(self.current_time),
            self.bitcoin_data.get_price_at(self.current_time),
            self.get_partial_profit_percent(),
            self.percent_profit
        ])
    
    def get_action_space(self):
        return np.array([
            self.ACTION_SELL, 
            self.ACTION_HOLD, 
            self.ACTION_BUY
        ]).shape[0]
    
    def get_observation_space(self):
        return self.observation_space.shape[0]
    