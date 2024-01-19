import tensorflow as tf
from tensorflow import keras
import numpy as np
from Models.config import ConfigModel
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LeakyReLU, ReLU
from keras.optimizers import Adam
from Models.config import Memory
import tensorflow as tf
from Models.config import ConfigModel
import random
import os


class DDDQN(tf.keras.Model):
    def __init__(self, action_space, observation_space):
      super(DDDQN, self).__init__()
      self.d1 = tf.keras.layers.Dense(512, input_shape=(observation_space, ), activation=LeakyReLU())
      self.d2 = tf.keras.layers.Dense(512, activation=LeakyReLU())
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(action_space, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a

class DuelingDoubleDQN(ConfigModel):
    def __init__(self, action_space, observation_space, model_path="", load_version=-1, load_model_num=-1, gamma=0.98, replace=100, lr=0.001):
        super().__init__(observation_space=observation_space, action_space=action_space, model_path=model_path, load_version=load_version, load_model_num=load_model_num)

        self.gamma = gamma
        self.action_space = action_space
        self.epsilon = 1.0
        self.min_epsilon = 0.03
        self.epsilon_decay = 0.999
        self.replace = replace
        self.trainstep = 0
        self.memory = Memory(1000000, observation_space=observation_space)
        self.batch_size = 64
        self.actions_taked_by_itself = [0, 0, 0]

        if load_version == -1:
            self.q_net = DDDQN(action_space=action_space, observation_space=observation_space)
        else:
            if self.model_num_save == self.model_latest_save:
                model = load_model(os.path.join(self.model_path, f"Model {self.model_num_save}.keras"))
            else:
                model = load_model(os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME, f"Model {self.model_num_save}.keras"))

            self.memory = Memory(self.MEMORY_SIZE, load_path=os.path.join(self.model_path, self.MEMORY_FILENAME), observation_space=self.observation_space)
            
            self.q_net = model
        
        self.target_net = DDDQN(action_space, observation_space)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
       

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            self.actions_taked_by_itself[action] += 1
            return action

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self, epochs=15):
        if self.memory.current_index < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target()

        states, actions, rewards, next_states, dones = self.memory.get_batch(self.batch_size)
        actions = actions.astype(int)

        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1,)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = np.copy(target)

        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones

        # am adaugat iful
        if self.trainstep % 2 == 0:
            self.q_net.fit(states, q_target, batch_size=self.batch_size, epochs=epochs, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        self.trainstep += 1

