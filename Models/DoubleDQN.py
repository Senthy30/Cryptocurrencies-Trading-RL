import random
import os
import numpy as np
import time

from .config import ConfigModel
from .Memory import Memory
from TradingEnvironment import STATE, get_legal_actions
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LeakyReLU, ReLU
from keras.optimizers import Adam

class DQNV2(ConfigModel):

    UPDATE_TARGET_MODEL_EVERY = 1000

    def __init__(self, environment, observation_space, action_space, model_path="", load_version=-1, load_model_num=-1):
        super().__init__(environment=environment, observation_space=observation_space, action_space=action_space, model_path=model_path, load_version=load_version, load_model_num=load_model_num)

        if load_version == -1:
            self.model = self._build_model()
        else:
            self.model = self._load_model()
        
        self.model_target = clone_model(self.model)

    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(512, input_shape=(self.observation_space,), activation=ReLU()))
        model.add(Dense(256, activation=ReLU()))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))

        return model
    
    def _load_model(self):
        if self.model_num_save == self.model_latest_save:
            model = load_model(os.path.join(self.model_path, f"Model {self.model_num_save}.keras"))
        else:
            model = load_model(os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME, f"Model {self.model_num_save}.keras"))

        self.memory = Memory(self.MEMORY_SIZE, load_path=os.path.join(self.model_path, self.MEMORY_FILENAME), observation_space=self.observation_space)
        
        return model

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            idx = random.randrange(0, len(self.environment.legal_actions))
            return self.environment.legal_actions[idx]
        return self.act_greedy(state)
    
    def act_greedy(self, state):
        legal_actions = self.environment.legal_actions
        mask_legal_actions = np.zeros(self.action_space)
        mask_legal_actions[legal_actions] = 1
        mask_legal_actions = mask_legal_actions.astype(int)

        q_values = self.model.predict(state)
        q_values[0][mask_legal_actions == 0] = -99999999
        action = np.argmax(q_values[0])

        self.actions_taked_by_itself[action] += 1

        return action
    
    def learn(self):
        self.learns += 1
        if self.memory.current_index < self.BATCH_SIZE:
            return
        
        if self.learns % self.LEARN_EVERY != 0:
            return
        
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.BATCH_SIZE)

        labels = self.model.predict(states)
        next_state_values = self.model_target.predict(next_states)

        for idx in range(self.BATCH_SIZE):
            legal_actions = get_legal_actions(next_states[idx][STATE.POSITION.value])
            mask_legal_actions = np.zeros(self.action_space)
            mask_legal_actions[legal_actions] = 1
            mask_legal_actions = mask_legal_actions.astype(int)

            next_state_values[idx][mask_legal_actions == 0] = -99999999

        for i in range(self.BATCH_SIZE):
            labels[i][int(actions[i])] = rewards[i] + self.GAMMA * np.max(next_state_values[i]) * (1 - dones[i])

        self.model.fit(states, labels, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=0)

        if self.exploration_rate > self.EXPLORATION_MIN:
            self.exploration_rate *= self.EXPLORATION_DECAY
            self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        
        if self.learns % self.UPDATE_TARGET_MODEL_EVERY == 0:
            self.model_target.set_weights(self.model.get_weights())

class DQNV1(ConfigModel):

    UPDATE_TARGET_MODEL_EVERY = 1000

    def __init__(self, environment, observation_space, action_space, model_path="", load_version=-1, load_model_num=-1):
        super().__init__(environment=environment, observation_space=observation_space, action_space=action_space, model_path=model_path, load_version=load_version, load_model_num=load_model_num)

        if load_version == -1:
            self.model = self._build_model()
        else:
            self.model = self._load_model()
        
        self.model_target = clone_model(self.model)

    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(512, input_shape=(self.observation_space,), activation=ReLU()))
        model.add(Dense(256, activation=ReLU()))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))

        return model
    
    def _load_model(self):
        if self.model_num_save == self.model_latest_save:
            model = load_model(os.path.join(self.model_path, f"Model {self.model_num_save}.keras"))
        else:
            model = load_model(os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME, f"Model {self.model_num_save}.keras"))

        self.memory = Memory(self.MEMORY_SIZE, load_path=os.path.join(self.model_path, self.MEMORY_FILENAME), observation_space=self.observation_space)
        
        return model
    
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(0, self.action_space)
        return self.act_greedy(state)

    def act_greedy(self, state):
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])

        self.actions_taked_by_itself[action] += 1

        return action

    def learn(self):
        self.learns += 1
        if self.memory.current_index < self.BATCH_SIZE:
            return
        
        if self.learns % self.LEARN_EVERY != 0:
            return
        
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.BATCH_SIZE)

        labels = self.model.predict(states)
        next_state_values = self.model_target.predict(next_states)

        for i in range(self.BATCH_SIZE):
            labels[i][int(actions[i])] = rewards[i] + self.GAMMA * np.max(next_state_values[i]) * (1 - dones[i])

        self.model.fit(states, labels, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=0)

        if self.exploration_rate > self.EXPLORATION_MIN:
            self.exploration_rate *= self.EXPLORATION_DECAY
            self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        
        if self.learns % self.UPDATE_TARGET_MODEL_EVERY == 0:
            self.model_target.set_weights(self.model.get_weights())
