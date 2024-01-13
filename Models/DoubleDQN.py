import random
import os
import numpy as np
from .config import ConfigModel
from .Memory import Memory
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam

class DoubleDQN(ConfigModel):

    UPDATE_TARGET_MODEL_EVERY = 20

    def __init__(self, observation_space, action_space, model_path="", load_version=-1, load_model_num=-1):
        super().__init__(observation_space=observation_space, action_space=action_space, model_path=model_path, load_version=load_version, load_model_num=load_model_num)

        if load_version == -1:
            self.model = self._build_model()
        else:
            self.model = self._load_model()
        
        self.model_target = clone_model(self.model)

    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(128, input_shape=(self.observation_space,), activation=LeakyReLU()))
        model.add(Dense(128, activation=LeakyReLU()))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))

        return model
    
    def _load_model(self):
        if self.model_num_save == self.model_latest_save:
            model = load_model(os.path.join(self.model_path, f"Model {self.model_num_save}.keras"))
        else:
            model = load_model(os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME, f"Model {self.model_num_save}.keras"))

        self.memory = Memory(self.MEMORY_SIZE, load_path=os.path.join(self.model_path, self.MEMORY_FILENAME))
        
        return model
    
    def learn(self):
        self.learns += 1
        if len(self.memory.indecies) < self.BATCH_SIZE or self.learns % self.LEARN_EVERY != 0:
            return
        
        batch = random.sample(self.memory.indecies, self.BATCH_SIZE)
        states = np.array(self.memory.states)[batch]
        actions = np.array(self.memory.actions)[batch]
        rewards = np.array(self.memory.rewards)[batch]
        next_states = np.array(self.memory.next_states)[batch]
        dones = np.array(self.memory.dones)[batch]

        labels = self.model.predict(np.array(states))
        next_state_values = self.model_target.predict(np.array(next_states))

        for i in range(self.BATCH_SIZE):
            labels[i][actions[i]] = rewards[i] + self.GAMMA * np.max(next_state_values[i]) * (1 - dones[i])

        self.model.fit(np.array(states), labels, batch_size=self.BATCH_SIZE, epochs=1, verbose=0)

        if self.exploration_rate > self.EXPLORATION_MIN:
            self.exploration_rate *= self.EXPLORATION_DECAY
            self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        
        if self.learns % self.UPDATE_TARGET_MODEL_EVERY == 0:
            self.model_target.set_weights(self.model.get_weights())
