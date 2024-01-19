import numpy as np
import random
import os
from .Memory import Memory
import tensorflow as tf

class ConfigModel():
    GAMMA = 0.98
    LEARNING_RATE = 0.001

    MEMORY_SIZE = 1000000

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.001
    EXPLORATION_DECAY = 0.99

    BATCH_SIZE = 64
    EPOCHS = 40

    LEARN_EVERY = 2

    MEMORY_FILENAME = "Memory"
    LAST_VERSIONS_FILENAME = "Last Versions"

    def __init__(self, observation_space, action_space, model_path = "", load_version = -1, load_model_num = -1):
        self.exploration_rate = self.EXPLORATION_MAX
        self.observation_space = observation_space
        self.action_space = action_space

        self.memory = Memory(self.MEMORY_SIZE, observation_space=self.observation_space)

        self.model = None
        if load_version == -1:
            self._build_new_version(model_path)
        else:
            self._load_version(model_path, load_version, load_model_num)
        self.model_latest_save = self.get_current_save_model()

        self.learns = 0

        self.actions_taked_by_itself = [0, 0, 0]

    def _build_new_version(self, model_path):
        self.model_path = os.path.join("Models", "Checkpoint", model_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model_num_save = 0
        self.model_version = self.get_current_version_model()
        self.model_path = os.path.join(self.model_path, f"Version {self.model_version}")

        os.makedirs(self.model_path)
        os.makedirs(os.path.join(self.model_path, self.MEMORY_FILENAME))
        os.makedirs(os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME))

    def _load_version(self, model_path, load_version, load_model_num):
        self.model_path = os.path.join("Models", "Checkpoint", model_path, f"Version {load_version}")
        self.model_version = load_version

        if load_model_num != -1:
            self.model_num_save = load_model_num
        else:
            self.model_num_save = self.get_current_save_model()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return self.act_greedy(state)
    
    def act_greedy(self, state):
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        self.actions_taked_by_itself[action] += 1

        return action
    
    def learn(self):
        pass

    def save_model(self):
        if f"Model {self.model_num_save}.keras" in os.listdir(self.model_path):
            last_model_saved = f"Model {self.model_num_save}.keras"
            os.rename(
                os.path.join(self.model_path, last_model_saved), 
                os.path.join(self.model_path, self.LAST_VERSIONS_FILENAME, last_model_saved)
            )
        
        self.model_num_save += 1
        self.q_net.save(os.path.join(self.model_path, f"Model {self.model_num_save}.keras"))

        path_to_memory = os.path.join(self.model_path, self.MEMORY_FILENAME)
        for file in os.listdir(path_to_memory):
            os.remove(os.path.join(path_to_memory, file))

        self.memory.save_memory(path_to_memory)

    def get_current_version_model(self):
        last_version = 0
        for file in os.listdir(self.model_path):
            try:
                file = int(file.split(" ")[1])
            except:
                continue

            if file > last_version:
                last_version = file

        return last_version + 1
    
    def get_current_save_model(self):
        last_save = 0
        for file in os.listdir(self.model_path):
            try:
                file = int(file.split(".keras")[0].split(" ")[1])
            except:
                continue

            if file > last_save:
                last_save = file

        return last_save

    