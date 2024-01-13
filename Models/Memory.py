import pickle
import os
from collections import deque

class Memory():

    MEMORY_STATES_FILENAME = "States.pkl"
    MEMORY_ACTIONS_FILENAME = "Actions.pkl"
    MEMORY_REWARDS_FILENAME = "Rewards.pkl"
    MEMORY_NEXT_STATES_FILENAME = "Next States.pkl"
    MEMORY_DONES_FILENAME = "Dones.pkl"

    def __init__(self, memory_size, load_path=None):
        
        self.load_path = load_path
        if self.load_path is None:
            self._build_memory(memory_size)
        else:
            self._load_memory(memory_size)

    def _build_memory(self, memory_size):
        self.memory_size = memory_size
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.next_states = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)
        
        self.indecies = deque(maxlen=memory_size)
        self.current_index = 0

    def _load_memory(self, memory_size):
        self.states = self.load_memory_deque(self.MEMORY_STATES_FILENAME)
        self.actions = self.load_memory_deque(self.MEMORY_ACTIONS_FILENAME)
        self.rewards = self.load_memory_deque(self.MEMORY_REWARDS_FILENAME)
        self.next_states = self.load_memory_deque(self.MEMORY_NEXT_STATES_FILENAME)
        self.dones = self.load_memory_deque(self.MEMORY_DONES_FILENAME)

        self.memory_size = self.states.maxlen

        self.indecies = deque(maxlen=self.memory_size)
        for i in range(len(self.actions)):
            self.indecies.append(i)
            
        self.current_index = len(self.indecies)

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        self.indecies.append(self.current_index)
        self.current_index += 1

    def save_memory(self, path_to_memory):
        self.save_memory_deque(self.states, path_to_memory, self.MEMORY_STATES_FILENAME)
        self.save_memory_deque(self.actions, path_to_memory, self.MEMORY_ACTIONS_FILENAME)
        self.save_memory_deque(self.rewards, path_to_memory, self.MEMORY_REWARDS_FILENAME)
        self.save_memory_deque(self.next_states, path_to_memory, self.MEMORY_NEXT_STATES_FILENAME)
        self.save_memory_deque(self.dones, path_to_memory, self.MEMORY_DONES_FILENAME)

    def save_memory_deque(self, deque, path_to_memory, filename):
        with open(os.path.join(path_to_memory, filename), "wb") as file:
            pickle.dump(deque, file)

    def load_memory_deque(self, filename):
        with open(os.path.join(self.load_path, filename), "rb") as file:
            return pickle.load(file)