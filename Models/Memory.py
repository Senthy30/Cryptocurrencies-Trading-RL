import pickle
import os
import random
import numpy as np
from collections import deque

class Memory():

    MEMORY_STATES_FILENAME = "States.pkl"
    MEMORY_ACTIONS_FILENAME = "Actions.pkl"
    MEMORY_REWARDS_FILENAME = "Rewards.pkl"
    MEMORY_NEXT_STATES_FILENAME = "Next States.pkl"
    MEMORY_DONES_FILENAME = "Dones.pkl"

    def __init__(self, memory_size, load_path=None, observation_space=1):
        self.observation_space = observation_space
        self.load_path = load_path
        if self.load_path is None:
            self._build_memory(memory_size)
        else:
            self._load_memory(memory_size)

    def _build_memory(self, memory_size):
        self.memory_size = memory_size
        self.states = np.zeros((memory_size, self.observation_space))
        self.actions = np.zeros(memory_size)
        self.rewards = np.zeros(memory_size)
        self.next_states = np.zeros((memory_size, self.observation_space))
        self.dones = np.zeros(memory_size)
        
        self.indecies = np.zeros(memory_size)
        self.current_index = 0

    def _load_memory(self, memory_size):
        self.states = self.load_memory_np_array(self.MEMORY_STATES_FILENAME)
        self.actions = self.load_memory_np_array(self.MEMORY_ACTIONS_FILENAME)
        self.rewards = self.load_memory_np_array(self.MEMORY_REWARDS_FILENAME)
        self.next_states = self.load_memory_np_array(self.MEMORY_NEXT_STATES_FILENAME)
        self.dones = self.load_memory_np_array(self.MEMORY_DONES_FILENAME)

        self.memory_size = self.actions.shape[0]

        self.indecies = np.zeros(memory_size)
        for i in range(self.memory_size):
            self.indecies[i] = i
            
        self.current_index = self.actions.shape[0]

    def remember(self, state, action, reward, next_state, done):
        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.dones[self.current_index] = done

        self.indecies[self.current_index] = self.current_index
        self.current_index += 1

        if self.current_index >= self.memory_size:
            self.current_index = 0

    def get_batch(self, batch_size):
        batch = np.random.choice(self.current_index, batch_size, replace=False)
        states = np.zeros((batch_size, self.observation_space))
        actions = np.zeros(batch_size)
        rewards = np.zeros(batch_size)
        next_states = np.zeros((batch_size, self.observation_space))
        dones = np.zeros(batch_size)

        for i in range(len(batch)):
            states[i] = self.states[batch[i]]
            actions[i] = self.actions[batch[i]]
            rewards[i] = self.rewards[batch[i]]
            next_states[i] = self.next_states[batch[i]]
            dones[i] = self.dones[batch[i]]

        return states, actions, rewards, next_states, dones

    def save_memory(self, path_to_memory):
        self.save_memory_np_array(self.states, path_to_memory, self.MEMORY_STATES_FILENAME)
        self.save_memory_np_array(self.actions, path_to_memory, self.MEMORY_ACTIONS_FILENAME)
        self.save_memory_np_array(self.rewards, path_to_memory, self.MEMORY_REWARDS_FILENAME)
        self.save_memory_np_array(self.next_states, path_to_memory, self.MEMORY_NEXT_STATES_FILENAME)
        self.save_memory_np_array(self.dones, path_to_memory, self.MEMORY_DONES_FILENAME)

    def save_memory_np_array(self, deque, path_to_memory, filename):
        with open(os.path.join(path_to_memory, filename), "wb") as file:
            pickle.dump(deque, file)

    def load_memory_np_array(self, filename):
        with open(os.path.join(self.load_path, filename), "rb") as file:
            return pickle.load(file)