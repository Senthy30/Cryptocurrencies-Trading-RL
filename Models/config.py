import numpy as np
import random
import time
from collections import deque

class Memory():

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.next_states = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)
        
        self.indecies = deque(maxlen=memory_size)
        self.current_index = 0

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        self.indecies.append(self.current_index)
        self.current_index += 1

class ConfigModel():

    GAMMA = 0.98
    LEARNING_RATE = 0.001

    MEMORY_SIZE = 1000000

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.12
    EXPLORATION_DECAY = 0.995

    BATCH_SIZE = 40

    UPDATE_TARGET_MODEL_EVERY = 1000

    def __init__(self, observation_space, action_space):
        self.exploration_rate = self.EXPLORATION_MAX
        self.observation_space = observation_space
        self.action_space = action_space

        self.memory = Memory(self.MEMORY_SIZE)

        self.learns = 0
        self.model = None
        self.target_model = None

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return self.act_greedy(state)
    
    def act_greedy(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def learn(self):
        if len(self.memory.indecies) < self.BATCH_SIZE:
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
        
        self.learns += 1
        if self.learns % self.UPDATE_TARGET_MODEL_EVERY == 0:
            self.model_target.set_weights(self.model.get_weights())