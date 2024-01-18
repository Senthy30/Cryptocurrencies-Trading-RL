import numpy as np
from .DQN import DQN

class DDQN(DQN):
    def learn(self):
        self.learns += 1
        if self.memory.current_index < self.BATCH_SIZE:
            return
        
        if self.learns % self.LEARN_EVERY != 0:
            return
        
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.BATCH_SIZE)

        # Use target network for action selection
        next_actions = np.argmax(self.model.predict(next_states), axis=1)
        next_state_values = self.model_target.predict(next_states)
        selected_next_state_values = next_state_values[range(self.BATCH_SIZE), next_actions]

        labels = self.model.predict(states)

        for i in range(self.BATCH_SIZE):
            labels[i][int(actions[i])] = rewards[i] + self.GAMMA * selected_next_state_values[i] * (1 - dones[i])

        self.model.fit(states, labels, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=0)

        if self.exploration_rate > self.EXPLORATION_MIN:
            self.exploration_rate *= self.EXPLORATION_DECAY
            self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        
        if self.learns % self.UPDATE_TARGET_MODEL_EVERY == 0:
            self.model_target.set_weights(self.model.get_weights())
