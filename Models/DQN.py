from .config import ConfigModel
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

class DQN(ConfigModel):

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.model = self.build_model()
        self.model_target = clone_model(self.model)

    def build_model(self):
        model = Sequential()
        
        model.add(Dense(24, input_shape=(self.observation_space,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))

        return model