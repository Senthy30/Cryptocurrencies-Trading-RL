from .config import ConfigModel
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam

class DQN(ConfigModel):

    def __init__(self, observation_space, action_space, load_model_value = -1):
        super().__init__(observation_space=observation_space, action_space=action_space)

        if (load_model_value == -1):
            self.model = self.build_model()
        else:
            self.model = load_model("Models/Checkpoint/Model_{load_model_value}.keras")
        self.model_target = clone_model(self.model)

    def build_model(self):
        model = Sequential()
        
        model.add(Dense(128, input_shape=(self.observation_space,), activation=LeakyReLU()))
        model.add(Dense(128, activation=LeakyReLU()))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))

        return model