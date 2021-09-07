from PIL import Image
import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

#ENV_NAME = 'Pong-v0'
#ENV_NAME = 'SpaceInvaders-v0'
#ENV_NAME = 'SpaceInvadersNoFrameskip-v4'
#ENV_NAME = 'SpaceInvadersDeterministic-v4'
ENV_NAME = 'BreakoutDeterministic-v4'

env = gym.make(ENV_NAME)
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
height, width, n_channels = env.observation_space.shape

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8') 

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

model = Sequential(name= f'{ENV_NAME} agent')
model.add(Permute((2,3,1), input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE))
model.add(Conv2D(32, (8,8), strides = (4,4), activation='relu'))
model.add(Conv2D(64, (4,4), strides= (2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(n_actions, activation='linear'))

training_steps = 10_000_000
annealed_steps = 9_000_000
memory_limit = 1_500_000
callback_interval = training_steps//5

memory = SequentialMemory(limit = memory_limit, window_length = WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr = 'eps', value_max=1., value_min=0.1, value_test = 0.05, nb_steps = annealed_steps)
dqn = DQNAgent(model = model, policy = policy, memory = memory, nb_actions = n_actions, 
               processor = AtariProcessor(), nb_steps_warmup = 50_000,
               gamma = 0.99, train_interval = 4, delta_clip = 1., target_model_update = 10_000)
dqn.compile(Adam(learning_rate = 0.0001), metrics=['mae'])

weights_filename = f'dqn_{ENV_NAME}_weights.h5f'
checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
log_filename = f'dqn_{ENV_NAME}_log.json'

callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=callback_interval), FileLogger(log_filename, interval=100)]
dqn_history = dqn.fit(env, callbacks=callbacks, nb_steps=training_steps, visualize=False, verbose = 1)

dqn.save_weights(weights_filename, overwrite=True)

with open('trainHistoryDict.pickle', 'wb') as file_pi:
    pickle.dump(dqn_history.history, file_pi)

