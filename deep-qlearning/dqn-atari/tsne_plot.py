from PIL import Image
import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt
# import argparse
# import seaborn as sns
# sns.set_theme()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from tensorflow.keras.callbacks import History, CSVLogger


from rl.agents import SARSAAgent, DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from copy import deepcopy
from sklearn.manifold import TSNE
from matplotlib import pyplot
import matplotlib



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

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

model = Sequential(name= f'{ENV_NAME} agent')
model.add(Permute((2,3,1), input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE))
model.add(Conv2D(32, (8,8), strides = (4,4), activation='relu'))
model.add(Conv2D(64, (4,4), strides= (2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(n_actions, activation='linear'))


modelMID = Sequential(name= f'{ENV_NAME} agent2')
modelMID.add(Permute((2,3,1), input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE))
modelMID.add(Conv2D(32, (8,8), strides = (4,4), activation='relu'))
modelMID.add(Conv2D(64, (4,4), strides= (2,2), activation='relu'))
modelMID.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
modelMID.add(Flatten())
modelMID.add(Dense(512, activation='relu'))
modelMID.add(Dense(n_actions, activation='linear'))

modelBEG = Sequential(name= f'{ENV_NAME} agent3')
modelBEG.add(Permute((2,3,1), input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE))
modelBEG.add(Conv2D(32, (8,8), strides = (4,4), activation='relu'))
modelBEG.add(Conv2D(64, (4,4), strides= (2,2), activation='relu'))
modelBEG.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
modelBEG.add(Flatten())
modelBEG.add(Dense(512, activation='relu'))
modelBEG.add(Dense(n_actions, activation='linear'))

training_steps = 10_000_000
decrease_steps = 9_000_000

memory = SequentialMemory(limit = 1_000_000, window_length = WINDOW_LENGTH)
policy = GreedyQPolicy()

dqn = DQNAgent(model = model, policy = policy, memory = memory, nb_actions = n_actions, 
               processor = AtariProcessor(), nb_steps_warmup = 50_000,
               gamma = 0.99, train_interval = 4, delta_clip = 1, target_model_update = 50_000,
               enable_double_dqn=False, enable_dueling_network=False)

dqn.compile(Adam(lr = 1e-4), metrics=['mae'])



dqnMID = DQNAgent(model = modelMID, policy = policy, memory = memory, nb_actions = n_actions, 
               processor = AtariProcessor(), nb_steps_warmup = 50_000,
               gamma = 0.99, train_interval = 4, delta_clip = 1, target_model_update = 50_000,
               enable_double_dqn=False, enable_dueling_network=False)

dqnMID.compile(Adam(lr = 1e-4), metrics=['mae'])


dqnBEG = DQNAgent(model = modelBEG, policy = policy, memory = memory, nb_actions = n_actions, 
               processor = AtariProcessor(), nb_steps_warmup = 50_000,
               gamma = 0.99, train_interval = 4, delta_clip = 1, target_model_update = 50_000,
               enable_double_dqn=False, enable_dueling_network=False)

dqnBEG.compile(Adam(lr = 1e-4), metrics=['mae'])


dqn.load_weights('dqn_BreakoutDeterministic-v4_weights_secondRun.h5f')
dqnMID.load_weights('dqn_BreakoutDeterministic-v4_weights_6000000.h5f')

intermediate_model = Sequential()
for layer in model.layers[:-1]: # go through until last layer
    intermediate_model.add(layer)

intermediate_modelMID = Sequential()
for layer in modelMID.layers[:-1]: # go through until last layer
    intermediate_modelMID.add(layer)

intermediate_modelBEG = Sequential()
for layer in modelBEG.layers[:-1]: # go through until last layer
    intermediate_modelBEG.add(layer)
    

env.close()
env = gym.make(ENV_NAME) 
procss = AtariProcessor()

data_tsne = {'states': [], 'max_q_value': [] , 'last_layers': [] , 'last_layer_mid':[], 'last_layer_beg':[]}

for i in range(400):
    score = 0
    state = deepcopy(env.reset())
    done = False
    steps = 0
    while not done:
        processed_state = procss.process_observation(state)
        state_4 = dqn.memory.get_recent_state(processed_state)
        action = dqn.forward(processed_state)
        if np.random.rand()<0.05:
            action = env.action_space.sample()
        max_q_value =max(dqn.compute_q_values(state_4))
        #action = env.action_space.sample()
        last_layer_output = intermediate_model.predict(dqn.process_state_batch([state_4]))

        last_layer_outputMID = intermediate_modelMID.predict(dqn.process_state_batch([state_4]))
        last_layer_outputBEG = intermediate_modelBEG.predict(dqn.process_state_batch([state_4]))

        data_tsne['states'].append(state)
        data_tsne['max_q_value'].append(max_q_value)
        data_tsne['last_layers'].append(last_layer_output.flatten())
        data_tsne['last_layer_mid'].append(last_layer_outputMID.flatten())
        data_tsne['last_layer_beg'].append(last_layer_outputBEG.flatten())

        state, reward, done, info = env.step(action)
        state = deepcopy(state)
        score+= reward
        steps +=1
        dqn.backward(reward, terminal = done)

    print(f'Episode {i}, Score {score}, Steps {steps}')
env.close()


last_layer_array = np.stack( data_tsne['last_layers'], axis=0 )
last_layer_embedded = TSNE(n_components=2, perplexity=50).fit_transform(last_layer_array)

last_layer_array_mid = np.stack( data_tsne['last_layer_mid'], axis=0 )
last_layer_embedded_mid = TSNE(n_components=2, perplexity=50).fit_transform(last_layer_array_mid)

last_layer_array_beg = np.stack( data_tsne['last_layer_beg'], axis=0 )
last_layer_embedded_beg = TSNE(n_components=2, perplexity=50).fit_transform(last_layer_array_beg)
## 
last_layer_array = np.stack( data_tsne['last_layers'], axis=0 )
last_layer_embedded = TSNE(n_components=2, perplexity=50).fit_transform(last_layer_array)

###Plot tSNE maps 
matplotlib.style.use('default') 
font = {'family' : 'serif',
        'size'   : 20}
matplotlib.rc('font', **font)

pyplot.figure(figsize=(25, 15)) 
plt.scatter(last_layer_embedded[:,0], last_layer_embedded[:,1], c = data_tsne['max_q_value'], s=5 , cmap = 'nipy_spectral')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('max Q(s,a)', rotation=270)

plt.title('t-SNE for Breakout (15M steps)')
plt.axis('off')
plt.savefig(f'tsne-{ENV_NAME}_15M.png', transparent=True)


pyplot.figure(figsize=(25, 15)) 
plt.scatter(last_layer_embedded_mid[:,0], last_layer_embedded_mid[:,1], c = data_tsne['max_q_value'], s=5 , cmap = 'nipy_spectral')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('max Q(s,a)', rotation=270)

plt.title('t-SNE for Breakout (8M steps)')
plt.axis('off')
plt.savefig(f'tsne-{ENV_NAME}_6M.png', transparent=True)


pyplot.figure(figsize=(25, 15)) 
plt.scatter(last_layer_embedded_beg[:,0], last_layer_embedded_beg[:,1], c = data_tsne['max_q_value'], s=5 , cmap = 'nipy_spectral')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('max Q(s,a)', rotation=270)

plt.title('t-SNE for Breakout (0 steps)')
plt.axis('off')
plt.savefig(f'tsne-{ENV_NAME}_0steps.png', transparent=True)
