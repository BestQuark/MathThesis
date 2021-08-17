"""
Author: @BestQuark
"""

import gym
from gym.spaces import MultiDiscrete, Discrete
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle

class droneEnv(gym.Env):

    def __init__(self, start_seed=[0,0], randomize=False, size=10, all_rewards_value = 0, reward_size=[3], reward_loc=[[0,0]], reward_value=[100], prohibit_size=[0], prohibit_loc = [[5,5]], prohibit_value=[-50], fuel=0):
        """
        start_seed: where drone starts
        randomize: If True, every episode the start location changes randomly, if False, starts in start_seed
        size: size of the map
        all_rewards_value: rewards for every state

        reward_size[i]: size of ith reward zone (all asumed squares)
        reward_loc[i]: location of ith reward zone 
        reward_value[i]: value of ith reward zone

        prohibit_size[i]: size of ith prohibited zone (all asumed squares)
        prohibit_loc[i]: location of ith prohibited zone
        prohibit_value[i]: value of ith prohibited zone

        fuel: how many steps before stopping

        """

        self.size = size
        self.start = start_seed
        self.is_random = randomize
        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([(self.size,self.size)])
        self.state = np.array(self.start)
        self.rewards = all_rewards_value*np.ones((self.size, self.size))

        assert len(reward_size)==len(reward_loc), "reward_size and reward_loc have different lenghts"
        assert len(prohibit_size)==len(prohibit_loc), "prohibit_size and prohibit_loc have different lengths"
        assert start_seed[0]<=size and start_seed[1]<=size, "Start point out of bounds"

        for indx, rew_size in enumerate(reward_size):
            loc_x, loc_y = reward_loc[indx] 
            val = reward_value[indx]
            try:
                self.rewards[loc_x: loc_x+rew_size, loc_y:loc_y+rew_size] = val*np.ones((rew_size, rew_size))
            except:
                raise Exception("Couldn't add rewards")

        for indx, proh_size in enumerate(prohibit_size):
            loc_x, loc_y = prohibit_loc[indx]
            val = prohibit_value[indx]
            try:
                self.rewards[loc_x: loc_x+proh_size, loc_y:loc_y+proh_size] = val*np.ones((proh_size, proh_size))
            except:
                raise Exception("Couldn't add prohibited rewards")
        
        if fuel==0:
            self.fuel_tank = int(self.size*self.size*0.2)
        else:
            self.fuel_tank = fuel
        
        self.fuel = self.fuel_tank
        
    
    def step(self, action):
        #Next Step

        self.fuel -= 1
        n_grid = self.size-1
        x,y = self.state

        if action == 0:
            sig_pasos = [[x,min(y+1, n_grid)], [x, min(y+2, n_grid)], [max(x-1,0), min(y+2, n_grid)], [max(x-1, 0), min(y+1, n_grid)]]
            sig_paso = sig_pasos[np.random.choice(np.arange(4), 1, p = [0.3, 0.4, 0.2, 0.1])[0]]

        elif action== 1:
            sig_pasos = [ [x,y], [x, max(y-1, 0)], [max(x-1,0), y], [max(x-1,0), max(y-1,0)] ]
            sig_paso = sig_pasos[np.random.choice(np.arange(4), 1, p = [0.3, 0.3, 0.2, 0.2])[0]]

        elif action == 2:
            sig_pasos = [ [max(x-1,0), min(y+1, n_grid)], [max(x-1, 0), y], [max(x-2,0), y], [max(x-2,0), min(y+1, n_grid)] ]
            sig_paso = sig_pasos[np.random.choice(np.arange(4), 1, p = [0.3, 0.2, 0.3, 0.2])[0]]

        elif action == 3:
            sig_pasos = [[min(x+1, n_grid), y], [min(x+1, n_grid), min(y+1, n_grid)], [x,y], [x, min(y+1, n_grid)]]
            sig_paso = sig_pasos[np.random.choice(np.arange(4), 1, p = [0.3, 0.4, 0.2, 0.1])[0]]

        
        self.state = np.array(sig_paso)
        x,y = sig_paso
        reward = self.rewards[x,y]
        
        if self.fuel<=0:
            done = True
        else:
            done = False
            
        info = {}
        return self.state, reward, done, info
    
    def render(self, mode = 'human'):
        #Visualizations
        
        matriz = np.copy(self.rewards)
        x,y = self.state
        matriz[x,y] = 150
        clear_output(wait=True)
        imag = plt.imshow(matriz, cmap='gray', vmin=-50)
        plt.axis('off')
        plt.show()
    
    def reset(self):
        #Reset

        if self.is_random==False:
            self.state = np.array(self.start)
        elif self.is_random==True:
            self.state = np.array([np.random.randint(0,self.size),np.random.randint(0,self.size)])

        self.fuel = self.fuel_tank

        return self.state
    
