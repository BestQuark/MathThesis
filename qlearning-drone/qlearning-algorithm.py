"""
Author: @BestQuark
"""
import drone
import qlearning as ql
import numpy as np
import gym
import matplotlib.pyplot as plt


#CAMBIAR No INTENTO CADA RUN 
episodios = 1000000
episodios = 10



#no-barrier
env = drone.droneEnv(size= 15, reward_loc=[[2,2]], start_seed=[2,2], randomize=True, fuel=30)
polyStar = np.load("poliStar.npy")


#barrier
# env = drone.droneEnv(size= 15, reward_loc=[[2,2]], start_seed=[2,2], prohibit_loc=[[0,7], [2,7], [4,7], [6,7], [8,7], [10,7]], prohibit_size=6*[2],prohibit_value=6*[-50], randomize=True, fuel=50)
# polyStar = np.load("poliStar_barrera.npy")


#epsilon-greedy
epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
algorithms = [0,1]

intento = 0
for algorithm in algorithms:
    for epsilon in epsilons:
        intento += 1
        qtable, scores, diffs = ql.epsilon_greedy(env, lr=0.3 ,algorithm=algorithm,epsilon=epsilon, ntry=intento, save_every=episodios, episodes = episodios,optimal_policy=polyStar, render=False)

        np.save(f"scores_algorithm_{algorithm}_epsilon_{epsilon}_try_{intento}.npy",scores)

        if not not diffs:
            np.save(f"diffs_algorithm_{algorithm}_epsilon_{epsilon}_try_{intento}.npy", diffs)

        print(f"Intento: {intento}, Epsilon: {epsilon}, Algoritmo: {algorithm}")