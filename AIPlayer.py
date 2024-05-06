import environment
import pygame
import numpy as np
from collections import deque
from model import Agent
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
NUMSNAKES = 4

if __name__ == '__main__':
    game = environment.SnakeGameEnv(num_snakes=NUMSNAKES)
    # game loop
    actions = np.ones(NUMSNAKES)*environment.Direction.NONE

    agents = [Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.0, input_dims=[12], lr=0.003) for i in range(NUMSNAKES)]
    #agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=4, eps_end=0.01, input_dims=[12], lr=0.003)

    scores = []
    eps_history = [[] for i in range(NUMSNAKES)]
    n_games = 1000
    for gamenum in range(n_games):
        score = 0
        game_over = False
        obs = [np.zeros(12) for i in range(NUMSNAKES)]
        while not game_over:
            actions = [agents[i].choose_action(obs[i]) for i in range(NUMSNAKES)]
            new_obs, reward, game_over, alive = game.play_step(actions)
            score += reward

            for j in range(NUMSNAKES):
                if sum(obs[j]) != 0:
                    agents[j].store_transition(obs[j], actions[j], reward[j], new_obs[j], not alive[j])
                    agents[j].learn()

            obs = new_obs
            if gamenum > 300:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
        scores.append(score)
        for k in range(NUMSNAKES):
            eps_history[k].append(agents[k].epsilon)
        avg_score = np.mean(scores, axis=0)
        print("Game " + str(gamenum)+" Over: "+str(score))
        print(avg_score)
        print(str([eps[-1]for eps in eps_history]))
        print("")
        game.reset()



