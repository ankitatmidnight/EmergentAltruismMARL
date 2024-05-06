import environment
import pygame
import numpy as np
import time

if __name__ == '__main__':
    game = environment.SnakeGameEnv(num_snakes=2)
    # game loop
    actionA = environment.Direction.NONE
    actionB = environment.Direction.NONE
    game_over = False
    score = [0,0]
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_LEFT:
                        actionA = environment.Direction.LEFT
                    case pygame.K_RIGHT:
                        actionA = environment.Direction.RIGHT
                    case pygame.K_UP:
                        actionA = environment.Direction.UP
                    case pygame.K_DOWN:
                        actionA = environment.Direction.DOWN
                    case pygame.K_a:
                        actionB = environment.Direction.LEFT
                    case pygame.K_d:
                        actionB = environment.Direction.RIGHT
                    case pygame.K_w:
                        actionB = environment.Direction.UP
                    case pygame.K_s:
                        actionB = environment.Direction.DOWN
        new_obs, reward, game_over, info = game.play_step([actionA, actionB])
        score += reward
        print(score)
        time.sleep(0.1)

    print('over')

