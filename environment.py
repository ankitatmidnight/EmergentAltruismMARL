import pygame
import random
from enum import Enum
from enum import IntEnum
from collections import namedtuple
import numpy as np

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
BLOCK_X = int(WIDTH / BLOCK_SIZE)
BLOCK_Y = int(HEIGHT / BLOCK_SIZE)
LOOKAHEAD = 1
FONTSIZE = 12
RENDER = 1
RESPAWN = 0
Point = namedtuple('Point', 'x, y')
pygame.init()
font = pygame.font.Font('freesansbold.ttf', FONTSIZE)

# font = pygame.font.Font('arial.ttf', 25)

class Direction(IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    NONE = -1


###     Snake Class     ###
class Snake:
    def __init__(self, x=0, y=0):
        self.head = Point(x, y)
        self.orient = Direction.NONE
        self.body = [self.head]
        self.color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
        self.collision_state = np.zeros(4)
        self.food_state = np.zeros(4)
        self.opps_state = np.zeros(4)
        self.alive = True
        self.score = 0

    def move(self, action):
        #if self.orient != (action+2)%4:
        self.orient = action
        match self.orient:
            case Direction.RIGHT:
                self.head = Point(self.head.x + 1, self.head.y)
            case Direction.LEFT:
                self.head = Point(self.head.x - 1, self.head.y)
            case Direction.UP:
                self.head = Point(self.head.x, self.head.y - 1)
            case Direction.DOWN:
                self.head = Point(self.head.x, self.head.y + 1)
        self.body.append(self.head)

    def pop(self):
        self.body.pop(0)

    def die(self):
        self.head = Point(0, 0)
        self.orient = Direction.DOWN
        self.body = [self.head]
        self.collision_state = np.zeros(4, dtype=np.float32)
        self.food_state = np.zeros(4, dtype=np.float32)
        self.opps_state = np.zeros(4, dtype=np.float32)

    def respawn(self, x=0, y=0):
        self.head = Point(x, y)
        self.orient = Direction.NONE
        self.body = [self.head]
        self.collision_state = np.zeros(4)
        self.food_state = np.zeros(4)
        self.opps_state = np.zeros(4)
        self.alive = True
        self.score = 0

    def set_collision_state(self, collision_matrix):
        self.collision_state = np.zeros(4)
        self.collision_state[0] = 1 if sum(sum(collision_matrix[0:2 * LOOKAHEAD + 1, 0:LOOKAHEAD])) else 0
        self.collision_state[1] = 1 if sum(sum(collision_matrix[0:LOOKAHEAD, 0:2 * LOOKAHEAD + 1])) else 0
        self.collision_state[2] = 1 if sum(sum(collision_matrix[0:2 * LOOKAHEAD + 1, LOOKAHEAD + 1:2 * LOOKAHEAD + 1])) else 0
        self.collision_state[3] = 1 if sum(sum(collision_matrix[LOOKAHEAD + 1:2 * LOOKAHEAD + 1, 0:2 * LOOKAHEAD + 1])) else 0

    def set_food_state(self, food):
        self.food_state = np.zeros(4)
        #LEFT
        self.food_state[0] = 1 if food.x < self.head.x else 0
        #UP
        self.food_state[1] = 1 if food.y < self.head.y else 0
        #RIGHT
        self.food_state[2] = 1 if food.x > self.head.x else 0
        #DOWN
        self.food_state[3] = 1 if food.y > self.head.y else 0

    def set_opps_state(self, snakes, num):
        self.opps_state = np.zeros(4)
        for count, snake in enumerate(snakes):
            if count != num and snake.alive:
                #LEFT
                if snake.head.x < self.head.x:
                    self.opps_state[0] = 1
                #UP
                if snake.head.y < self.head.y:
                    self.opps_state[1] = 1
                #RIGHT
                if snake.head.x > self.head.x:
                    self.opps_state[2] = 1
                #DOWN
                if snake.head.y > self.head.y:
                    self.opps_state[3] = 1

    def get_state(self):
        return np.append(self.collision_state, np.append(self.food_state, self.opps_state))


###     Snake Game Env Class     ###
class SnakeGameEnv:
    def __init__(self, w=640, h=480, num_snakes=3):
        self.num_snakes = num_snakes
        self.snakes = []
        self.lookahead_collide = []
        self.lookahead_noncollide = []
        self.display = pygame.display.set_mode((w, h))
        self.food = Point(0, 0)
        self.gamenum = 1

        snake_pos = random.sample(range(int((BLOCK_X - 2) * (BLOCK_Y - 2))), num_snakes)
        for pos in snake_pos:
            self.snakes.append(Snake(x=int(pos / (BLOCK_X - 2)) + 1, y=pos % (BLOCK_Y - 2) + 1))
        self.place_food()

    def reset(self):
        snake_pos = random.sample(range(int((BLOCK_X - 2) * (BLOCK_Y - 2))), self.num_snakes)
        for count, pos in enumerate(snake_pos):
            self.snakes[count].respawn(x=int(pos / (BLOCK_X - 2)) + 1, y=pos % (BLOCK_Y - 2) + 1)
        self.place_food()
        self.gamenum += 1

    def play_step(self, actions):
        reward = np.zeros(self.num_snakes)
        game_over = 1
        alive = np.ones(self.num_snakes)

        for i in range(self.num_snakes):
            if self.snakes[i].alive:
                self.snakes[i].move(actions[i])
                if self.snakes[i].food_state[self.snakes[i].orient] and self.snakes[i].orient is not Direction.NONE:
                    reward[i] += 1
                else:
                    reward[i] -= 1

                if self.snakes[i].head == self.food:
                    self.place_food()
                    reward += 10
                    #reward[i] += 10
                    self.snakes[i].score += 1
                else:
                    self.snakes[i].pop()

                if self.is_collision(self.snakes[i].head, i):
                    self.snakes[i].alive = False
                    #reward -= 10
                    reward[i] -= 50

                self.snakes[i].set_collision_state(self.lookahead(self.snakes[i].head, i, LOOKAHEAD))
                self.snakes[i].set_food_state(self.food)
                self.snakes[i].set_opps_state(self.snakes, i)

        for i in range(self.num_snakes):
            alive[i] = self.snakes[i].alive
            if self.snakes[i].alive:
                game_over = 0
            else:
                self.snakes[i].die()
                if RESPAWN:
                    self.snakes[i].alive = True
        if True:#self.gamenum > 750:
            self.render()

        new_obs = [snake.get_state() for snake in self.snakes]

        return new_obs, reward, game_over, alive

    def lookahead(self, head, snake_id, look_dist):
        collision_array = np.zeros((2 * look_dist + 1, 2 * look_dist + 1))
        for x in np.linspace(start=-look_dist, stop=look_dist, num=2 * look_dist + 1, dtype=int):
            for y in np.linspace(start=-look_dist, stop=look_dist, num=2 * look_dist + 1, dtype=int):
                lookahead_pt = Point(head.x + x, head.y + y)
                if self.in_bounds(lookahead_pt) and not (x == 0 and y == 0):
                    if abs(x)+abs(y) <= look_dist:
                        if self.is_collision(lookahead_pt, snake_id):
                            self.lookahead_collide.append(lookahead_pt)
                            collision_array[x + look_dist][y + look_dist] = 1
                        else:
                            self.lookahead_noncollide.append(lookahead_pt)
        return collision_array

    def place_food(self):
        while True:
            open_space = True
            self.food = self.rand_pos()
            for snake in self.snakes:
                if self.food in snake.body:
                    open_space = False
            if open_space:
                return

    def in_bounds(self, pt):
        if pt.x < 0 or pt.x > BLOCK_X - 1:
            return False
        if pt.y < 0 or pt.y > BLOCK_Y - 1:
            return False
        return True

    def is_collision(self, pt, snake_id):
        head = pt
        # Wall Collision
        if head.x <= 0 or head.x >= BLOCK_X - 1:
            return True
        if head.y <= 0 or head.y >= BLOCK_Y - 1:
            return True
        # Snake Collision
        for count, snake in enumerate(self.snakes):
            # Collision with self
            if count == snake_id:
                if head in snake.body[:-1] and len(snake.body) > 1:
                    return True
            # Collision with other snake
            elif head in snake.body:
                return True
        return False

    def render(self):
        # Background
        self.display.fill(pygame.Color('Black'))
        # Walls
        for x in range(BLOCK_X):
            pygame.draw.rect(self.display, pygame.Color('Gray'), pygame.Rect(x * BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, pygame.Color('Gray'),
                             pygame.Rect(x * BLOCK_SIZE, BLOCK_SIZE * (BLOCK_Y - 1), BLOCK_SIZE, BLOCK_SIZE))
        for y in range(BLOCK_Y):
            pygame.draw.rect(self.display, pygame.Color('Gray'), pygame.Rect(0, BLOCK_SIZE * y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, pygame.Color('Gray'),
                             pygame.Rect(BLOCK_SIZE * (BLOCK_X - 1), BLOCK_SIZE * y, BLOCK_SIZE, BLOCK_SIZE))
        # Fruit
        pygame.draw.rect(self.display, pygame.Color('Orange'),
                         pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Snakes
        for snake in self.snakes:
            for pt in snake.body:
                pygame.draw.rect(self.display, snake.color,
                                 pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Lookahead Collisions
        for collision in self.lookahead_collide:
            color = list(self.display.get_at((BLOCK_SIZE * collision.x, BLOCK_SIZE * collision.y)))[0:3]
            color = [min(color[0] + 100, 255), color[1], color[2]]
            pygame.draw.rect(self.display, color,
                             pygame.Rect(BLOCK_SIZE * collision.x, BLOCK_SIZE * collision.y, BLOCK_SIZE, BLOCK_SIZE))
        self.lookahead_collide.clear()
        for collision in self.lookahead_noncollide:
            color = list(self.display.get_at((BLOCK_SIZE * collision.x, BLOCK_SIZE * collision.y)))[0:3]
            color = [color[0], min(color[1] + 50, 255), color[2]]
            pygame.draw.rect(self.display, color,
                             pygame.Rect(BLOCK_SIZE * collision.x, BLOCK_SIZE * collision.y, BLOCK_SIZE, BLOCK_SIZE))
        self.lookahead_noncollide.clear()

        for snake in self.snakes:
            eye_1_offset = Point(BLOCK_SIZE / 8 * 1, BLOCK_SIZE / 8 * 3)
            eye_2_offset = Point(BLOCK_SIZE / 8 * 5, BLOCK_SIZE / 8 * 3)
            match snake.orient:
                case Direction.RIGHT:
                    eye_1_offset = Point(BLOCK_SIZE / 8 * 5, BLOCK_SIZE / 8 * 1)
                    eye_2_offset = Point(BLOCK_SIZE / 8 * 5, BLOCK_SIZE / 8 * 5)
                case Direction.LEFT:
                    eye_1_offset = Point(BLOCK_SIZE / 8 * 1, BLOCK_SIZE / 8 * 1)
                    eye_2_offset = Point(BLOCK_SIZE / 8 * 1, BLOCK_SIZE / 8 * 5)
                case Direction.UP:
                    eye_1_offset = Point(BLOCK_SIZE / 8 * 1, BLOCK_SIZE / 8 * 1)
                    eye_2_offset = Point(BLOCK_SIZE / 8 * 5, BLOCK_SIZE / 8 * 1)
                case Direction.DOWN:
                    eye_1_offset = Point(BLOCK_SIZE / 8 * 1, BLOCK_SIZE / 8 * 5)
                    eye_2_offset = Point(BLOCK_SIZE / 8 * 5, BLOCK_SIZE / 8 * 5)
            pygame.draw.rect(self.display, pygame.Color('Black'),
                             pygame.Rect((snake.head.x * BLOCK_SIZE) + eye_1_offset.x,
                                         (snake.head.y * BLOCK_SIZE) + eye_1_offset.y, BLOCK_SIZE / 4, BLOCK_SIZE / 4))
            pygame.draw.rect(self.display, pygame.Color('Black'),
                             pygame.Rect((snake.head.x * BLOCK_SIZE) + eye_2_offset.x,
                                         (snake.head.y * BLOCK_SIZE) + eye_2_offset.y, BLOCK_SIZE / 4, BLOCK_SIZE / 4))
            # text = font.render(str(snake.collision_state), True, pygame.Color("white"))
            # self.display.blit(text, [snake.head.x * BLOCK_SIZE + BLOCK_SIZE, snake.head.y * BLOCK_SIZE - BLOCK_SIZE])
            # text = font.render(str(snake.food_state), True, pygame.Color("white"))
            # self.display.blit(text, [snake.head.x * BLOCK_SIZE + BLOCK_SIZE,
            #                          snake.head.y * BLOCK_SIZE - BLOCK_SIZE + FONTSIZE])
            # text = font.render(str(snake.opps_state), True, pygame.Color("white"))
            # self.display.blit(text, [snake.head.x * BLOCK_SIZE + BLOCK_SIZE,
            #                          snake.head.y * BLOCK_SIZE - BLOCK_SIZE + FONTSIZE + FONTSIZE])
        text = font.render("Game: "+str(self.gamenum), True, pygame.Color("white"))
        self.display.blit(text, [BLOCK_SIZE, BLOCK_SIZE])
        text = font.render("Snake Len: "+str([snake.score for snake in self.snakes]), True, pygame.Color("white"))
        self.display.blit(text, [BLOCK_SIZE, BLOCK_SIZE*2])

        # Display
        pygame.display.flip()

    def rand_pos(self):
        randx = random.randint(0, BLOCK_X - 3) + 1
        randy = random.randint(0, BLOCK_Y - 3) + 1
        return Point(randx, randy)
