# Play around with Q-learning algorithm to solve the desert problem. The desert problem is a simple grid world where
# the agent has to navigate through a desert to find water. You can inspect the code output: ./examples/q-learning.ipynb

from copy import deepcopy
import numpy as np
import random

# initializing the grid environment:
DESERT = "desert"
AGENT = "agent"
WATER = "water"
EMPTY = "*"

grid = [[AGENT, EMPTY, DESERT], [EMPTY, EMPTY, WATER]]

# initializing the agent's movement:
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

# initializing the parameters:
random.seed(2024)

N_STATES = 6
N_EPISODES = 20

MAX_EPISODE_STEPS = 100

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

q_table = dict()


class State:

    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos

    def __eq__(self, other):
        return (
            isinstance(other, State)
            and self.grid == other.grid
            and self.agent_pos == other.agent_pos
        )

    def __hash__(self):
        return hash(str(self.grid) + str(self.agent_pos))

    def __str__(self):
        return f"State(grid={self.grid}, agent_pos={self.agent_pos})"


# initializing the start state:
start_state = State(grid=grid, agent_pos=[1, 1])


def new_agent_pos(state, action):
    p = deepcopy(state.agent_pos)
    if action == UP:
        p[0] = max(0, p[0] - 1)
    elif action == DOWN:
        p[0] = min(len(state.grid) - 1, p[0] + 1)
    elif action == LEFT:
        p[1] = max(0, p[1] - 1)
    elif action == RIGHT:
        p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
    else:
        raise ValueError(f"Unknown action {action}")
    return p


def act(state, action):

    p = new_agent_pos(state, action)
    grid_item = state.grid[p[0]][p[1]]

    new_grid = deepcopy(state.grid)

    if grid_item == DESERT:
        reward = -100
        is_done = True
        new_grid[p[0]][p[1]] += AGENT

    elif grid_item == WATER:
        reward = 1000
        is_done = True
        new_grid[p[0]][p[1]] += AGENT

    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        old = state.agent_pos
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] = AGENT

    elif grid_item == AGENT:
        reward = -1
        is_done = False

    else:
        raise ValueError(f"Unknown grid item {grid_item}")

    return State(grid=new_grid, agent_pos=p), reward, is_done


def q(state, action=None):

    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if action is None:
        return q_table[state]

    return q_table[state][action]


def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)
    else:
        return np.argmax(q(state))


def main():
    # training the agent:
    for e in range(N_EPISODES):

        state = start_state
        total_reward = 0
        alpha = alphas[e]

        for _ in range(MAX_EPISODE_STEPS):
            action = choose_action(state)
            next_state, reward, done = act(state, action)
            total_reward += reward

            q(state)[action] = q(state, action) + alpha * (
                reward + gamma * np.max(q(next_state)) - q(state, action)
            )
            state = next_state
            if done:
                break
        print(f"Episode {e + 1}: total reward -> {total_reward}")
