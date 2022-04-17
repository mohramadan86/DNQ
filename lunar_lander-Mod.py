# Landing pad is always at coordinates (0,0). Coordinates are the first
# two numbers in state vector. Reward for moving from the top of the screen
# to landing pad and zero speed is about 100..140 points. If lander moves
# away from landing pad it loses reward back. Episode finishes if the lander
# crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
# Solved is 200 points. Landing outside landing pad is possible. Fuel is
# infinite, so an agent can learn to fly and then land on its first attempt.
# Four discrete actions available: do nothing, fire left orientation engine,
# fire main engine, fire right orientation engine.


import gym
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.activations import relu, linear

import numpy as np

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


class DeepQNetwork:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.eps = 1.0
        self.eps_min = .01
        self.eps_decay = .996
        self.gamma = .99
        self.batch_size = 64
        self.lr = 0.001
        self.memory = deque(maxlen=1000000)
        self.model = self.build_CNN_model()

    def build_CNN_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=self.state_space, activation=relu))
        model.add(Dense(90, activation=relu))
        model.add(Dense(150, activation=relu))
        model.add(Dense(300, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def perform (self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_space)
        performvalue = self.model.predict(state)
        return np.argmax(performvalue[0])

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        sampleBatch = random.sample(self.memory, self.batch_size)
        #print (f'Sample Batch Test = {sampleBatch}')
        states = np.array([i[0] for i in sampleBatch])
        actions = np.array([i[1] for i in sampleBatch])
        rewards = np.array([i[2] for i in sampleBatch])
        next_states = np.array([i[3] for i in sampleBatch])
        dones = np.array([i[4] for i in sampleBatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        #print(f'targets_full = {targets_full}')
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        #print(f'targets = {targets}')
        #print(f'targets_full = {targets_full}')
        #print(f'ind = {ind}')

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay


def randomAgent():
    env.reset()
    done = False
    while not done:
        action = random.randrange(env.action_space.n)
        print (f'action = {action}')
        env.render()
        next_state, reward, done, _ = env.step(action)


def train_dqn(episode):
    loss = []
    agent = DeepQNetwork(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.perform(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.save(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss


if __name__ == '__main__':
    print(f'env.observation_space  = {env.observation_space}')
    print(f'env.action_space = {env.action_space}')
    episodes = 400
    loss = train_dqn(episodes)
    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()


