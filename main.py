import gym
from gym import logger as gymlogger
import numpy as np

gymlogger.set_level(50)  # error only

game = 'CartPole-v1'    # other interesting and simple environments: Pong-v0, MsPacman-v0, CarRacing-v0
show_all_runs = True    # either plot all runs or only the last (default: all)

nr_of_runs = 10
current_run = 1

env = gym.make(game)
observation = env.reset()
timestep = 0

while current_run < nr_of_runs + 1:
    if current_run == nr_of_runs or show_all_runs:
        env.render()

    # your agent goes here
    # action_space.sample() results in a random action being picked
    action = env.action_space.sample()

    # apply the action to the real environment and forward the game
    observation, reward, done, info = env.step(action)
    timestep += 1

    if done:
        # for Monte Carlo Method you will need to update your value matrix here
        # Temporal Difference Learning updates the value matrix in every step

        # this test ends after 'nr_of_runs' (default = 10)
        # change the variable at the top if you want to train longer (recommended)
        print("run: " + str(current_run) + " took " + str(timestep) + " timesteps")
        current_run += 1
        timestep = 0
        observation = env.reset()

env.close()


""" On Windows the last line may finish with the error:
TypeError: 'NoneType' object is not iterable

Just ignore this since it will not influence your test. Please let us know in case you fixed the error, 
so we could include it here.
"""
