import roboschool
import gym
from OpenGL import GLU

env = gym.make('RoboschoolHumanoid-v1')
env.reset()
while True:
    env.step(env.action_space.sample())
    env.render()
