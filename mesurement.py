import numpy as np
import gym, roboschool
import os
import time
from datetime import datetime
from OpenGL import GLU

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tensorflow gpu 設定
#import tensorflow as tf
#tf.Session(config=tf.ConfigProto(device_count = {'GPU': 2}))


def make_env(env_name, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_name: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


class MyRemodelendEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, a):

        x = self.env.ordered_joints[0].current_position()

        origin_value = np.array([j2.current_position() for j2 in self.env.ordered_joints], dtype=np.float32).flatten()
        obs, reward, terminal, info = self.env.step(a)

        return obs, reward, terminal, origin_value


#import myenv
#test_env = gym.make('myhumanoid-v0')


# 学習設定
train = False       # 学習をするかどうか
validation = True   # 学習結果を使って評価をするかどうか

env_name = 'RoboschoolHumanoid-v1'
num_cpu = 1         # 学習に使用するCPU数
learn_timesteps = 10**3     # 学習タイムステップ

ori_env = gym.make(env_name)
env = DummyVecEnv([lambda: ori_env])
#env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])
env.reset()
#env.render()
#time.sleep(5)

savedir = './stable_baselines2/{}/'.format(env_name)
#logdir = '{}tensorboard_log/'.format(savedir)
#os.makedirs(savedir, exist_ok=True)


env_name = 'RoboschoolHumanoid-v1'
test_env = gym.make(env_name)
test_env = MyRemodelendEnv(test_env)
test_env = DummyVecEnv([lambda: test_env])

# 学習結果の確認
if validation:
    model = PPO2.load('{}ppo2_model'.format(savedir))

    done = False
    #obs = env.reset()
    obs = test_env.reset()

    for step in range(1):
        if done:
            time.sleep(1)
            o = test_env.reset()
            break

        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        print(obs.shape)
        print(obs)

    data = info[0]

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel(('X'))
    ax.set_xlabel(('Y'))
    ax.set_xlabel(('Z'))

    print(type(data))
    print(len(data))
    print(type(data[12]))
    print(type(data[10]))
    print(type(data[8]))
    abdomen = [data[4]], [data[2]], [data[0]]
    r_hip = [data[6]], [data[10]], [data[8]]
    l_hip = [data[14]], [data[18]], [data[16]]
    ax.plot([data[4]], [data[2]], [data[0]], marker='o', label='abdomen')
    ax.plot([data[6]], [data[10]], [data[8]], marker='x', label='r_hip')
    ax.plot([data[14]], [data[18]], [data[16]], marker='+', label='l_hip')
    ax.legend()

    plt.show()


env.close()

#print(starttime)
#print(endtime)
