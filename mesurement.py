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
        self.env.reset()
        self.status_name = ['z-self.initial_z', 'np.sin_self.angel_to_target', 'np.cos_self.angel_to_target',
                            '0.3vx', '0.3vy', '0.3vz', 'r', 'p']

        for jnt in self.env.ordered_joints:
            self.status_name.append('{}_p'.format(jnt.name))
            self.status_name.append('{}_v'.format(jnt.name))

        self.status_name.extend(['feet_contact_0_r', 'feet_contact_1_l'])
        np.save('./labels', np.array(self.status_name))

        self.origin_state = None

    def step(self, a):
        '''
        instance = self.env.ordered_joints[0]

        for method in dir(instance):
            if callable(getattr(instance, str(method))):
                print(method)
        '''
        '''
        joints = self.env.ordered_joints
        aa = joints[0].current_position()
        bb = joints[0].current_relative_position()
        #lowerLimit = joints[0].lowerLimit
        #upperLimit = joints[0].upperLimit
        #pos_mid = 0.5 * (lowerLimit + upperLimit)

        origin_value = np.array([j2.current_position() for j2 in self.env.ordered_joints], dtype=np.float32).flatten()
        origin_value_relative = np.array([j2.current_relative_position() for j2 in self.env.ordered_joints], dtype=np.float32).flatten()
        angle_value = origin_value[0::2]
        norm_value = []
        for i in range(len(angle_value)):
            if i == 2 or i == 5:
                lowerLimit = -45
                upperLimit = 45
            else:
                lowerLimit = -150
                upperLimit = 0
            pos_mid = 0.5 * (lowerLimit + upperLimit)
            norm_value.append(2 * (angle_value[i] - pos_mid) / (upperLimit - lowerLimit))
        print(origin_value[0::2])
        print(origin_value_relative[0::2])
        print(norm_value)
        #origin_value = [0]*34
        '''
        origin_value = np.array([j2.current_position() for j2 in self.env.ordered_joints], dtype=np.float32).flatten()
        obs, reward, terminal, info = self.env.step(a)
        a = origin_value[0::2].reshape(1, -1)
        b = obs[20:].reshape(1, -1)
        self.origin_state = np.concatenate([origin_value[0::2].reshape(-1, 1), obs[20:].reshape(-1, 1)], axis=0)

        return obs, reward, terminal, info


#import myenv
#test_env = gym.make('myhumanoid-v0')


# 学習設定
train = False       # 学習をするかどうか
validation = True   # 学習結果を使って評価をするかどうか

#env_name = 'RoboschoolHumanoid-v1'
env_name = 'RoboschoolWalker2d-v1'
num_cpu = 1         # 学習に使用するCPU数
learn_timesteps = 10**3     # 学習タイムステップ

ori_env = gym.make(env_name)
env = DummyVecEnv([lambda: ori_env])
#env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])
env.reset()
#env.render()
#time.sleep(5)

savedir = './stable_baselines/{}/'.format(env_name)
#logdir = '{}tensorboard_log/'.format(savedir)
#os.makedirs(savedir, exist_ok=True)


#env_name = 'RoboschoolHumanoid-v1'
test_env = gym.make(env_name)
test_env = MyRemodelendEnv(test_env)
#test_env = DummyVecEnv([lambda: test_env])

plotdir = './testplot/'
os.makedirs(plotdir, exist_ok=True)
data = None
step_len = 300
# 学習結果の確認
if validation:
    model = PPO2.load('{}ppo2_model'.format(savedir))
    from gym import wrappers

    video_path = '{}video'.format(plotdir)
    wrap_env = wrappers.Monitor(test_env, video_path, force=True)
    #wrap_env = wrappers.Monitor(ori_env, video_path, force=True)

    done = False
    #obs = env.reset()
    #obs = test_env.reset()
    obs = wrap_env.reset()

    for step in range(step_len):
        if step % 10 == 0: print("step :", step)
        if done:
            time.sleep(1)
            #o = test_env.reset()
            o = wrap_env.reset()
            break

        action, _states = model.predict(obs)
        #obs, rewards, done, info = test_env.step(action)
        obs, rewards, done, info = wrap_env.step(action)

        ori_state = wrap_env.origin_state
        #ori_state = test_env.envs[0].origin_state
        if data is None:
            data = np.array(ori_state).reshape(-1, 1)
        else:
            data = np.hstack([data, np.array(ori_state).reshape(-1, 1)])

        if 100 < step < 105:
            print('--------')
            print(obs[8:20:2])
            print(ori_state.T)

        '''
        if data is None:
            data = np.array(obs).reshape(-1, 1)
        else:
            data = np.hstack([data, np.array(obs).reshape(-1, 1)])
        '''
        #print(step, obs.shape, data.shape)
    #wrap_env.close()
    print(data.shape)

    np.save('./walker2d_status', np.array(data))
    #np.save('./humanoid_status', np.array(data))
    labels = np.load('./labels.npy')

    fig_num = 8
    x = np.arange(data.shape[1])*0.0166
    plt.clf()
    fig = plt.figure(figsize=(len(x)/20, fig_num*2.5))

    title = 'angle'
    plt.title(title)
    ax1 = fig.add_subplot(fig_num, 1, 1)
    ax1.plot(x, data[0, :])
    ax1.grid()
    ax1.set_ylabel(labels[8])
    ax2 = fig.add_subplot(fig_num, 1, 2)
    ax2.plot(x, data[1, :])
    ax2.grid()
    ax2.set_ylabel(labels[10])
    ax3 = fig.add_subplot(fig_num, 1, 3)
    ax3.plot(x, data[2, :])
    ax3.grid()
    ax3.set_ylabel(labels[12])
    ax4 = fig.add_subplot(fig_num, 1, 4)
    ax4.plot(x, data[3, :])
    ax4.grid()
    ax4.set_ylabel(labels[14])
    ax5 = fig.add_subplot(fig_num, 1, 5)
    ax5.plot(x, data[4, :])
    ax5.grid()
    ax5.set_ylabel(labels[16])
    ax6 = fig.add_subplot(fig_num, 1, 6)
    ax6.plot(x, data[5, :])
    ax6.grid()
    ax6.set_ylabel(labels[18])
    ax7 = fig.add_subplot(fig_num, 1, 7)
    ax7.plot(x, data[6, :])
    ax7.grid()
    ax7.set_ylabel(labels[20])
    ax8 = fig.add_subplot(fig_num, 1, 8)
    ax8.plot(x, data[7, :])
    ax8.grid()
    ax8.set_ylabel(labels[21])
    plt.savefig('{}{}.png'.format(plotdir, title))

    '''
    fig_num = 7
    x = np.arange(data.shape[1])*0.015
    plt.clf()
    fig = plt.figure(figsize=(len(x)/20, fig_num*2.5))

    title = 'angle'
    plt.title(title)
    ax1 = fig.add_subplot(fig_num, 1, 1)
    ax1.plot(x, data[10, :])
    ax1.grid()
    ax1.set_ylabel(labels[10])
    ax2 = fig.add_subplot(fig_num, 1, 2)
    ax2.plot(x, data[18, :])
    ax2.grid()
    ax2.set_ylabel(labels[18])
    ax3 = fig.add_subplot(fig_num, 1, 3)
    ax3.plot(x, data[20, :])
    ax3.grid()
    ax3.set_ylabel(labels[20])
    ax4 = fig.add_subplot(fig_num, 1, 4)
    ax4.plot(x, data[26, :])
    ax4.grid()
    ax4.set_ylabel(labels[26])
    ax5 = fig.add_subplot(fig_num, 1, 5)
    ax5.plot(x, data[28, :])
    ax5.grid()
    ax5.set_ylabel(labels[28])
    ax6 = fig.add_subplot(fig_num, 1, 6)
    ax6.plot(x, data[42, :])
    ax6.grid()
    ax6.set_ylabel(labels[42])
    ax7 = fig.add_subplot(fig_num, 1, 7)
    ax7.plot(x, data[43, :])
    ax7.grid()
    ax7.set_ylabel(labels[43])
    plt.savefig('{}{}.png'.format(plotdir, title))
    '''


env.close()
test_env.close()
ori_env.close()

#print(starttime)
#print(endtime)
