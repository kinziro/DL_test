import numpy as np
# import gym, roboschool
import gym, pybullet_envs
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
# import tensorflow as tf
# tf.Session(config=tf.ConfigProto(device_count = {'GPU': 2}))

from pybullet_envs.robot_bases import Joint, BodyPart
import pybullet
import pybullet_data


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
        self.action_name = []
        self.status_name = ['z-self.initial_z', 'np.sin_self.angel_to_target', 'np.cos_self.angel_to_target',
                            '0.3vx', '0.3vy', '0.3vz', 'r', 'p']

        for jnt in self.env.ordered_joints:
            self.status_name.append('{}_p'.format(jnt.joint_name))
            self.status_name.append('{}_v'.format(jnt.joint_name))
            self.action_name.append(jnt.joint_name)

        self.status_name.extend(['feet_contact_0_r', 'feet_contact_1_l'])
        np.save('./labels', np.array(self.status_name))
        self.origin_state = None

        # 各座標取得用のインスタンスを格納
        self.parts = {}

        self.right_bodies_name = ['torso', 'thigh', 'leg', 'foot']
        self.left_bodies_name = ['torso', 'thigh_left', 'leg_left', 'foot_left']
        self.right_joints_name = ['thigh_joint', 'leg_joint', 'foot_joint']
        self.left_joints_name = ['thigh_left_joint', 'leg_left_joint', 'foot_left_joint']
        self.right_parts_name = ['torso', 'thigh_joint', 'thigh', 'leg_joint', 'leg',
                                 'foot_joint', 'foot']

        self.left_parts_name = ['torso', 'thigh_left_joint', 'thigh_left', 'leg_left_joint', 'leg_left',
                                'foot_left_joint', 'foot_left']
        self.bodies_name = ['torso', 'thigh', 'leg', 'foot', 'torso', 'thigh_left', 'leg_left', 'foot_left']

        self.joints_name = ['thigh_joint', 'leg_joint', 'foot_joint',
                            'thigh_left_joint', 'leg_left_joint', 'foot_left_joint']

        self.parts_name = ['torso', 'thigh_joint', 'thigh', 'leg_joint', 'leg', 'foot_joint', 'foot',
                           'thigh_left_joint', 'thigh_left', 'leg_left_joint', 'leg_left', 'foot_left_joint',
                           'foot_left']

        for j in range(self.env.env._p.getNumJoints(1)):
            jointInfo = self.env.env._p.getJointInfo(1, j)
            joint_name = jointInfo[1].decode("utf8")
            part_name = jointInfo[12].decode("utf8")

            if part_name in self.bodies_name:
                self.parts[part_name] = self.env.env.parts[part_name]
            if joint_name in self.joints_name:
                self.parts[joint_name] = self.env.env.parts[part_name]

    def get_right_parts_posi(self):
        '''
        右のPartsの座標をshape(3, part数)で返す

        :return:
        '''
        d = []
        for name in self.right_parts_name:
            d.append(self.parts[name].current_position())

        return np.array(d).T

    def get_left_parts_posi(self):
        '''
        左のPartsの座標をshape(3, part数)で返す

        :return:
        '''

        d = []
        for name in self.left_parts_name:
            d.append(self.parts[name].current_position())

        return np.array(d).T

    def get_parts_posi(self):
        '''
        Partsの座標をshape(3, part数)で返す

        :return:
        '''
        d = []
        for name in self.parts_name:
            d.append(self.parts[name].current_position())

        return np.array(d).T

    def get_motion_angle(self):
        d = self.get_parts_posi()
        d_dict = {}

        for i in range(len(self.parts_name)):
            d_dict[self.parts_name[i]] = [d[0, i], d[2, i]]

        angles = []
        angle_labels = ['foot_r', 'shank_r', 'thigh_r', 'thunk', 'foot_l', 'shank_l', 'thigh_l']
        pairs = [['foot_joint', 'foot'], ['leg_joint', 'foot_joint'], ['thigh_joint', 'leg_joint'],
                 ['torso', 'thigh_joint'],
                 ['foot_left_joint', 'foot_left'], ['leg_left_joint', 'foot_left_joint'],
                 ['thigh_left_joint', 'leg_left_joint']]

        for l, p in zip(angle_labels, pairs):
            if 'foot' in l:
                angle = self.cal_angle(d_dict[p[0]], d_dict[p[1]], axis=0)
            else:
                angle = self.cal_angle(d_dict[p[0]], d_dict[p[1]], axis=1)
            angles.append(angle)

        return angles, angle_labels

    def cal_angle(self, point1, point2, axis):
        '''
        引数の線分と垂直軸のなす角を計算して返す
        '''
        if axis == 1:
            if point1[1] >= point2[1]:
                point1_x = point1[0]
                point1_y = point1[1]
                point2_x = point2[0]
                point2_y = point2[1]
            else:
                point1_x = point2[0]
                point1_y = point2[1]
                point2_x = point1[0]
                point2_y = point1[1]
            angle = np.arctan((point1_x - point2_x) / (point1_y - point2_y))

        elif axis == 0:
            if point1[0] >= point2[0]:
                point1_x = point1[0]
                point1_y = point1[1]
                point2_x = point2[0]
                point2_y = point2[1]
            else:
                point1_x = point2[0]
                point1_y = point2[1]
                point2_x = point1[0]
                point2_y = point1[1]
            angle = -1 * np.arctan((point1_y - point2_y) / (point1_x - point2_x))
        return angle

    def step(self, a):

        '''
        instance = self.env.ordered_joints[0]

        for method in dir(instance):
            if callable(getattr(instance, str(method))):
                print(method)
        '''
        # origin_value = np.array([j2.current_position() for j2 in self.env.ordered_joints], dtype=np.float32).flatten()
        obs, reward, terminal, info = self.env.step(a)
        # self.origin_state = np.concatenate([origin_value[0::2].reshape(-1, 1), obs[20:].reshape(-1, 1)], axis=0)

        return obs, reward, terminal, info


def plot_ts(ts, labels, title, plotdir, sampling=0.0166):
    # 角度をプロット
    fig_num = ts.shape[0]
    x = np.arange(ts.shape[1]) * sampling
    plt.clf()
    fig = plt.figure(figsize=(len(x) / 20 + 5, fig_num * 2.5))
    #title = 'angle'
    #plt.title(title)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        ax1.plot(x, ts[i, :])
        ax1.grid()
        ax1.set_ylabel(labels[i])

    plt.savefig('{}{}.png'.format(plotdir, title))

def plot_ts2(ts, labels, title, plotdir, sampling=0.0166):
    # 角度をプロット
    fig_num = ts.shape[0]
    x = np.arange(ts.shape[1]) * sampling
    plt.clf()
    fig = plt.figure(figsize=(len(x) / 2.5 + 5, fig_num * 2.5))
    #title = 'angle'
    #plt.title(title)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        ax1.plot(x, ts[i, :])
        ax1.grid()
        ax1.set_ylabel(labels[i])
        ax1.set_xticks(list(range(0, 501, 1)))

    plt.savefig('{}{}.png'.format(plotdir, title))



# 学習設定
# env_name = 'RoboschoolHumanoid-v1'
# env_name = 'RoboschoolWalker2d-v1'
env_name = 'Walker2DBulletEnv-v0'

# num_cpu = 1         # 学習に使用するCPU数
ori_env = gym.make(env_name)
env = DummyVecEnv([lambda: ori_env])
# env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])

env.reset()
modeldir = './stable_baselines/{}/1e7_2/'.format(env_name)
test_env = gym.make(env_name)
test_env = MyRemodelendEnv(test_env)
# test_env = DummyVecEnv([lambda: test_env])

savedir = './synergy/agent/1e7/'
os.makedirs(savedir, exist_ok=True)
step_len = 500
data = None
right_x = None
right_z = None
left_x = None
left_z = None
angle_t = None
obs_t = None
action_t = None
parts_z_t = None

# 学習結果の確認
model = PPO2.load('{}ppo2_model'.format(modeldir))
from gym import wrappers
video_path = '{}video'.format(savedir)
wrap_env = wrappers.Monitor(test_env, video_path, force=True)
# wrap_env = wrappers.Monitor(ori_env, video_path, force=True)

done = False
# obs = env.reset()
# obs = test_env.reset()
obs = wrap_env.reset()

for step in range(step_len):
    if step % 10 == 0: print("step :", step)
    if done:
        time.sleep(1)
        # o = test_env.reset()
        o = wrap_env.reset()

        break

    action, _states = model.predict(obs)
    # obs, rewards, done, info = test_env.step(action)
    obs, rewards, done, info = wrap_env.step(action)

    # データの加工
    parts_posi = test_env.get_parts_posi()

    angles, angle_labels = test_env.get_motion_angle()
    angles = np.array(angles).reshape(-1, 1)
    obses = np.array(obs).reshape(-1, 1)
    parts_z = np.array(parts_posi[2, :]).reshape(-1 ,1)
    actions = np.array(action).reshape(-1, 1)
    if angle_t is None:
        angle_t = angles
        obs_t = obses
        parts_z_t = parts_z
        action_t = actions
    else:
        angle_t = np.hstack([angle_t, angles])
        obs_t = np.hstack([obs_t, obses])
        parts_z_t = np.hstack([parts_z_t, parts_z])
        action_t = np.hstack([action_t, actions])

    if step % 20 == 0:
        body_posi = test_env.get_right_parts_posi()
        body_posi_l = test_env.get_left_parts_posi()
        # body_posi = test_env.get_right_parts_posi()
        # body_posi_l = test_env.get_left_parts_posi()

        if right_x is None:
            right_x = np.array(body_posi[0, :]).reshape(-1, 1)
            right_z = np.array(body_posi[2, :]).reshape(-1, 1)
            left_x = np.array(body_posi_l[0, :]).reshape(-1, 1)
            left_z = np.array(body_posi_l[2, :]).reshape(-1, 1)
        else:
            right_x = np.hstack([right_x, np.array(body_posi[0, :]).reshape(-1, 1)])
            right_z = np.hstack([right_z, np.array(body_posi[2, :]).reshape(-1, 1)])
            left_x = np.hstack([left_x, np.array(body_posi_l[0, :]).reshape(-1, 1)])
            left_z = np.hstack([left_z, np.array(body_posi_l[2, :]).reshape(-1, 1)])

wrap_env.close()

# パーツの座標をプロット
fig = plt.figure(figsize=(10, 2))
for i in range(right_x.shape[1]):
    plt.plot(left_x[:, i], left_z[:, i], marker='x')
    plt.plot(right_x[:, i], right_z[:, i], marker='o')

plt.grid()
title = 'walker'
plt.savefig('{}{}.png'.format(savedir, title))
# np.save('./walker2d_status', np.array(data))
# np.save('./humanoid_status', np.array(data))
# labels = np.load('./labels.npy')

np.save('{}angle_t'.format(savedir), angle_t)
np.save('{}obs_t'.format(savedir), obs_t)
np.save('{}parts_z_t'.format(savedir), parts_z_t)
np.save('{}action_t'.format(savedir), action_t)
#angle_labels.extend(['feet_contact_r', 'feet_contact_l', 'foot_joint_z', 'foot_z', 'foot_left_joint_z', 'foot_left_z'])

# 角度をプロット
plot_ts(ts=angle_t, labels=angle_labels, title='angle', plotdir=savedir)
# パーツのz座標をプロット
plot_ts(ts=parts_z_t, labels=test_env.parts_name, title='parts_z', plotdir=savedir)
# 行動をプロット
plot_ts(ts=action_t, labels=test_env.action_name, title='actions', plotdir=savedir)
# 状態をプロット
plot_ts(ts=obs_t, labels=test_env.status_name, title='status', plotdir=savedir)

# 右足歩行開始点確認用プロット
right_t = np.vstack([parts_z_t[5:7, :], obs_t[20:22, :]])
right_labels = test_env.parts_name[5:7]
right_labels.extend(['contact_r', 'contact_l'])
plot_ts2(ts=right_t, labels=right_labels, title='right_data', plotdir=savedir, sampling=1)

'''
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

env.close()
test_env.close()
ori_env.close()

# print(starttime)
# print(endtime)