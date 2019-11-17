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


class Point:
    def __init__(self, poses, names):
        self.poses = poses
        self.point_names = names
        self.n2p = {}
        for i in range(len(names)):
            self.n2p[names[i]] = [self.poses[0, i], self.poses[1, i]]


class Body:
    def __init__(self, parts, angles):
        self.parts = np.array(parts)
        self.angles = np.array(angles)
        self.p2a = {}
        for p, a in zip(parts, angles):
            self.p2a[p] = a
        self.points = {}
        self.poses = {}
        self.point_names = {}

    def cal_pose(self, name, order, point_names, baseline, axis):
        poses = []
        poses.append(baseline[0])
        poses.append(baseline[1])

        top_x = baseline[0][0]
        top_y = baseline[0][1]
        base_x = baseline[1][0]
        base_y = baseline[1][1]

        b_x = top_x-base_x
        b_y = top_y-base_y
        if b_x == 0:
            if b_y >= 0:
                base_a = np.pi/2
            else:
                base_a = -1*np.pi/2
        else:
            base_a = np.arctan(b_y/b_x)

        if axis == -1:
            base_a += np.pi

        for o in order:
            angle = self.p2a[o]+base_a
            pose_x = np.cos(angle) + base_x
            pose_y = np.sin(angle) + base_y

            poses.append([pose_x, pose_y])

            base_a = angle
            base_x = pose_x
            base_y = pose_y

        poses = np.array(poses).T
        #self.poses[name] = poses
        self.points[name] = Point(poses=poses, names=point_names)
        return self.points[name]

    def add_angle(self, angle):
        self.angles = self.angles + angle
        self.p2a = {}
        for p, a in zip(self.parts, self.angles):
            self.p2a[p] = a


data = np.load('./walker2d_status.npy')

savedir = './testplot/'


fig = plt.figure(figsize=(30, 8))
for i in range(30):
    bd = Body(parts=['thigh', 'leg', 'foot', 'thigh_l', 'leg_l', 'foot_l'],
              angles=[data[0, i], data[1, i], data[2, i]+np.pi/2, data[3, i], data[4, i], data[5, i]+np.pi/2])

    poses_l = bd.cal_pose(name='left', order=['thigh_l', 'leg_l', 'foot_l'],
                          point_names=['torso', 'thigh', 'leg', 'foot', 'toe'],
                          baseline=[[i, 1], [i, 0]], axis=-1).poses
    poses_r = bd.cal_pose(name='right', order=['thigh', 'leg', 'foot'],
                          point_names=['torso', 'thigh', 'leg', 'foot', 'toe'],
                          baseline=[[i, 1], [i, 0]], axis=-1).poses

    if data[6, i] == 1:
        x_diff = poses_r[0, 4] - poses_r[0, 3]
        y_diff = poses_r[1, 4] - poses_r[1, 3]
    else:
        x_diff = poses_l[0, 4] - poses_l[0, 3]
        y_diff = poses_l[1, 4] - poses_l[1, 3]

    #correction_a = -1 * np.arctan(y_diff/x_diff)
    #bd.add_angle(correction_a)

    #poses_l = bd.cal_pose(name='left', order=reversed(['thigh_l', 'leg_l', 'foot_l']), baseline=[[i+1, 0], [i, 0]], axis=-1)
    #poses_r = bd.cal_pose(name='right', order=reversed(['thigh', 'leg', 'foot']), baseline=[[i+1, 0], [i, 0]], axis=-1)

    x_plot_l = poses_l[0, :]
    y_plot_l = poses_l[1, :]
    plt.plot(x_plot_l, y_plot_l, marker='o', color='r')

    x_plot = poses_r[0, :]
    y_plot = poses_r[1, :]
    plt.plot(x_plot, y_plot, marker='o', color='b')

title = 'walking'
plt.grid()
plt.xlim([-4, 30 + 5])
plt.ylim([-4, 4])
plt.savefig('{}{}.png'.format(savedir, title))


"""

step_len = 300
pitch = 10
fig = plt.figure(figsize=(step_len/pitch, 8))
for i in range(0, step_len, pitch):
    foot = data[2, i] + np.pi/2 + np.pi
    leg = data[1, i] + foot
    thigh = data[0, i] + leg
    thigh_l = data[3, i] + thigh

    foot_x = i/pitch*1
    foot_y = 0
    toe_x = 0.5 + foot_x
    toe_y = foot_y
    knee_x = np.cos(-1*foot) + foot_x
    knee_y = np.sin(-1*foot) + foot_y
    thigh_x = np.cos(-1*leg) + knee_x
    thigh_y = np.sin(-1*leg) + knee_y
    torso_x = np.cos(-1*thigh) + thigh_x
    torso_y = np.sin(-1*thigh) + thigh_y
    knee_l_x = np.cos(-1*thigh_l) + thigh_x
    knee_l_y = np.sin(-1*thigh_l) + thigh_y

    '''
    torso = np.pi/2
    thigh = data[0, i] + np.pi + torso
    leg = data[1, i] + np.pi + thigh + np.pi
    foot = data[2, i] + np.pi + leg + np.pi + np.pi/2
    
    torso_x = i*1
    torso_y = 0
    thigh_x = np.cos(thigh) + torso_x
    thigh_y = np.sin(thigh) + torso_y
    leg_x = np.cos(leg) + thigh_x
    leg_y = np.sin(leg) + thigh_y
    foot_x = 1/2*np.cos(foot) + leg_x
    foot_y = 1/2*np.sin(foot) + leg_y
    '''

    x_plot = [torso_x, thigh_x, knee_x, foot_x, toe_x]
    y_plot = [torso_y, thigh_y, knee_y, foot_y, toe_y]
    plt.plot(x_plot, y_plot, marker='o', color='b')
    x_plot_l = [thigh_x, knee_l_x]
    y_plot_l = [thigh_y, knee_l_y]
    plt.plot(x_plot_l, y_plot_l, marker='o', color='r')
title = 'walking'
plt.grid()
plt.xlim([-1, step_len/pitch + 5])
plt.ylim([-4, 4])
plt.savefig('{}{}.png'.format(savedir, title))

print(data[0, 0], data[1, 0], data[2, 0], data[3, 0], data[4, 0], data[5, 0])
"""
