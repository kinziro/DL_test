import c3d
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal
from scipy import fftpack
from sklearn.decomposition import NMF
import math
from scipy.linalg import svd


class DataBox:
    def __init__(self, ts, rank, name):
        self.name = name
        self.rank = rank
        self.ts_ori = ts
        self.ts_ave = np.average(ts, axis=1).reshape(-1, 1)
        self.ts_adj = self.ts_ori - self.ts_ave
        self.x = np.arange(0, 1, 1/self.ts_adj.shape[1])
        self.U, self.S, self.Vh = svd(self.ts_adj)
        lambda_d = np.diag(self.S[:rank])
        self.lambda_Vh = np.dot(lambda_d, self.Vh[:rank, :])

    def contribution(self):
        S_sum = np.sum(self.S)
        ret = []
        for i in range(1, self.rank+1):
            ret.append(np.sum(self.S[:i])/S_sum)

        return ret

def get_walking_start(ts, ind1, ind2):
    min_d1 = 100
    min_d2 = 100
    p_d1 = 100
    p_d2 = 100
    ignore = 10
    ret = []
    for i, (d1, d2) in enumerate(zip(ts[ind1, :], ts[ind2, :])):
        if ignore > 0:
            ignore -= 1
            continue

        if d1 < min_d1:
            min_d1 = d1
        if d2 < min_d2:
            min_d2 = d2

        if p_d1 == min_d1 and p_d2 == min_d2:
            if d1 > p_d1 or d2 > p_d2:
                ret.append(i)
                min_d1 = 100
                min_d2 = 100
                ignore = 10

        p_d1 = d1
        p_d2 = d2

    return ret


basedir = './synergy/'
human_flag = True
agent_flag = True
num_of_synergy = 3

# 人間
if human_flag:
    humanbasedir = '{}human/'.format(basedir)
    hns = ['ActorA_Walk', 'ActorE_Walk', 'ActorG_Walk', 'ActorI_Walk']
    hA_cut_list = [[10, 40],
                  [40, 69]]
    hE_cut_list = [[7, 31],
                   [31, 54]]
    hG_cut_list = [[10, 36],
                   [36, 64]]
    hI_cut_list = [[7, 31],
                   [31, 54]]

    human_d_list = []
    for hn, h_cut_list in zip(hns, [hA_cut_list, hE_cut_list, hG_cut_list, hI_cut_list]):
        human_d = np.load('{}{}/angle_t.npy'.format(humanbasedir, hn))

        for i, c in enumerate(h_cut_list):
            ts = human_d[:7, c[0]:c[1]]
            human_d_list.append(DataBox(ts, rank=4, name='{}_{}'.format(hn, i+1)))

# エージェント
if agent_flag:
    datadir = '{}agent/'.format(basedir)
    a_cut_list = [[110, 166],
                  [165, 219],
                  [218, 281],
                  [280, 339]]
    agent_d = np.load('{}angle_t.npy'.format(datadir))
    z_d = np.load('{}parts_z_t.npy'.format(datadir))
    start_point = get_walking_start(ts=z_d, ind1=5, ind2=6)

    agent_r = agent_d[9, :]
    agent_d_list = []
    for i, c in enumerate(a_cut_list):
        ts = agent_d[:7, c[0]:c[1]]
        agent_d_list.append(DataBox(ts, rank=4, name='agent_{}'.format(i+1)))


# plot
angle_labels = ['foot_r', 'shank_r', 'thigh_r', 'thunk', 'foot_l', 'shank_l', 'thigh_l']

savedir = '{}kinematic/'.format(basedir)
os.makedirs(savedir, exist_ok=True)

# 人とエージェントの比較
if human_flag and agent_flag:
    # シナジー 空間
    fig_num = 3
    fig = plt.figure(figsize=(10, fig_num * 3))
    title = 'U'
    width = 0.3
    x_h = np.arange(len(angle_labels))
    x_a = np.arange(len(angle_labels)) + width
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        ax1.bar(x_h, human_d_list[0].U[:, i], tick_label=angle_labels, width=width, align="center", label='human')
        ax1.bar(x_a, agent_d_list[0].U[:, i], tick_label=angle_labels, width=width, align="center", label='agent')
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel(str(i))
    plt.savefig('{}{}.png'.format(savedir, title))

    # シナジー 時間
    x_h = human_d_list[0].x
    x_a = agent_d_list[0].x

    fig_num = 3
    fig = plt.figure(figsize=(10, fig_num * 3))
    title = 'lambda_Vh'
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        ax1.plot(x_h, human_d_list[0].lambda_Vh[i, :], label='human')
        ax1.plot(x_a, agent_d_list[0].lambda_Vh[i, :], label='agent')
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel(str(i))
    plt.savefig('{}{}.png'.format(savedir, title))


    # 角度をプロット
    fig_num = 7
    fig = plt.figure(figsize=(10, fig_num * 2.5))
    title = 'angle'
    #plt.title(title)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        ax1.plot(x_h, human_d_list[0].ts_ori[i, :], label='human')
        ax1.plot(x_a, agent_d_list[0].ts_ori[i, :], label='agent')
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel(angle_labels[i])
    plt.savefig('{}{}.png'.format(savedir, title))


# ヒューマンの比較
d_lists = []
d_labels = []
coeffs = []
if human_flag:
    d_lists.append(human_d_list)
    d_labels.append('human')

    coeffs.append([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, -1, -1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                  ])
if agent_flag:
    d_lists.append(agent_d_list)
    d_labels.append('agent')

for d_list, d_label, coeff in zip(d_lists, d_labels, coeffs):
    # シナジー 空間
    fig_num = num_of_synergy
    fig = plt.figure(figsize=(10, fig_num * 3))
    plt.subplots_adjust(right=0.75)
    title = '{}_U'.format(d_label)
    width = 0.8 / len(d_list)
    xs = []
    for i in range(len(d_list)):
        xs.append(np.arange(len(angle_labels)) + width*i)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        for j, (x, db) in enumerate(zip(xs, d_list)):
            ax1.bar(x, coeff[i][j]*db.U[:, i], tick_label=angle_labels, width=width, align="center", label=db.name)
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1.05, 1))
        ax1.set_ylabel('synergy {}'.format(i+1))
    plt.savefig('{}{}.png'.format(savedir, title))

    # シナジー 時間
    fig_num = 3
    fig = plt.figure(figsize=(10, fig_num * 3))
    plt.subplots_adjust(right=0.75)
    title = '{}_lambda_Vh'.format(d_label)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        for j, db in enumerate(d_list):
            ax1.plot(db.x, coeff[i][j]*db.lambda_Vh[i, :], label=db.name)
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1.05, 1))
        ax1.set_ylabel('synergy {}'.format(i + 1))
    plt.savefig('{}{}.png'.format(savedir, title))

    # 角度をプロット
    fig_num = 7
    fig = plt.figure(figsize=(10, fig_num * 2.5))
    plt.subplots_adjust(right=0.75)
    title = '{}_angle'.format(d_label)
    #plt.title(title)
    fig.suptitle(title)
    for i in range(fig_num):
        ax1 = fig.add_subplot(fig_num, 1, i+1)
        for j, db in enumerate(d_list):
            ax1.plot(db.x, db.ts_ori[i, :], label=db.name)
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1.05, 1))
        ax1.set_ylabel(angle_labels[i])
    plt.savefig('{}{}.png'.format(savedir, title))

'''
# エージェント間比較
coeff = [[-1, -1, 1, 1],
         [1, -1, 1, 1],
         [1, -1, -1, 1]
         ]

# シナジー 空間
fig_num = 3
fig = plt.figure(figsize=(10, fig_num * 3))
title = 'agent_U'
width = 0.8/len(agent_d_list)
xs = []
for i in range(len(agent_d_list)):
    xs.append(np.arange(len(angle_labels)) + width*i)
fig.suptitle(title)
for i in range(fig_num):
    ax1 = fig.add_subplot(fig_num, 1, i+1)
    for j, (x, db) in enumerate(zip(xs, agent_d_list)):
        ax1.bar(x, coeff[i][j]*db.U[:, i], tick_label=angle_labels, width=width, align="center", label=db.name)
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel(str(i))
plt.savefig('{}{}.png'.format(savedir, title))


# シナジー 時間
fig_num = 3
fig = plt.figure(figsize=(10, fig_num * 3))
title = 'agent_lambda_Vh'
fig.suptitle(title)
for i in range(fig_num):
    ax1 = fig.add_subplot(fig_num, 1, i+1)
    for j, db in enumerate(agent_d_list):
        ax1.plot(db.x, coeff[i][j]*db.lambda_Vh[i, :], label=db.name)
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel(str(i))
plt.savefig('{}{}.png'.format(savedir, title))


# 角度をプロット
fig_num = 7
fig = plt.figure(figsize=(10, fig_num * 2.5))
title = 'agent_angle'
#plt.title(title)
fig.suptitle(title)
for i in range(fig_num):
    ax1 = fig.add_subplot(fig_num, 1, i+1)
    for db in agent_d_list:
        ax1.plot(db.x, db.ts_ori[i, :], label=db.name)
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel(angle_labels[i])
plt.savefig('{}{}.png'.format(savedir, title))


print('-- human --')
for i in range(2):
    print(human_d_list[i].contribution())
print('-- agent --')
for i in range(4):
    print(agent_d_list[i].contribution())

#plt.savefig('{}{}.png'.format(plotdir, title))


def synergy_plot(angle, title):
    thunk_x = np.sin(angle[3, :]).reshape(1, -1)
    thunk_y = np.cos(angle[3, :]).reshape(1, -1)

    root_x = np.zeros(thunk_x.shape[1]).reshape(1, -1)
    root_y = np.zeros(thunk_y.shape[1]).reshape(1, -1)

    thigh_x = -1*np.sin(angle[2, :]).reshape(1, -1)
    thigh_y = -1*np.cos(angle[2, :]).reshape(1, -1)

    shank_x = -1*np.sin(angle[1, :]).reshape(1, -1) + thigh_x
    shank_y = -1*np.cos(angle[1, :]).reshape(1, -1) + thigh_y

    foot_x = 0.5*np.cos(angle[0, :]).reshape(1, -1) + shank_x
    foot_y = -0.5*np.sin(angle[0, :]).reshape(1, -1) + shank_y

    thigh_l_x = -1*np.sin(angle[6, :]).reshape(1, -1)
    thigh_l_y = -1*np.cos(angle[6, :]).reshape(1, -1)

    shank_l_x = -1*np.sin(angle[5, :]).reshape(1, -1) + thigh_l_x
    shank_l_y = -1*np.cos(angle[5, :]).reshape(1, -1) + thigh_l_y

    foot_l_x = 0.5*np.cos(angle[4, :]).reshape(1, -1) + shank_l_x
    foot_l_y = -0.5*np.sin(angle[4, :]).reshape(1, -1) + shank_l_y


    x = np.vstack([thunk_x, root_x, thigh_x, shank_x, foot_x])
    y = np.vstack([thunk_y, root_y, thigh_y, shank_y, foot_y])

    x_l = np.vstack([thunk_x, root_x, thigh_l_x, shank_l_x, foot_l_x])
    y_l = np.vstack([thunk_y, root_y, thigh_l_y, shank_l_y, foot_l_y])

    print(thunk_x.shape[1])
    fig = plt.figure(figsize=(thunk_x.shape[1] * 2.5, 10))
    for i in range(thunk_x.shape[1]):
        plt.plot(x_l[:, i] + i, y_l[:, i], color='r', marker='o')
        plt.plot(x[:, i] + i, y[:, i], color='b', marker='o')
    plt.title(title)
    plt.grid()
    plt.savefig('{}{}.png'.format(savedir, title))


angle_t = np.dot(human_d_list[0].U[:, 0:1], human_d_list[0].lambda_Vh[0:1, :])
synergy_plot(angle_t, 'human_synergy1')
angle_t = np.dot(human_d_list[0].U[:, 1:2], human_d_list[0].lambda_Vh[1:2, :])
synergy_plot(angle_t, 'human_synergy2')
angle_t = np.dot(human_d_list[0].U[:, 2:3], human_d_list[0].lambda_Vh[2:3, :])
synergy_plot(angle_t, 'human_synergy3')

angle_t = np.dot(agent_d_list[0].U[:, 0:1], agent_d_list[0].lambda_Vh[0:1, :])
synergy_plot(angle_t, 'agent_synergy1')
angle_t = np.dot(agent_d_list[0].U[:, 1:2], agent_d_list[0].lambda_Vh[1:2, :])
synergy_plot(angle_t, 'agent_synergy2')
angle_t = np.dot(agent_d_list[0].U[:, 2:3], agent_d_list[0].lambda_Vh[2:3, :])
synergy_plot(angle_t, 'agent_synergy3')
'''

#plt.show()
