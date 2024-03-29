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
        self.x = self.x[:self.ts_adj.shape[1]]
        self.U, self.S, self.Vh = svd(self.ts_adj)
        lambda_d = np.diag(self.S[:rank])
        self.lambda_Vh = np.dot(lambda_d, self.Vh[:rank, :])

    def contribution(self):
        S_sum = np.sum(self.S)
        ret = []
        for i in range(1, len(self.S)+1):
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


def cos_sim(v1, v2):
    '''コサイン類似度の計算'''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def to_csv_cos_sim(label, d_list1, coeff1, d_list2=None, coeff2=None):
    '''
    コサイン類似度を計算してcsvに保存する
    :param label:
    :param d_list:
    :param coeff:
    :return:
    '''
    if d_list2 is None or coeff2 is None:
        d_list2 = d_list1
        coeff2 = coeff1

    mins = []
    for rank in range(3):
        index = []
        columns = []
        cos_sims = []
        for i, h1 in enumerate(d_list1):
            index.append(h1.name)
            row_sims = []
            for j, h2 in enumerate(d_list2):
                row_sims.append(cos_sim(coeff1[rank][i]*h1.U[:, rank], coeff2[rank][j]*h2.U[:, rank]))
                if i == 0:
                    columns.append(h2.name)
            cos_sims.append(row_sims)
        cos_sims = np.array(cos_sims)
        pd.DataFrame(cos_sims, index=index, columns=columns).to_csv('{}{}_cos_sim_{}.csv'.format(savedir, label, rank+1))
        mins.append(np.min(np.abs(cos_sims)))

    pd.DataFrame(mins, index=['synergy1', 'synergy2', 'synergy3']).to_csv('{}{}_min_cos_sim.csv'.format(savedir, label))


def plot_synergy(label, d_list, coeff):
    '''
    シナジー関係のプロット
    :param label:
    :param d_list:
    :param coeff:
    :return:
    '''
    # シナジー 空間
    fig_num = num_of_synergy
    fig = plt.figure(figsize=(10, fig_num * 3))
    plt.subplots_adjust(right=0.75)
    title = '{}_U'.format(label)
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
    title = '{}_lambda_Vh'.format(label)
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
    title = '{}_angle'.format(label)
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


# 共通
basedir = './synergy/'
savedir = '{}kinematic/'.format(basedir)
os.makedirs(savedir, exist_ok=True)

angle_labels = ['foot_r', 'shank_r', 'thigh_r', 'thunk', 'foot_l', 'shank_l', 'thigh_l']
human_flag = True
agent_flag = True
num_of_synergy = 3

contributions = []
columns = []
# 人間
if human_flag:
    humanbasedir = '{}human/'.format(basedir)
    hns = ['ActorA_Walk', 'ActorE_Walk', 'ActorG_Walk', 'ActorI_Walk']
    hls = ['humanA', 'humanB', 'humanC', 'humanD']
    hA_cut_list = [[10, 40],
                   [40, 69]]
    hE_cut_list = [[7, 31],
                   [31, 54]]
    hG_cut_list = [[10, 36],
                   [36, 64]]
    hI_cut_list = [[7, 31],
                   [31, 54]]

    human_d_list = []
    for hn, hl, h_cut_list in zip(hns, hls, [hA_cut_list, hE_cut_list, hG_cut_list, hI_cut_list]):
        human_d = np.load('{}{}/angle_t.npy'.format(humanbasedir, hn))

        for i, c in enumerate(h_cut_list):
            ts = human_d[:7, c[0]:c[1]]
            human_d_list.append(DataBox(ts, rank=num_of_synergy, name='{}_{}'.format(hl, i+1)))
            contributions.append(human_d_list[-1].contribution())
            columns.append(human_d_list[-1].name)

    human_coeff = [[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, -1, -1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   ]

# エージェント
if agent_flag:
    datadir = '{}agent/'.format(basedir)

    aA_cut_list = [[64, 117],
                   [117, 168]]
    aB_cut_list = [[59, 110],
                   [110, 162]]
    aC_cut_list = [[52, 111],
                   [111, 162]]
    aD_cut_list = [[60, 108],
                   [108, 158]]
    agent_d_list = []
    for l, a_cut_list in zip(['A', 'B', 'C', 'D'], [aA_cut_list, aB_cut_list, aC_cut_list, aD_cut_list]):
        agent_d = np.load('{}{}/angle_t.npy'.format(datadir, l))

        for i, c in enumerate(a_cut_list):
            ts = agent_d[:7, c[0]:c[1]]
            agent_d_list.append(DataBox(ts, rank=num_of_synergy, name='agent{}_{}'.format(l, i+1)))
            contributions.append(agent_d_list[-1].contribution())
            columns.append(agent_d_list[-1].name)

    agent_coeff = [[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, -1, 1, -1, 1, -1, -1, -1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   ]
# 同一エージェントの長時間
if agent_flag:
    datadir = '{}agent/'.format(basedir)

    aA_long_cut_list = [[69, 122],
                        [122, 174],
                        [174, 226],
                        [226, 287],
                        [287, 338],
                        [338, 393],
                        [393, 452],
                        [452, 497],
                        ]

    agent_long_d_list = []
    agent_long_d = np.load('{}A_long/angle_t.npy'.format(datadir))

    for i, c in enumerate(aA_long_cut_list):
        ts = agent_long_d[:7, c[0]:c[1]]
        agent_long_d_list.append(DataBox(ts, rank=num_of_synergy, name='agentA_{}'.format(i+1)))
        contributions.append(agent_long_d_list[-1].contribution())
        columns.append(agent_long_d_list[-1].name)

    agent_long_coeff = [[1, 1, 1, 1, -1, 1, -1, -1],
                        [-1, 1, -1, -1, -1, -1, 1, 1],
                        [1, -1, 1, 1, 1, 1, 1, 1],
                        ]

# 失敗したエージェント
if agent_flag:
    datadir = '{}agent/'.format(basedir)

    a_FailB_cut_list = [[10, 70],
                        [70, 130],
                        [130, 190],
                        [190, 250],
                        [250, 310],
                        [310, 370],
                        [370, 430],
                        [430, 490],
                        ]

    agent_FailB_d_list = []
    agent_FailB_d = np.load('{}FailB/angle_t.npy'.format(datadir))

    for i, c in enumerate(a_FailB_cut_list):
        ts = agent_FailB_d[:7, c[0]:c[1]]
        agent_FailB_d_list.append(DataBox(ts, rank=num_of_synergy, name='agent_FailB_{}'.format(i+1)))
        contributions.append(agent_FailB_d_list[-1].contribution())
        columns.append(agent_FailB_d_list[-1].name)

    agent_FailB_coeff = [[1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         ]

# 1e7エージェント
if agent_flag:
    datadir = '{}agent/'.format(basedir)

    a_1e7_cut_list = [[6, 35]
                       ]

    agent_1e7_d_list = []
    agent_1e7_d = np.load('{}1e7/angle_t.npy'.format(datadir))

    for i, c in enumerate(a_1e7_cut_list):
        ts = agent_1e7_d[:7, c[0]:c[1]]
        agent_1e7_d_list.append(DataBox(ts, rank=num_of_synergy, name='agent_1e7_{}'.format(i+1)))
        contributions.append(agent_1e7_d_list[-1].contribution())
        columns.append(agent_1e7_d_list[-1].name)

    agent_1e7_coeff = [[1],
                       [-1],
                       [-1],
                       ]




# 寄与度の保存
contributions = np.array(contributions).T
pd.DataFrame(contributions, columns=columns).to_csv('{}contributions.csv'.format(savedir))

# コサイン類似度の計算
to_csv_cos_sim('human', d_list1=human_d_list, coeff1=human_coeff)
to_csv_cos_sim('agent', d_list1=agent_d_list, coeff1=agent_coeff)
to_csv_cos_sim('h_vs_a', d_list1=human_d_list, coeff1=human_coeff, d_list2=agent_d_list, coeff2=agent_coeff)
to_csv_cos_sim('agentA_only', d_list1=agent_long_d_list, coeff1=agent_long_coeff)
to_csv_cos_sim('agent_FailB', d_list1=agent_FailB_d_list, coeff1=agent_FailB_coeff)
to_csv_cos_sim('agent_1e7', d_list1=agent_1e7_d_list, coeff1=agent_1e7_coeff, d_list2=agent_d_list, coeff2=agent_coeff)

# ヒューマン, エージェントそれぞれ同士の比較
mix_d_list = [human_d_list[0]]
mix_d_list.extend(agent_d_list)
mix_coeff = np.array(human_coeff)[:, 0:1]
mix_coeff = np.hstack([mix_coeff, np.array(agent_coeff)])
plot_synergy(label='human', d_list=human_d_list, coeff=human_coeff)
plot_synergy(label='agent', d_list=agent_d_list, coeff=agent_coeff)
plot_synergy(label='h_vs_a', d_list=mix_d_list, coeff=mix_coeff)
plot_synergy(label='agentA_only', d_list=agent_long_d_list, coeff=agent_long_coeff)
plot_synergy(label='human_A', d_list=[human_d_list[0]], coeff=[[1], [1], [1]])
plot_synergy(label='agent_FailB', d_list=agent_FailB_d_list, coeff=agent_FailB_coeff)

mix_d_list = [agent_d_list[0]]
mix_d_list.extend(agent_1e7_d_list)
mix_coeff = np.array(agent_coeff)[:, 0:1]
mix_coeff = np.hstack([mix_coeff, np.array(agent_1e7_coeff)])
plot_synergy(label='8_vs_7', d_list=mix_d_list, coeff=mix_coeff)
