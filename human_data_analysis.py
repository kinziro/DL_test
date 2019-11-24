import c3d
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal
from scipy import fftpack
from sklearn.decomposition import NMF
import math


def cal_angle(point1, point2, axis):
    '''
    引数の線分と指定した軸とのなす角を計算して返す
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


def coordinate_to_angle(coord_dict, pair_labels, angle_labels, axises):
        angles = []

        for l, p, a in zip(angle_labels, pair_labels, axises):
            points_1 = coord_dict[p[0]]
            points_2 = coord_dict[p[1]]
            now_coord_ang = []
            for i in range(points_1.shape[1]):
                angle = cal_angle(points_1[:, i], points_2[:, i], axis=a)
                now_coord_ang.append(angle)

            angles.append(now_coord_ang)
        angles = np.array(angles)

        return angles, angle_labels

#データをcsvから取得
#filepath = './human_data/0018_Walking001.csv'
#use_labels = ['C7', 'RMWT', 'RHIP', 'RKNE', 'RSHN', 'RANK', 'RHEL', 'RTOE', 'LHIP', 'LKNE', 'LSHN', 'LANK', 'LHEL', 'LTOE']
#rd = pd.read_csv(filepath)
#coordinates = {}
#for l in use_labels:
#    fl = 'Subject0018:{}'.format(l)
#    data = []
#    for c in ['X', 'Y', 'Z']:
#        label = '{} {}'.format(fl, c)
#        data.append(rd[label])
#    # xがプラスの方向への歩行成分だけ取り出す
#    coordinates[l] = np.array(data)[:, cut_range[0]:cut_range[1]]
#
#pairs = [['RHEL', 'RTOE'], ['RKNE', 'RANK'], ['RHIP', 'RKNE'],
#         ['C7', 'RMWT'],
#         ['LHEL', 'LTOE'], ['LKNE', 'LANK'], ['LHIP', 'LKNE']]

plotdir = './testplot2/'
os.makedirs(plotdir, exist_ok=True)

filepath = './human_data/ActorA_Walk.c3d'
use_labels = ['TopHead', 'Root', 'RtFtHip', 'RtKnee', 'RtAnkle', 'RtHeel', 'RtToe', 'LtFtHip', 'LtKnee', 'LtAnkle', 'LtHeel', 'LtToe']
cut_range = [0, 89]
#cut_range = [11, 41]

lab_to_ind = {}
coordinates = {}
with open(filepath, 'rb') as handle:
    reader = c3d.Reader(handle)
    # ラベルのインデックスを取得
    for i, l in enumerate(reader.point_labels):
        l = l.replace(' ', '')
        if l in use_labels:
            lab_to_ind[l] = i

    # データの加工
    for l in use_labels:
        ind = lab_to_ind[l]
        datas = []
        for i, (d) in enumerate(reader.read_frames()):
            all_datas = d[1]
            datas.append(all_datas[ind][:3])
        datas = np.array(datas).T
        datas = datas[:, cut_range[0]:cut_range[1]]
        coordinates[l] = datas

# 角度を計算
use_labels = ['TopHead', 'Root', 'RtFtHip', 'RtKnee', 'RtAnkle', 'RtHeel', 'RtToe', 'LtFtHip', 'LtKnee', 'LtAnkle', 'LtHeel', 'LtToe']

angle_labels = ['foot_r', 'shank_r', 'thigh_r', 'thunk', 'foot_l', 'shank_l', 'thigh_l']
axises = [0, 1, 1, 1, 0, 1, 1]
pairs = [['RtHeel', 'RtToe'], ['RtKnee', 'RtAnkle'], ['RtFtHip', 'RtKnee'],
         ['TopHead', 'Root'],
         ['LtHeel', 'LtToe'], ['LtKnee', 'LtAnkle'], ['LtFtHip', 'LtKnee']]

for k in coordinates:
    coordinates[k] = coordinates[k][[1, 2], :]      # y, z軸だけ取り出し

angle_t, _ = coordinate_to_angle(coord_dict=coordinates, pair_labels=pairs, angle_labels=angle_labels, axises=axises)
np.save('{}angle_t'.format(plotdir), angle_t)

# 角度をプロット
fig_num = 7
x = np.arange(angle_t.shape[1]) * 0.0166
plt.clf()
fig = plt.figure(figsize=(len(x) / 20 + 5, fig_num * 2.5))
title = 'angle'
#plt.title(title)
fig.suptitle(title)
ax1 = fig.add_subplot(fig_num, 1, 1)
ax1.plot(x, angle_t[0, :])
ax1.grid()
ax1.set_ylabel(angle_labels[0])
ax2 = fig.add_subplot(fig_num, 1, 2)
ax2.plot(x, angle_t[1, :])
ax2.grid()
ax2.set_ylabel(angle_labels[1])
ax3 = fig.add_subplot(fig_num, 1, 3)
ax3.plot(x, angle_t[2, :])
ax3.grid()
ax3.set_ylabel(angle_labels[2])
ax4 = fig.add_subplot(fig_num, 1, 4)
ax4.plot(x, angle_t[3, :])
ax4.grid()
ax4.set_ylabel(angle_labels[3])
ax5 = fig.add_subplot(fig_num, 1, 5)
ax5.plot(x, angle_t[4, :])
ax5.grid()
ax5.set_ylabel(angle_labels[4])
ax6 = fig.add_subplot(fig_num, 1, 6)
ax6.plot(x, angle_t[5, :])
ax6.grid()
ax6.set_ylabel(angle_labels[5])
ax7 = fig.add_subplot(fig_num, 1, 7)
ax7.plot(x, angle_t[6, :])
ax7.grid()
ax7.set_ylabel(angle_labels[6])
plt.savefig('{}{}.png'.format(plotdir, title))


