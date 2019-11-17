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


# 学習設定
train = True       # 学習をするかどうか
validation = True   # 学習結果を使って評価をするかどうか

#env_name = 'RoboschoolHumanoid-v1'
env_name = 'RoboschoolWalker2d-v1'
num_cpu = 2         # 学習に使用するCPU数
learn_timesteps = 10**6     # 学習タイムステップ

ori_env = gym.make(env_name)
#env = DummyVecEnv([lambda: ori_env])
env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])
env.reset()
#env.render()
#time.sleep(5)

savedir = './stable_baselines/{}/'.format(env_name)
logdir = '{}tensorboard_log/'.format(savedir)
os.makedirs(savedir, exist_ok=True)

starttime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
# 学習の実行
if train:
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=learn_timesteps)
    model.save('{}ppo2_model'.format(savedir))

endtime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")


# 学習結果の確認
if validation:
    model = PPO2.load('{}ppo2_model'.format(savedir))
    from gym import wrappers

    video_path = '{}video'.format(savedir)
    wrap_env = wrappers.Monitor(ori_env, video_path, force=True)

    done = False
    obs = wrap_env.reset()

    for step in range(10000):
        if step % 10 == 0: print("step :", step)
        if done:
            time.sleep(1)
            o = wrap_env.reset()
            break

        action, _states = model.predict(obs)
        obs, rewards, done, info = wrap_env.step(action)

    wrap_env.close()
env.close()

print(starttime)
print(endtime)
