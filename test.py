from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from matplotlib import pyplot as plt

from wrappers import wrap_mario, MaxAndSkipEnv, WarpFrame

env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MaxAndSkipEnv(env)
env = WarpFrame(env)

total_reward = 0.0
done = False
s = env.reset()
while not done:
    env.render()
    s_prime, r, done, info = env.step(1)
    total_reward += r
    s = s_prime
    print(s_prime.shape)
    plt.imshow(s_prime, interpolation='nearest', cmap="gray")
    plt.show()
