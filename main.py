from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import sys

from wrappers import wrap_mario
from callback import SMB_Callback

NPROC = 1

def make_env(seed, smb_version="v3", movement = SIMPLE_MOVEMENT):
    def _f():
        env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-{smb_version}")
        env = JoypadSpace(env, movement)
        env = wrap_mario(env)
        env.seed(seed)
        return env
    return _f

def train_DQN(**kwargs):
    callback = SMB_Callback(model_save_freq=200000, model_path="./models/dqn")
    env = SubprocVecEnv([make_env(seed, **kwargs) for seed in range(NPROC)])
    # hyperparameters source: https://github.com/jiseongHAN/Super-Mario-RL.git
    model = DQN("CnnPolicy", env, learning_rate=1e-6, buffer_size=50000, learning_starts=2500,  batch_size=256, train_freq=1, target_update_interval=50, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=512*NPROC*1000, callback=callback)

def train_PPO(**kwargs):
    callback = SMB_Callback(model_save_freq=200000, model_path="./models/ppo")
    env = SubprocVecEnv([make_env(seed, **kwargs) for seed in range(NPROC)])
    n_steps = 512
    model = PPO("CnnPolicy", env, learning_rate=1e-6, n_steps=n_steps, batch_size=32, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=n_steps*NPROC*1000, callback=callback)

if __name__ == "__main__":
    args_count = len(sys.argv)
    if args_count == 2:
        if sys.argv[1] == "dqn":
            train_DQN()
        else:
            train_PPO()
