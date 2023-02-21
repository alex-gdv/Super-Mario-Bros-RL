from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import sys
import gym

from wrappers import wrap_mario
from callback import SMB_Callback

DAYTIME_CASTLE_STAGES = ["1-1", "1-3", "2-1", "2-3", "4-1", "5-1", "5-2", "5-3", "7-1", "7-3", "8-1", "8-2", "8-3"]
NPROC = 4

def make_env(seed=None, smb_version="v0", movement=SIMPLE_MOVEMENT):
    def _f():
        env = gym.make(f"SuperMarioBrosRandomStages-{smb_version}", stages=DAYTIME_CASTLE_STAGES)
        env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-{smb_version}")
        env = JoypadSpace(env, movement)
        env = wrap_mario(env)
        if seed is not None:
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

def test(algorithm, model_path, **kwargs):
    env = DummyVecEnv([make_env(**kwargs)])
    model = algorithm.load(model_path, env)
    for _ in range(10):
        obs = env.reset()
        while True:
            env.render()
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                break

if __name__ == "__main__":
    args_count = len(sys.argv)
    # if args_count == 2:
    #     if sys.argv[1] == "dqn":
    #         train_DQN()
    #     else:
    #         train_PPO()
    test(DQN, "./models/dqn_v3_simple_400000")
