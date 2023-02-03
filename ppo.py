from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import wrap_mario

NPROC = 4

def make_env(seed, smb_version="v3", movement=SIMPLE_MOVEMENT):
    def _f():
        env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-{smb_version}")
        env = JoypadSpace(env, movement)
        env = wrap_mario(env)
        env.seed(seed)
        return env
    return _f

def train(n_steps, batch_size, **kwargs):
    env = SubprocVecEnv([make_env(seed, **kwargs) for seed in range(NPROC)])
    model = PPO("CnnPolicy", env, learning_rate=1e-6, n_steps=n_steps, batch_size=batch_size, verbose=1)
    model.learn(total_timesteps=n_steps*NPROC*100)

if __name__ == "__main__":
    train(n_steps=512, batch_size=32)