from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import argparse
import gym

from wrappers import wrap_mario
from callback import SMB_Callback

ALL_STAGES = []
for world in [1, 2, 3, 4, 5, 6, 7, 8]:
    for stage in [1, 2, 3, 4]:
        ALL_STAGES.append(f"{world}-{stage}")
DAYTIME_CASTLE_STAGES = ["1-1", "1-3", "2-1", "2-3", "4-1", "5-1", "5-2", "5-3", "7-1", "7-3", "8-1", "8-2", "8-3"]
STAGES_DICT = {"all_stages":ALL_STAGES, "daytime_castle_stages":DAYTIME_CASTLE_STAGES}
NPROC = 4

def make_env(seed=None, stages=DAYTIME_CASTLE_STAGES, smb_version="v3", movement=SIMPLE_MOVEMENT):
    def _f():
        env = gym.make(f"SuperMarioBrosRandomStages-{smb_version}", stages=stages)
        env = JoypadSpace(env, movement)
        env = wrap_mario(env)
        if seed is not None:
            env.seed(seed)
        return env
    return _f

def train_DQN(model_name, **kwargs):
    callback = SMB_Callback(model_save_freq=200000, model_name=model_name)
    env = SubprocVecEnv([make_env(seed, **kwargs) for seed in range(NPROC)])
    # hyperparameters source: https://github.com/jiseongHAN/Super-Mario-RL.git
    model = DQN("CnnPolicy", env, learning_rate=1e-6, buffer_size=50000, learning_starts=2500,  batch_size=256, train_freq=1, target_update_interval=50, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=512*NPROC*1000, callback=callback)

def train_PPO(model_name, **kwargs):
    callback = SMB_Callback(model_save_freq=200000, model_name=model_name)
    env = SubprocVecEnv([make_env(seed, **kwargs) for seed in range(NPROC)])
    n_steps = 512
    model = PPO("CnnPolicy", env, learning_rate=1e-6, n_steps=n_steps, batch_size=32, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=n_steps*NPROC*1000, callback=callback)

def test(algorithm, model_name, model_path="./models/", **kwargs):
    env = DummyVecEnv([make_env(**kwargs)])
    model = algorithm.load(f"{model_path}{model_name}", env)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=["PPO", "DQN"])
    parser.add_argument("model_name")
    parser.add_argument("--stages", default="daytime_castle_stages", choices=["all_stages", "daytime_castle_stages"])
    parser.add_argument("--smb_version", default="v3", choices=["v0", "v1", "v2", "v3"])
    parser.add_argument("--mode", default="train", choices=["test", "train"])
    args = parser.parse_args()
    if args.mode == "train":
        if args.algorithm == "PPO":
            train_PPO(args.model_name, smb_version=args.smb_version, stages=STAGES_DICT[args.stages])
        else:
            train_DQN(args.model_name, smb_version=args.smb_version, stages=STAGES_DICT[args.stages])
    else:
        if args.algorithm == "PPO":
            test(PPO, args.model_name, smb_version=args.smb_version, stages=STAGES_DICT[args.stages])
        else:
            test(DQN, args.model_name, smb_version=args.smb_version, stages=STAGES_DICT[args.stages])
