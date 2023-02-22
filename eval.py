import pandas as pd
import gym_super_mario_bros
import time
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import math
from stable_baselines3 import PPO, DQN
import numpy as np

from wrappers import *

ALL_STAGES = []
for world in [1, 2, 3, 4, 5, 6, 7, 8]:
    for stage in [1, 2, 3, 4]:
        ALL_STAGES.append(f"{world}-{stage}")
DAYTIME_CASTLE_STAGES = ["1-1", "1-3", "2-1", "2-3", "4-1", "5-1", "5-2", "5-3", "7-1", "7-3", "8-1", "8-2", "8-3"]

def mean_std(lst):
    avg = sum(lst) / len(lst)
    std = math.sqrt(sum([(num-avg)**2 for num in lst])/len(lst))
    return avg, std

# model_name follows the rule {algorithm}_{SMB version}_{movement type}
def eval_smb_model(algorithm, model_name, stages=ALL_STAGES, smb_version="v3", movement=SIMPLE_MOVEMENT, results_path="./results/", model_path="./models/"):
    results = []
    model = algorithm.load(model_path + model_name)
    for stage in stages:
        print(f"World-Stage: {stage}")
        N = 5
        rewards = []
        distances = []
        completion_rate = 0

        env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-{smb_version}")
        env = JoypadSpace(env, movement)
        env = wrap_mario(env)

        for _ in range(N):
            total_reward = 0.0
            done = False
            s = env.reset()
            while not done:
                env.render()
                a, _ = model.predict(s)
                if type(a) == np.ndarray:
                    a = a.item()
                s_prime, r, done, info = env.step(a)
                total_reward += r
                s = s_prime
                time.sleep(0.001)
            
            rewards.append(total_reward)
            distances.append(info["x_pos"])
            completion_rate += info["flag_get"]

        reward_avg, reward_std = mean_std(rewards)
        distance_avg, distance_std = mean_std(distances)
        completion_rate /= N

        results.append([stage, reward_avg, reward_std, distance_avg, distance_std, completion_rate])
        env.close()

    results = pd.DataFrame(data=results, columns=["Level", "Reward Avg", "Reward Std", "Distance Avg", "Distance Std", "Completion Rate"]).set_index("Level")
    results = results.round({"Reward Avg":2, "Reward Std":2, "Distance Avg":2, "Distance Std":2, "Completion Rate":4})
    results.to_csv(path_or_buf=f"{results_path}{model_name}.csv")
    
    averages = results[["Reward Avg", "Distance Avg", "Completion Rate"]].mean()
    stds = results[["Reward Avg", "Distance Avg", "Completion Rate"]].std()
    file = open(f"{results_path}/averages.csv", "a")
    # Model Name, Reward Mean, Reward Std, Distance Mean, Distance Std, Completion Rate Avg, Completion Rate Std
    file.write(f"{model_name},{averages['Reward Avg']:.2f},{stds['Reward Avg']:.2f},{averages['Distance Avg']:.2f},{stds['Distance Avg']:.2f},{averages['Completion Rate']:.4f},{stds['Completion Rate']:.4f}\n")
    file.close()

if __name__ == "__main__":
    eval_smb_model(PPO, "ppo_allStages_v3_simple_400000")
