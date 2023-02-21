import pandas as pd
import gym_super_mario_bros
import time
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import math
from stable_baselines3 import PPO, DQN

from wrappers import *

WORLDS = [1, 2, 3, 4, 5, 6, 7, 8]
STAGES = [1, 2, 3, 4]

def mean_std(lst):
    avg = sum(lst) / len(lst)
    std = math.sqrt(sum([(num-avg)**2 for num in lst])/len(lst))
    return avg, std

# model_name follows the rule {algorithm}_{SMB version}_{movement type}
def eval_smb_model(algorithm, model_name, smb_version="v0", movement=SIMPLE_MOVEMENT, results_path="./results/", model_path="./models/"):
    results = []
    model = algorithm.load(model_path + model_name)
    for world in WORLDS:
        for stage in STAGES:
            print(f"World {world} Stage {stage}")
            N = 5
            rewards = []
            distances = []
            completion_rate = 0

            env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-{smb_version}")
            env = JoypadSpace(env, movement)
            env = wrap_mario(env)

            for _ in range(N):
                total_reward = 0.0
                done = False
                s = env.reset()
                while not done:
                    env.render()
                    a, _ = model.predict(s)
                    print(type(a))
                    print(a)
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

            results.append([f"{world}-{stage}", reward_avg, reward_std, distance_avg, distance_std, completion_rate])
            env.close()
            
    results = pd.DataFrame(data=results, columns=["Level", "Reward Avg", "Reward Std", "Distance Avg", "Distance Std", "Completion Rate"]).set_index("Level")
    results.to_csv(path_or_buf=f"{results_path}{model_name}.csv")
    
    averages = results[["Reward Avg", "Distance Avg"]].mean()
    stds = results[["Reward Avg", "Distance Avg"]].std()
    completed = results["Completion Rate"].sum()
    file = open(f"{results_path}/averages.csv", "a")
    # Model Name, Reward Mean, Reward Std, Distance Mean, Distance Std, Number Of Levels Completed
    file.write(f"{model_name},{averages['Reward Avg']},{stds['Reward Avg']},{averages['Distance Avg']},{stds['Distance Avg']},{completed}")
    file.close()

if __name__ == "__main__":
    eval_smb_model(DQN, "dqn_v3_simple_400000", smb_version="v3")
