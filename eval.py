import pandas as pd
import gym_super_mario_bros
import torch
import time
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from wrappers import *

WORLDS = [1, 2, 3, 4, 5, 6, 7, 8]
STAGES = [1, 2, 3, 4]

def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)

# model_name follows the rule {algorithm}_{SMB version}_{movement type}
def eval_smb_model(model, model_name, smb_version="v0", movement=SIMPLE_MOVEMENT, results_path="./results/"):
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    for world in WORLDS:
        for stage in STAGES:
            env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-{smb_version}")
            env = JoypadSpace(env, movement)
            env = wrap_mario(env)

            total_reward = 0.0
            done = False
            s = arrange(env.reset())
            while not done:
                env.render()
                if device == 'cpu':
                    a = np.argmax(model(s).detach().numpy())
                else:
                    a = np.argmax(model(s).cpu().detach().numpy())
                s_prime, r, done, info = env.step(a)
                s_prime = arrange(s_prime)
                total_reward += r
                s = s_prime
                time.sleep(0.001)
            
            results.append([f"{world}-{stage}", total_reward, info["x_pos"], info["flag_get"]])
            env.close()
            
    results = pd.DataFrame(data=results, columns=["Level", "Reward", "Distance", "Completed"]).set_index("Level")
    results.to_csv(path_or_buf=f"{results_path}{model_name}.csv")
    
    averages = results[["Reward", "Distance"]].mean()
    stds = results[["Reward", "Distance"]].std()
    completed = results["Completed"].sum()
    file = open(f"{results_path}/averages.csv", "a")
    # Model Name, Reward Mean, Reward Std, Distance Mean, Distance Std, Number Of Levels Completed
    file.write(f"{model_name},{averages['Reward']},{stds['Reward']},{averages['Distance']},{stds['Distance']},{completed}")
    file.close()
