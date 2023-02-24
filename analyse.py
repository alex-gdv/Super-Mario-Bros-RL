import numpy as np

def analyse_results(results_path="./results/"):
    with open(results_path + "averages.csv") as file:
        reward_avgs = []
        for line in file:
            lst = line.split(",")
            reward_avgs.append(tuple([lst[1], lst[0]]))
        reward_avgs_sorted = sorted(reward_avgs, key=lambda tup: tup[0], reverse=True)
        print("Best models in terms of average total reward per level:")
        for index, (_, name) in enumerate(reward_avgs_sorted):
            print(f"{index+1}. {name}")

if __name__ == "__main__":
    analyse_results()
