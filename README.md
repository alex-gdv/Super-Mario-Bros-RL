# Super-Mario-Bros-RL-Player
The goal of this project is to train a general Super Mario Bros. reinforcement learning agent i.e. 
an agent that can solve any SMB level, even if it has not seen it before.
This task is trickier that it seems at first. 
There are many tutorials out there that teach you how to train an RL agent to play SMB.
The problem is that in these tutorials they always train an agent to play the game starting with World 1 Stage 1, then World 1 Stage 2, etc.
Sometimes, they train a separate model for each stage!
The problem with these approaches is that the agent does not learn to solve SMB generally, 
rather it learns to memorise each level and what to do when presented with a certain frame.
Therefore, if the agent is asked to solve a level it has not seen before, e.g. a custom level, it will do pretty badly.
So, with this project, I am trying to train a good-enough agent that can play any SMB level.
THIS IS STILL UNDER ACTIVE DEVELOPMENT.
