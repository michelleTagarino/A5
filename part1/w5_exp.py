#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Maze grid problem environment
  and the Dyna-Q agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("w5_env", "w5_agent")

import numpy as np
import random
import pickle

if __name__ == "__main__":
    num_runs = 10
    max_steps = 10000
    num_episodes = 50

    mat_plot = [[0 for x in range(num_episodes)] for y in range(num_runs)]
 
    
    for r in range(num_runs):
        RL_init()
        random.seed(r)
        np.random.seed(r)

        print "Run number: ", r

        for e in range(num_episodes):
            RL_episode(max_steps)
            mat_plot[r][e] = pickle.loads(RL_agent_message('ValueFunction'))

        RL_cleanup()

    """
    num_steps: numpy array of size 50
    Sum up the columns of mat_plot to get the average number
    of steps per episode over 10 runs
    """
    num_steps = np.mean(mat_plot, axis=0)

    # episode_array: an array that holds integers from 0 to 49
    episode_array = np.arange(num_episodes)

    # Compute average number of steps per episode for plot
    #local_stps = np.zeros(num_episodes)
    #for i in range(num_episodes):
    #    local_stps[i] = float(num_steps.item(i))/num_runs
    #num_steps = local_stps

    # Save the data for matplotlib
    data = np.column_stack((episode_array, num_steps))
    np.savetxt('plot0.txt', data, delimiter=',', fmt='%4.0f')





