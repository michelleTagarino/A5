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
    num_alpha = 6 # Number of alphas in the set
    num_steps = 0
    alpha = None

    #mat_plot = [[0 for x in range(num_alpha)] for y in range(num_runs)]
    steps_array = np.zeros(num_alpha)

    for a in range(num_alpha):
        if a == 0:
            alpha = 0.03125
        elif a == 1:
            alpha = 0.0625
        elif a == 2:
            alpha = 0.125 
        elif a == 3:
            alpha = 0.25 
        elif a == 4:
            alpha = 0.5
        elif a == 5:
            alpha = 1.0

        print "Testing alpha: ", alpha

        for r in range(num_runs):
            RL_init(alpha)
            random.seed(r)
            np.random.seed(r)

            for e in range(num_episodes):
                RL_episode(max_steps)
                if r == 0:
                    num_steps = num_steps + pickle.loads(RL_agent_message('ValueFunction'))
            print "."

        steps_array[a] = num_steps/num_runs
        num_steps = 0

        RL_cleanup()

    # alpha_array: an array that holds integers from 0 to 6
    alpha_array = np.arange(num_alpha)

    # Save the data for matplotlib
    data = np.column_stack((alpha_array, steps_array))
    np.savetxt('plot50.txt', data, delimiter=',', fmt='%4.0f')





