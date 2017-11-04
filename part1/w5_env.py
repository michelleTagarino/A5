#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Simple Maze grid problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 8.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

col = 9
row = 6
states_array  = np.zeros((row,col), int)
current_state = None
start_state   = None
goal_state    = None
obstacle_set  = None
wind_1 = None
wind_2 = None
right_edge = None
left_edge  = None
min_limit = 0
max_limit = 46
up    = -9
down  =  9
right =  1
left  = -1

temp_state = 0

def env_init():
    global current_state, states_array, start_state, goal_state, top_edge, bottom_edge, right_edge, left_edge, obstacle_set
    current_state = np.zeros(1)
    top_edge      = set()
    bottom_edge   = set()
    right_edge    = set()
    left_edge     = set()
    obstacle_set  = set()

    # Initialize state numbers
    x = 0
    for i in range(row):
        for j in range(col):
            states_array[i][j] = x
            x += 1

    # Classify which states are obstacle states
    for i in range(row):
        for j in range(col):
            if (i == 0 and j == 7):
                obstacle_set.add(states_array[i][j])
            if i == 1 and (j == 2 or j == 7):
                obstacle_set.add(states_array[i][j])
            if i == 2 and (j == 2 or j == 7):
                obstacle_set.add(states_array[i][j])
            if i == 3 and j == 2:
                obstacle_set.add(states_array[i][j])
            if i ==4 and j == 5:
                obstacle_set.add(states_array[i][j])

    # Classify which states are edges
    for i in range(row):
        for j in range(col):
            if i == 0 and states_array[i][j] not in obstacle_set:
                top_edge.add((states_array[i][j]))
            if i == 5 and states_array[i][j] not in obstacle_set:
                bottom_edge.add((states_array[i][j]))
            if j == 8 and states_array[i][j] not in obstacle_set:
                right_edge.add((states_array[i][j]))
            if j == 0 and states_array[i][j] not in obstacle_set:
                left_edge.add((states_array[i][j]))

    # Set start state and goal state
    start_state = states_array[2][0]
    goal_state  = states_array[0][8]

def env_start():
    """ returns numpy array """
    global current_state
    state = start_state
    current_state = np.asarray([state])
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state, up, down, right, left
    global top_edge, bottom_edge, right_edge, left_edge, obstacle_set, temp_state

    temp_state = 0

    if action == 0: # up
        if current_state[0] not in top_edge:
            temp_state = current_state[0] + up
        if temp_state not in obstacle_set and temp_state >= min_limit:
            current_state[0] = temp_state

    elif action == 1: # down
        if current_state[0] not in bottom_edge:
            temp_state = current_state[0] + down
            if temp_state not in obstacle_set and temp_state <= max_limit:
                current_state[0] = temp_state

    elif action == 2: # right
        if current_state[0] not in right_edge:
            temp_state = current_state[0] + right
            if temp_state not in obstacle_set and temp_state <= max_limit:
                current_state[0] = temp_state

    elif action == 3: # left
        if current_state[0] not in left_edge:
            temp_state = current_state[0] + left
            if temp_state not in obstacle_set and temp_state >= min_limit:
                current_state[0] = temp_state

    reward = 0.0
    is_terminal = False
    if current_state[0] == goal_state:
        is_terminal = True
        current_state = None
        reward = 1.0

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
