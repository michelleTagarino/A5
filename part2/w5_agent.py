#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for Dyna Maze Starts Control Agent
           for use on A5 of Reinforcement learning course University of Alberta Fall 2017
"""

from utils import rand_in_range, rand_un
from collections import deque
import numpy as np
import random
import pickle

num_total_actions = 4     # Moves: up, down, right, left
num_total_states  = 54    # Maze (including obstacle states)
last_action       = None  # Integer that holds the last action in the previous step
last_state        = None  # Integer that holds the last state in the previous step
PI                = None  # Policy array for state
Q                 = None  # 2D array holds Q(s,a)
M                 = None  # (Model) 2D array holds R(t+1) and S(t+1) given S(t) and A(t)
alpha             = None  # Step size parameter
gamma             = 0.95  # Discount parameter
epsilon           = 0.1   # e-greedy parameter
terminal_Q        = 0     # If the episode is terminal, Q(s+1,a+1) = 0
planning_steps    = 50     # number n for Q-planning
seen_states       = None  # Array of states seen in an episode
num_steps         = 0     # Total number of steps per episode
max_steps         = 10000
seen_array        = None

def agent_init(alpha_param):
	"""
	Hint: Initialize the variables that need to be reset before each run begins
	Returns: nothing
	"""
	global last_action, last_state, num_steps, Q, M, PI, seen_pairs, seen_states, seen_array, alpha
	
	alpha = alpha_param
	last_action = 0
	last_state = 0
	num_steps = 0
	Q = np.zeros((num_total_states, num_total_actions))
	M = np.empty((num_total_states, num_total_actions), dtype=object)
	PI = np.zeros((num_total_states), int)
	seen_array = np.empty(num_total_states, dtype=object)
	seen_states = set()
	return


def agent_start(state):
	"""
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
	global num_steps, seen_states, PI, last_state, last_action, seen_array, M, PI

	last_state = 0
	last_action = 0
	num_steps = 0
	seen_states = set()
	seen_array = np.empty(num_total_states, dtype=object)
	M = np.empty((num_total_states, num_total_actions), dtype=object)

	# Choose action A from state S using policy derived from Q (e.g. e-greedy)
	action = PI[state[0]]
	
	# Save last state and action
	last_state  = state[0]
	last_action = action
	return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
	"""
	Arguments: reward: floating point, state: integer
	Returns: action: integer
	"""
	global Q, M, PI, alpha, gamma, max_eps, planning_steps
	global PI, last_state, last_action, seen_pairs, seen_array
	global max_steps, num_steps

	# Choose action A' from state S' using policy derived from Q (e-greedy)
	action = PI[state[0]]

	seen_states.add(last_state)
	if seen_array[last_state] == None:
		seen_array[last_state] = np.asarray(last_action)
	else:
		seen_array[last_state] = np.append(seen_array[last_state], last_action)

	# Get optimal Q given S(t+1)
	next_state_array = Q[state[0],:]
	max_Q = np.amax(next_state_array)

	# Update Q(s,a) and M(s,a)
	Q[last_state][last_action] = Q[last_state][last_action] + (alpha * (reward + (gamma * max_Q) - Q[last_state][last_action]))
	M[last_state][last_action] = list([reward, state[0]])
	update_PI(last_state)

	# Store the state and action pair
	last_state  = state[0]
	last_action = action

	# Repeat n times
	for n in range(planning_steps):
		s = get_random_state()
		a = get_random_action(s)

		# R, S' <-- Model(S,A)
		rs_pair = M[s][a]
		r = rs_pair[0]
		s_prime = rs_pair[1]

		# max(Q(S',a))
		next_state_array = Q[s_prime,:]
		max_Q = np.amax(next_state_array)
		Q[s][a] = Q[s][a] + (alpha * (r + (gamma * max_Q) - Q[s][a]))
		update_PI(s)

	if num_steps < max_steps:
		num_steps += 1

	return action


def agent_end(reward):
	"""
	Arguments: reward: floating point
	Returns: Nothing
	"""	
	global Q, alpha, gamma, last_state, last_action, terminal_Q
	global max_steps, num_steps

	# Update Q(s,a)
	Q[last_state][last_action] = Q[last_state][last_action] + (alpha * (reward + terminal_Q - Q[last_state][last_action]))
	update_PI(last_state)

	if num_steps < max_steps:
		num_steps += 1
	return


def update_PI(state):

	global Q, PI

	action = 0

	rand_actions = np.random.uniform(0,1)
	if rand_actions <= epsilon:
		action = rand_in_range(num_total_actions)

	else:
		state_array = Q[state,:]
		max_value = np.amax(state_array)
		max_array = np.asarray(np.where(state_array == max_value)).flatten()

		# Randomly choose from the max array if there is more than one max value
		if max_array.size > 1:
			i = np.random.choice(max_array, 1)
			action = int(i)
		else:
			action = int(max_array[0])

	PI[state] = action
	return


def get_random_state():
	global seen_states

	state_array = list(seen_states)
	state = random.choice(state_array)

	return state


def get_random_action(state):
	global seen_array

	actions = np.asarray(seen_array[state]).flatten()
	action = random.choice(actions)

	return action


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message): # returns string, in_message: string
    global num_steps
    """
    Arguments: in_message: string
    returns: The value function as a string.
    """
    if (in_message == 'ValueFunction'):
        return pickle.dumps((num_steps), protocol=0)
    else:
        return "I don't know what to return!!"