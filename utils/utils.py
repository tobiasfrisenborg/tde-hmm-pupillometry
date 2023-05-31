import numpy as np
import random


def viterbi_path_to_stc(viterbi_path, hmm_states):
    # TODO Implement checks from https://github.com/OHBA-analysis/HMM-MAR/blob/1b24c3ca2d6f58848f39462d6e590ba1cee303d4/utils/general/vpath_to_stc.m
    viterbi_path = viterbi_path.astype(int)
    
    stc = np.zeros([len(viterbi_path), hmm_states])
    
    for index, row in enumerate(stc):
        row[viterbi_path[index] - 1] = 1
        
    return stc


def surrogate_viterbi_path(viterbi_path, hmm_states):
    # First convert to correct format for STC
    stc = viterbi_path_to_stc(viterbi_path, hmm_states)
    # Then we can sample the from the fractional occupancies / probabilities of the states
    state_probs = stc.mean(axis=0).cumsum()
    # Setup the resulting vector
    viterbi_path_surrogate = np.zeros(viterbi_path.shape)
    
    # Sampling loop
    index = 0
    while index < len(viterbi_path):
        # Get index where next state is different from current
        t_next = np.where(viterbi_path[index:] != viterbi_path[index])[0]
        
        # Set end if no new state occurrences are found
        if len(t_next) == 0:
            t_next = len(viterbi_path)
        # Otherwise, update t_next according to current index
        else:
            t_next = t_next[0]
            t_next = index + t_next
        
        # Get a random state value
        state = np.where(state_probs >= random.uniform(0, 1))[0][0]
        
        # Add state to surrogate data and update index
        viterbi_path_surrogate[index:t_next] = state
        index = t_next
    
    return viterbi_path_surrogate

