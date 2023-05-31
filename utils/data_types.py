from pathlib import Path

import numpy as np

from .data_io import get_pupil
from .utils import viterbi_path_to_stc, surrogate_viterbi_path
from .hmm import adjust_hmm_state_persistence, read_hmm, apply_hmm, get_embedded_lags


def setup_sessions(data_path, hmm_states, constants=None, condition='all', keep_invalid=False, n_permutations=100):
    assert condition in ['bright-room', 'dark-room', 'all']
    
    # The repetition for the HMM solution
    rep = 1
    
    embedded_lags = get_embedded_lags(data_path, hmm_states, rep)
    min_t = calculate_min_t(data_path, embedded_lags)
    
    sessions = []
    
    for subject_i in range(1, 11):
        subject_sessions = [1, 2, 3]
        if subject_i == 5:
             subject_sessions = [2, 3, 4]

        for session_i in subject_sessions:            
            session = Session(subject_i, session_i)
            if session.condition == condition or condition == 'all':
                session.setup_data(data_path, min_t, embedded_lags, hmm_states)
                
                for constant in constants:
                    session.decode_hmm(data_path, constant)
                
                if session.hmm['None']['valid_session'] or keep_invalid:
                    print(f"Session is valid, appending")
                    session.setup_permutations(n_permutations)
                    sessions.append(session)

    return sessions

        
def calculate_min_t(data_path, embedded_lags):
    # Identify the shortest dataset because we need to align them later
    min_t = np.inf

    # Go through each subject and session and overwrite min_t if a shorter
    # session is found
    for subject in range(1, 11):
        sessions = [1, 2, 3] if subject != 5 else [2, 3, 4]
        for session in sessions:
            pupil = get_pupil(data_path, subject, session)
            pupil = pupil[embedded_lags : -embedded_lags]
            
            pupil_t = len(pupil[:, 0]) - 1
            min_t = min(min_t, pupil_t)
    
    return min_t


class Session():
    def __init__(self, subject, session):
        self.subject            = subject
        self.session            = session
        self.hmm_states         = None
        self.embedded_lags      = None
        self.t                  = None
        self.pupil              = None
        self.valid_pupil        = None
        self.hmm                = {}
        self.valid_session      = None
        self.statistics         = {}
        if session == 2:
            self.condition = 'bright-room'
        else:
            self.condition = 'dark-room'
    
    def setup_data(self, data_path, t, embedded_lags, hmm_states):
        self.hmm_states = hmm_states
        self.embedded_lags = embedded_lags
        self.t = t
        self.pupil, self.valid_pupil = self.get_pupil(data_path, t, embedded_lags)

    def get_pupil(self, data_path, t, embedded_lags):
        # Load the pupil data
        pupil_raw = get_pupil(data_path, self.subject, self.session)
        pupil_raw = pupil_raw[embedded_lags : -embedded_lags]
        pupil_raw = pupil_raw[:t]
        
        # Create a matrix for valid pupil time-points
        valid_pupil = ~np.isnan(pupil_raw.sum(axis=1))
        
        # Prepare and return the pupil array and the valid pupil time points
        pupil = np.zeros([t])
        pupil[valid_pupil] = pupil_raw[valid_pupil].mean(axis=1)
        
        return (pupil, valid_pupil)
    
    def decode_hmm(self, data_path, constant):
        print(f"- Setting up HMM for subject {self.subject}, session {self.session}, constant {constant}")
        hmm = read_hmm(data_path)
        
        if constant is not None:
            hmm = adjust_hmm_state_persistence(hmm, constant)
        
        gamma, viterbi_path = apply_hmm(
            hmm           = hmm,
            embedded_lags = self.embedded_lags,
            data_path     = data_path,
            subject       = self.subject,
            session       = self.session)
        
        gamma = gamma[:self.t]
        viterbi_path = viterbi_path[:self.t]
        
        self.hmm[str(constant)] = {}
        self.hmm[str(constant)]['gamma'] = gamma
        self.hmm[str(constant)]['viterbi_path'] = viterbi_path
        self.hmm[str(constant)]['valid_session'] = self.is_valid_session(viterbi_path)
    
    def is_valid_session(self, viterbi_path):
        stc = viterbi_path_to_stc(viterbi_path, self.hmm_states)
        state_proportions = stc.mean(axis=0)
        
        if state_proportions.max() > .75:
            valid_session = False
        else:
            valid_session = True
        
        return valid_session

    def setup_permutations(self, n_permutations):
        permutations_matrix = np.zeros([n_permutations, len(self.pupil)])
        
        for constant, val in self.hmm.items():
            print(f"- Setting up permutations for subject {self.subject}, session {self.session}, constant {constant}")
            for permutation in range(n_permutations):
                viterbi_path = val['viterbi_path'].copy()
                permutations_matrix[permutation] = surrogate_viterbi_path(viterbi_path, self.hmm_states)
            self.hmm[constant]['permutations'] = permutations_matrix
