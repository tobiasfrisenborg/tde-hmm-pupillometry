"""
Tobias Frisenborg Christensen, 2023
"""

from pathlib import Path

import numpy as np
import pickle
import scipy.io


def get_pupil(data_path, subject, session):
    """Utility function for loading in pupil data"""
    # Setup the file path
    pupil_path = Path(f"{data_path}/Pupil/subj{subject}_sess{session}.mat")
    # Read and preprocessing
    pupil = scipy.io.loadmat(pupil_path)
    pupil = pupil['pupil_filtered']
    
    return pupil


def save_to_pkl(data, path):
    """Utility function for saving to a pickle file"""
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path):
    """Utility function for loading from a pickle file"""
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    
    return data
