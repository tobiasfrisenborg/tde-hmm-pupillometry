# -*- coding: utf-8 -*-
"""
Functions for loading, managing, and manipulating the HMM.
@author: Tobias Frisenborg Christensen, 2023
"""
import copy
from pathlib import Path

import numpy as np
import scipy
from glhmm import preproc
from glhmm.glhmm import glhmm
from glhmm.io import read_flattened_hmm_mat


def get_embedded_lags(data_path: Path, hmm_states: int, rep: int) -> int:
    """Get the embedded lags setting from the Nature Communications paper solution.
    
    Parameters
    ----------
    data_path : Path
        Path to the data folder.
    hmm_states : int
        The number of states (either 6 or 12).
    rep : int
        The repetition for the solution.
    
    Returns
    -------
    int
        The embedded_lags setting.
    """
    # Setup the file path
    hmm_solution_path = Path(
        f"{data_path}/HMM/MRC-Notts_3pcc_embedded_K{hmm_states}_rep{rep}_stdised.mat")
    # Read and preprocessing
    hmm = scipy.io.loadmat(hmm_solution_path)
    hmm = hmm['metahmm']
    # Get embedded lags value
    embedded_lags = hmm['train'][0, 0]['embeddedlags'][0, 0][0][-1]
    
    return embedded_lags


def read_hmm(data_path):
    hmm_path = Path(f"{data_path}/flatten_hmm/data/hmm.mat")
    hmm = read_flattened_hmm_mat(hmm_path)
    hmm.trained = True
    
    return hmm


def adjust_hmm_state_persistence(hmm: glhmm, constant: int | float) -> glhmm:
    """Adjust the transition probability matrix of an HMM to increase
    the state persistence.

    Parameters
    ----------
    hmm : glhmm
        The HMM to adjust.
    constant : int | float
        A constant to add to the diagonal of the transition probability matrix.

    Returns
    -------
    glhmm
        The HMM with the adjusted state persistence.
    """
    updated_hmm = copy.deepcopy(hmm)
    
    # Convert to float
    updated_hmm.P = np.float64(updated_hmm.P)
    
    # Add constant to diagonal (state persistence) and normalize
    # rows so they sum to 1 (probability space)
    for i in range(len(updated_hmm.P)):
        updated_hmm.P[i, i] = updated_hmm.P[i, i] * constant
        updated_hmm.P[i] = updated_hmm.P[i] / updated_hmm.P[i].sum()

    return updated_hmm


def apply_hmm(
    hmm: glhmm, embedded_lags: int, data_path: Path, 
    subject: int, session: int) -> tuple[np.array, np.array]:
    """Load the HMM solution from Matlab, the relevant data, preprocess, and
    return the gamma and viterbi path.

    Parameters
    ----------
    hmm : glhmm
        The HMM to apply.
    embedded_lags : int
        Lags to use for the TDE-HMM model.
    data_path : Path
        Path to the data folder.
    subject : int
        The subject ID
    session : int
        The session ID

    Returns
    -------
    tuple[np.array, np.array]
        [1] gamma (a T X K matrix with state probabilities) and
        [2] the viterbi path (a T X 1 matrix with the most probable state).
    """
    # Read in the regular HMM and get the PCA settings (I think)
    hmm_path = Path(f"{data_path}/flatten_hmm/data/hmm.mat")
    hmmmat = scipy.io.loadmat(hmm_path)
    pca_proj = hmmmat["train"]["A"][0][0]
    
    # Read in a subject and preprocess the data
    session_data_path = Path(f"{data_path}/data/subj{subject}_sess{session}.mat")
    data = scipy.io.loadmat(session_data_path)

    X = data["X"]
    indices = np.expand_dims(np.array([0, X.shape[0]]), axis=0)

    # Initial preprocessing function
    X, indices = preproc.preprocess_data(
        data        = X,
        indices     = indices,
        fs          = 250,
        standardise = True, # True / False
        filter      = None, # Tuple with low-pass high-pass thresholds, or None
        detrend     = None, # True / False
        onpower     = False, # True / False
        pca         = None, # Number of components, % explained variance, or None
        whitening   = False, # True / False
        downsample  = None # new frequency, or None
    )

    lags = np.arange(-embedded_lags, embedded_lags+1)
    
    # Build X for the TDE HMM
    X, indices = preproc.build_data_tde(
        X,
        indices,
        lags = lags,
        pca  = pca_proj,
        standardise_pc = True)
        
    # Get Gamma and viterbi path
    gamma, _, _  = hmm.decode(None, X, indices)
    viterbi_path = hmm.decode(None, X, indices, viterbi=True)
    viterbi_path = viterbi_path.nonzero()[1]
    
    return (gamma, viterbi_path)
