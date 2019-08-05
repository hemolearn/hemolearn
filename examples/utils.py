"""Helper to fetcht he HCP fMRI data for the examples."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import csv
from glob import glob
import numpy as np
import pandas as pd

from hcp_builder.dataset import (get_data_dirs, download_experiment,
                                 fetch_subject_list)


EPOCH_DUR_HCP = 12.0
DUR_RUN_HCP_MOTOR = 3*60 + 34
N_SCANS_HCP_MOTOR = 284
TR_HCP_MOTOR = DUR_RUN_HCP_MOTOR / float(N_SCANS_HCP_MOTOR)

DUR_RUN_HCP_REST = 14*60 + 33
N_SCANS_HCP_REST = 1200
TR_HCP_REST = DUR_RUN_HCP_REST / float(N_SCANS_HCP_REST)


def _get_hcp_rest_fmri_fname(subject_id, anat_data=False):
    """ Return the tfMRI filename for rest data.

    Parameters
    ----------
    subject_id : int, HCP id of the subject to fetch fMRI rest data
    anat_data : bool, (default=False), whether to fetch the anatomical MRI data

    Return
    ------
    fmri_path : str, filepath to the functional MRI data
    anat_path : str, filepath to the anatomical MRI data
    """
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    fmri_path = path
    fmri_dirs = ['MNINonLinear', 'Results', 'rfMRI_REST1_RL',
                 'rfMRI_REST1_RL.nii.gz']
    for dir_ in fmri_dirs:
        fmri_path = os.path.join(fmri_path, dir_)
    anat_path = path
    anat_dirs = ['MNINonLinear', 'Results', 'rfMRI_REST1_RL',
                 'rfMRI_REST1_RL_SBRef.nii.gz']
    for dir_ in anat_dirs:
        anat_path = os.path.join(anat_path, dir_)

    if not os.path.exists(fmri_path):
        download_experiment(subject=subject_id, data_dir=None,
                            data_type='rest', tasks=None, sessions=None,
                            overwrite=False, mock=False, verbose=10)

    if anat_data:
        return fmri_path, anat_path
    else:
        return fmri_path


def _get_hcp_motor_task_fmri_fname(subject_id, anat_data=False):
    """Return the tfMRI filename.

    Parameters
    ----------
    subject_id : int, HCP id of the subject to fetch fMRI rest data
    anat_data : bool, (default=False), whether to fetch the anatomical MRI data

    Return
    ------
    fmri_path : str, filepath to the functional MRI data
    anat_path : str, filepath to the anatomical MRI data
    """
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    fmri_path = path
    fmri_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL.nii.gz']
    for dir_ in fmri_dirs:
        fmri_path = os.path.join(fmri_path, dir_)
    anat_path = path
    anat_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL_SBRef.nii.gz']
    for dir_ in anat_dirs:
        anat_path = os.path.join(anat_path, dir_)

    if not os.path.exists(fmri_path):
        download_experiment(subject=subject_id, data_dir=None,
                            data_type='task', tasks='MOTOR', sessions=None,
                            overwrite=False, mock=False, verbose=10)

    if anat_data:
        return fmri_path, anat_path
    else:
        return fmri_path


def get_paradigm_hcp_motor_task(subject_id, data_dir=None):
    """Return onsets and conditions of the HCP motor task experimental
    protocol.

    Parameters
    ----------
    subject_id : int, HCP id of the subject to fetch fMRI rest data
    data_dir : str or None, (default=None), dirpath to HCP data, if None the
        fetcher will look in the HOME directory, if the data is missing it will
        download it


    Return
    ------
    conditions : DataFrame, sum-up the task condition of the fMRI acquisition,
        the columns name are 'onset', 'duration', 'trial_type'
    timeline : array, temporal axe with temporal resolution of the fMRI data
    """
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL', 'EVs']
    for directory in dirs:
        path = os.path.join(path, directory)
    l_files = glob(os.path.join(path, "*"))
    l_files = [f for f in l_files if 'Sync.txt' not in f]

    tmp_dict = {}
    for filename in l_files:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            cond = os.path.splitext(os.path.basename(filename))[0]
            tmp_dict[cond] = [row for row in reader]

    onsets = []
    for trial_type, raw_onsets in tmp_dict.items():
        for t in raw_onsets:
            t[-1] = trial_type
            onsets.append(t)

    df = pd.DataFrame(onsets, columns=['onset', 'duration', 'trial_type'])
    serie_ = pd.Series(np.zeros(df.shape[0], dtype=int), index=df.index)
    df.insert(0, 'session', serie_)
    tmp = df[['onset', 'duration']].apply(pd.to_numeric)
    df[['onset', 'duration']] = tmp
    df = df.sort_values('onset')
    df = df[['session', 'trial_type', 'onset', 'duration']]
    df.reset_index(inplace=True, drop=True)

    return df, np.linspace(0.0, DUR_RUN_HCP_MOTOR, N_SCANS_HCP_MOTOR)