"""Helper for fMRI example."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
#
# License: BSD (3-clause)

import os
import csv
from glob import glob
import numpy as np
import pandas as pd

from hcp_builder.dataset import (get_data_dirs, download_experiment,
                                 fetch_subject_list)


EPOCH_DUR_HCP = 12.0
DUR_RUN_HCP = 3*60 + 34
N_SCANS_HCP = 284
TR_HCP = DUR_RUN_HCP / float(N_SCANS_HCP)


def get_hcp_fmri_fname(subject_id, anat_data=False):
    """Return the tfMRI filename."""
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    if not os.path.exists(path):
        download_experiment(subject=subject_id, data_dir=None,
                            data_type='task', tasks='MOTOR', sessions=None,
                            overwrite=True, mock=False, verbose=10)
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
    if anat_data:
        return fmri_path, anat_path
    else:
        return fmri_path


def get_paradigm_hcp(subject_id, data_dir=None):
    """Return onsets, conditions of the HCP task experimental protocol."""
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

    return df, np.linspace(0.0, DUR_RUN_HCP, N_SCANS_HCP)


def get_protocol_hcp(subject_id, trial, data_dir=None):
    """Get the HCP motor task protocol."""
    paradigm_full, _ = get_paradigm_hcp(subject_id)
    paradigm = paradigm_full[paradigm_full['trial_type'] == trial]

    onset = []
    for t, d in zip(paradigm['onset'], paradigm['duration']):
        onset.extend(list(t + np.arange(int(d / TR_HCP)) * TR_HCP))
    trial_type = [trial] * len(onset)
    onset = np.array(onset)
    trial_type = np.array(trial_type)

    t = np.linspace(0.0, DUR_RUN_HCP, N_SCANS_HCP)

    p_e = np.zeros(N_SCANS_HCP)
    mask_cond = (paradigm['trial_type'] == trial).values
    onset_t = paradigm['onset'][mask_cond].values
    onset_idx = (onset_t / TR_HCP).astype(int)
    durations = paradigm['duration'][mask_cond].values
    for i, d in zip(onset_idx, durations):
        p_e[i:int(i+d)] = 1.0

    return trial_type, onset, p_e, t
