"""
Real fMRI data example
======================

Example to recover the different neural temporal activities, the associated
functional networks maps and the HRFs per ROIs in the fMRI data, on the ADHD
dataset resting-state.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import argparse
import pickle
import numpy as np
from nilearn import datasets

from hemolearn import SLRDA
from hemolearn.utils import (fmri_preprocess,
                             sort_atoms_by_explained_variances)
from hemolearn.plotting import (plotting_spatial_comp, plotting_temporal_comp,
                                plotting_obj_values, plotting_hrf,
                                plotting_hrf_stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--n-atoms', type=int, default=30,
                        help='Number of atoms to decompose the fMRI data.')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--lmbd', type=float, default=0.5,
                        help='Set the regularisation parameter for the '
                        'experiment.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--plot-dir', type=str, default='results_adhd',
                        help='Set the name of the plot directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    TR = 2.0
    adhd_dataset = datasets.fetch_adhd(n_subjects=1)
    func_fname = adhd_dataset.func[0]
    confound_fname = adhd_dataset.confounds[0]
    X, _, _ = fmri_preprocess(func_fname, smoothing_fwhm=6.0, standardize=True,
                            detrend=True, low_pass=0.1, high_pass=0.01, t_r=TR,
                            memory='__cache__', verbose=0,
                            confounds=confound_fname)

    seed = np.random.randint(0, 1000) if args.seed is None else args.seed
    print(f'Seed used = {seed}')

    slrda = SLRDA(n_atoms=args.n_atoms, t_r=TR, n_times_atom=30,
                hrf_model='scaled_hrf', lbda=args.lmbd, max_iter=args.max_iter,
                eps=1.0e-3, deactivate_v_learning=False,
                prox_u='l1-positive-simplex', raise_on_increase=True,
                random_state=seed, n_jobs=args.cpu, cache_dir='__cache__',
                nb_fit_try=1, verbose=args.verbose)

    t0 = time.time()
    slrda.fit(X)
    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Fitting done in {}".format(delta_t))

    hrf_ref = slrda.v_init
    roi_label_from_hrf_idx = slrda.roi_label_from_hrf_idx
    pobj, times, a_hat = slrda.lobj, slrda.ltime, slrda.a_hat
    v_hat = slrda.v_hat
    u_hat, z_hat, variances = sort_atoms_by_explained_variances(slrda.u_hat,
                                                                slrda.z_hat,
                                                                slrda.v_hat,
                                                                slrda.hrf_rois)

    res = dict(pobj=pobj, times=times, u_hat=u_hat, v_hat=v_hat, z_hat=z_hat)
    filename = os.path.join(args.plot_dir, "results.pkl")
    print("Pickling results under '{0}'".format(filename))
    with open(filename, "wb") as pfile:
        pickle.dump(res, pfile)

    plotting_spatial_comp(u_hat, variances, slrda.masker_,
                          plot_dir=args.plot_dir, perc_voxels_to_retain=0.1,
                          verbose=True)
    plotting_temporal_comp(z_hat, variances, TR, plot_dir=args.plot_dir,
                           verbose=True)
    plotting_obj_values(times, pobj, plot_dir=args.plot_dir, verbose=True)
    plotting_hrf(v_hat, TR, roi_label_from_hrf_idx, hrf_ref=hrf_ref,
                 normalized=True, plot_dir=args.plot_dir, verbose=True)
    plotting_hrf_stats(v_hat, TR, roi_label_from_hrf_idx, hrf_ref=None,
                       stat_type='tp', plot_dir=args.plot_dir, verbose=True)
    plotting_hrf_stats(v_hat, TR, roi_label_from_hrf_idx, hrf_ref=None,
                       stat_type='fwhm', plot_dir=args.plot_dir, verbose=True)
