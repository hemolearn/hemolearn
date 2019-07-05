""" Example to recover the different spontanious tasks involved in the BOLD
signal on the ADHD"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import shutil
import pickle
from nilearn import datasets
from nilearn.decomposition import CanICA

from seven.utils import get_unique_dirname
from seven.atlas import fetch_atlas_basc_12_2015
from seven.plotting import plotting_spatial_comp


dirname = get_unique_dirname("results_ica_adhd_#")
if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

TR = 2.0
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_fname = adhd_dataset.func[0]
confound_fname = adhd_dataset.confounds[0]

seed = 0
n_atoms = 20
ica = CanICA(n_components=n_atoms, t_r=TR, memory=".cache", memory_level=2,
             threshold=3., verbose=2, random_state=seed, n_jobs=1)

t0 = time.time()
ica.fit(func_fname, confounds=confound_fname)
delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
print("Fitting done in {}".format(delta_t))

u_hat = ica.components_

res = dict(u_hat=u_hat)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

plotting_spatial_comp(u_hat, [0] * n_atoms, ica.masker_, plot_dir=dirname,
                      perc_voxels_to_retain=0.05, verbose=True)
