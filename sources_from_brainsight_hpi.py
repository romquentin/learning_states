"""Compute source signal."""

import numpy as np
import os.path as op
import os
import mne
from mne.viz import plot_bem
from config import path_data
from base import create_bem_surf

subject = 'ADGGGJAZ'
subjects_dir = op.join(path_data, 'subjects')  # freesurfer sub folder

# Create bem surface. recon-all need to be done at this point
create_bem_surf(subject=subject, subjects_dir=subjects_dir,
                overwrite=True)

# check that BEM are correct
fig = plot_bem(subject=subject, subjects_dir=subjects_dir,
               orientation='sagittal', show=True)

# Read epochs
fname = op.join(path_data, subject, subject+'-epo.fif')
epochs = mne.read_epochs(fname)
