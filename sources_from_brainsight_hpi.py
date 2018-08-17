"""Compute source signal."""

import numpy as np
import os.path as op
import os
import mne
from mne.viz import plot_bem, plot_alignment
from mne.channels import read_dig_montage
from mne.forward import read_forward_solution
from mne import (make_forward_solution, convert_forward_solution,
                 write_forward_solution, compute_covariance)
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from config import path_data
from base import create_bem_surf, read_hpi_mri

subject = 'ADGGGJAZ'
subjects_dir = op.join(path_data, 'subjects')  # freesurfer sub folder
overwrite = True
# Create bem surface. recon-all need to be done at this point
create_bem_surf(subject=subject, subjects_dir=subjects_dir,
                overwrite=True)

# check that BEM are correct
fig = plot_bem(subject=subject, subjects_dir=subjects_dir,
               orientation='sagittal', show=True)

# Read epochs
fname = op.join(path_data, subject, subject+'-epo.fif')
epochs = mne.read_epochs(fname)

# raw = read_raw_ctf(fname)
hpi = list()
idx = 0  # XXX: there be multiple HPI per info, which we to chose

# read hpi position in device (ctf_head?) space
for this_hpi in epochs.info['hpi_results'][idx]['dig_points']:
    if this_hpi['kind'] == 1 or this_hpi['kind'] == 2:
        hpi.append(this_hpi['r'])
hpi = np.array(hpi)

# read hpi_mri.txt (hpi position in MRI space)
hpi_fname = op.join(path_data, subject, 'neuronav/', 'fake_hpi_mri_surf.txt')
landmark = read_hpi_mri(hpi_fname)
point_names = ['NEC', 'LEC', 'REC']  # order is important,not sure why XXX

elp = np.array([landmark[key] for key in point_names])

# Apply the montage to the epoch
dig_montage = read_dig_montage(hsp=None, hpi=hpi, elp=elp,
                               point_names=point_names, unit='mm',
                               transform=False,
                               dev_head_t=True)
epochs.set_montage(dig_montage)

plot_alignment(epochs.info, trans=None, subject=subject, dig=True,
               meg='helmet', subjects_dir=subjects_dir)


# Make forward -----------------------------------------------------------
fwd_fname = op.join(path_data, subject, '%s-fwd.fif' % subject)
inv_fname = op.join(path_data, subject, '%s-inv.fif' % subject)
bem_dir = op.join(path_data, subjects_dir, subject, 'bem')
bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')
src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')

info = epochs.info
if not op.isfile(fwd_fname) or overwrite:
    fwd = make_forward_solution(
        info=info, trans=None, src=src_fname,
        bem=bem_sol_fname, meg=True, eeg=False, mindist=5.0)
    # Convert to surface orientation for better visualization
    fwd = convert_forward_solution(fwd, surf_ori=True)
    # save
    write_forward_solution(fwd_fname, fwd, overwrite=True)

fwd = read_forward_solution(fwd_fname)

# Compute noise covariance. -----------------------------------------------
cov = compute_covariance(epochs, tmin=None, tmax=0)

# Setup inverse model ----------------------------------------------------
inv = make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
# write_inverse_operator(inv_fname, inv)
# from mne.minimum_norm import read_inverse_operator
# inv = read_inverse_operator(inv_fname)

# Apply inverse ----------------------------------------------------------
epochs.pick_types(meg=True, ref_meg=True)
method = 'MNE'
# On evoked data and create a video of the output
if method in ['sLORETA', 'dSPM', 'MNE']:
    from mne.minimum_norm import apply_inverse
    snr = 3.0
    evoked = epochs.average()
    stc = apply_inverse(evoked, inv, lambda2=1.0 / (2 ** snr), method=method,
                        pick_ori=None)

brain = stc.plot(hemi='both', subject=subject,
                 subjects_dir=subjects_dir, surface='inflated',
                 background='w')
