"""Compute source signal from raw data."""

import numpy as np
import os.path as op
import os
import mne
from mne import Epochs
from mne.viz import plot_bem, plot_alignment
from mne.channels import read_dig_montage
from mne.forward import read_forward_solution
from mne import (make_forward_solution, convert_forward_solution,
                 write_forward_solution, compute_covariance)
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator,
                              apply_inverse, apply_inverse_raw)
from mne.beamformer import (make_lcmv, apply_lcmv, apply_lcmv_raw, make_lcmv,
                            apply_lcmv_epochs)
from config import path_data, freesurfer_path
from base import create_bem_surf, read_hpi_mri
import sys

subject = sys.argv[1]

subjects_dir = freesurfer_path  # freesurfer sub folder
source_method = 'beamformer'
overwrite = False
# Create bem surface. recon-all need to be done at this point
create_bem_surf(subject=subject, subjects_dir=subjects_dir,
                overwrite=overwrite)

# check that BEM are correct
# fig = plot_bem(subject=subject, subjects_dir=subjects_dir,
#                orientation='sagittal', show=True)

# Read raw
fname = op.join(path_data, subject, 'preprocessed_raw',
                subject+'.ica.1-100Hz.raw.fif')
raw = mne.io.read_raw_fif(fname, preload=True)
# raw = read_raw_ctf(fname)
hpi = list()
idx = 0  # XXX: there be multiple HPI per info, which we to chose

# read hpi position in device (ctf_head?) space
for this_hpi in raw.info['hpi_results'][idx]['dig_points']:
    if this_hpi['kind'] == 1 or this_hpi['kind'] == 2:
        hpi.append(this_hpi['r'])
hpi = np.array(hpi)

# read hpi_mri.txt (hpi position in MRI space)
hpi_fname = op.join(path_data, subject, 'neuronav/',
                    '%s_surf.txt' % subject)
landmark = read_hpi_mri(hpi_fname)
point_names = ['NAC', 'LEC', 'REC']  # order is important,not sure why XXX

elp = np.array([landmark[key] for key in point_names])

# Create epochs to calculate noise_cov and data_cov
events = mne.find_events(raw)
event_id = [2, 3]
tmin = -0.4
tmax = 3
epochs = Epochs(raw, events, event_id=event_id,
                tmin=tmin, tmax=tmax, preload=True,
                baseline=(None, 0), decim=5)
epochs.pick_types(stim=True, meg=True, ref_meg=True, eeg=False)

noise_cov = mne.compute_covariance(epochs, -0.2, tmax=0,
                                   method='shrunk', n_jobs=12)
data_cov = mne.compute_covariance(epochs, tmin=0., tmax=None,
                                  method='shrunk', n_jobs=12)

# Apply the montage to the raw
dig_montage = read_dig_montage(hsp=None, hpi=hpi, elp=elp,
                               point_names=point_names, unit='mm',
                               transform=False,
                               dev_head_t=True)
raw.set_montage(dig_montage)
epochs.set_montage(dig_montage)
evoked = epochs.average()

# os.environ['ETS_TOOLKIT'] = 'qt4'
# plot_alignment(epochs.info, trans=None, subject=subject, dig=True,
#                meg='helmet', subjects_dir=subjects_dir)

# Make forward -----------------------------------------------------------
fwd_fname = op.join(path_data, subject, '%s-fwd.fif' % subject)
inv_fname = op.join(path_data, subject, '%s-inv.fif' % subject)
bem_dir = op.join(path_data, subjects_dir, subject, 'bem')
bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')
src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')

# create fwd model
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

# Apply inverse
freesurf_subject = op.join(subjects_dir, subject)
labels = mne.read_labels_from_annot(freesurf_subject, parc='aparc')
ch_names = list()
evoked_stcs = list()
raw_stcs = list()
raw_stcs.append(raw.copy().pick_channels(['UPPT001']).get_data()[0])
ch_names.append('UPPT001')
raw.pick_types(meg=True, eeg=False)
if source_method == 'beamformer':
    for label in labels:
        print label.name
        filters = make_lcmv(raw.info, fwd, data_cov, reg=0.05,
                            noise_cov=noise_cov, label=label,
                            pick_ori=None, weight_norm='unit-noise-gain')
        # Apply lcmv on evoked and raw data
        evoked_stc = apply_lcmv(evoked, filters)
        raw_stc = apply_lcmv_raw(raw, filters)
        ch_names.append(label.name)
        evoked_stcs.append(evoked_stc.data.mean(0))
        raw_stcs.append(raw_stc.data.mean(0))
    evoked_stcs = np.array(evoked_stcs)
    raw_stcs = np.array(raw_stcs)
else:
    # Setup inverse model ----------------------------------------------------
    inv = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2,
                                depth=0.8)
    snr_evoked = 3.0
    snr_raw = 1.0
    for label in labels:
        print label.name
        evoked_stc = apply_inverse(evoked, inv, lambda2=1.0 / (2 ** snr_evoked),
                                   method=source_method, label=label,
                                   pick_ori=None)
        raw_stc = apply_inverse_raw(raw, inv, lambda2=1.0 / (2 ** snr_raw),
                                    method=source_method, label=label,
                                    pick_ori=None)
        ch_names.append(label.name)
        evoked_stcs.append(evoked_stc.data.mean(0))
        raw_stcs.append(raw_stc.data.mean(0))
    evoked_stcs = np.array(evoked_stcs)
    raw_stcs = np.array(raw_stcs)

# Create info for raw_source and evoked_source
info_raw = mne.create_info(ch_names=ch_names,
                           ch_types=['stim'] + ['mag']*len(labels),
                           sfreq=600)
info_evoked = mne.create_info(ch_names=ch_names[1:],
                              ch_types=['mag']*len(labels),
                              sfreq=600)
raw_source = mne.io.RawArray(raw_stcs, info_raw)
evoked_source = mne.EvokedArray(evoked_stcs, info_evoked)

# save raw_source and evoked_source
results_folder = op.join(path_data, subject, 'source_signal')
if not op.isdir(results_folder):
    os.mkdir(results_folder)
fname = op.join(results_folder, '%s_raw_source.fif' % subject)
raw_source.save(fname, overwrite=True)
fname = op.join(results_folder, '%s_evoked_source.fif' % subject)
evoked_source.save(fname)

# save the labels info
fname = op.join(results_folder, 'labels_ch_names.txt')
with open(fname, 'w') as filehandle:
    for listitem in ch_names:
        filehandle.write('%s\n' % listitem)
