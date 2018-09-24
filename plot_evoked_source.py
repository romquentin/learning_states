import mne
import matplotlib.pyplot as plt
import os.path as op
import os
from config import path_data
subject = 'NJCOADSS'
results_folder = op.join(path_data, subject, 'source_signal')

fname = op.join(results_folder, '%s_evoked_source.fif' % subject)
evoked_source = mne.read_evokeds(fname, condition=0)
times = evoked_source.times

left_calca = evoked_source.copy().pick_channels(['pericalcarine-l']).data[0]
right_calca = evoked_source.copy().pick_channels(['pericalcarine-r']).data[0]
left_motor = evoked_source.copy().pick_channels(['precentral-lh']).data[0]
right_motor = evoked_source.copy().pick_channels(['precentral-rh']).data[0]
left_orbitofrontal = evoked_source.copy().pick_channels(['lateralorbito-0']).data[0]
right_orbitofrontal = evoked_source.copy().pick_channels(['lateralorbito-1']).data[0]
left_caudalmiddlefrontal = evoked_source.copy().pick_channels(['caudalmiddlef-0']).data[0]
right_caudalmiddlefrontal = evoked_source.copy().pick_channels(['caudalmiddlef-1']).data[0]

plt.plot(times, left_calca, color='C0', alpha=0.9, label='Left calcarine')
plt.plot(times, right_calca, color='C0', alpha=0.5, label='Right calcarine')
plt.plot(times, left_motor, color='C1', alpha=0.9, label='Left motor')
plt.plot(times, right_motor, color='C1', alpha=0.5, label='Right motor')
plt.plot(times, left_orbitofrontal, color='C2', alpha=0.9,
         label='Left orbitofrontal')
plt.plot(times, right_orbitofrontal, color='C2', alpha=0.5,
         label='Right orbitofrontal')
plt.plot(times, left_caudalmiddlefrontal, color='C3', alpha=0.9,
         label='Left middle frontal')
plt.plot(times, right_caudalmiddlefrontal, color='C3', alpha=0.5,
         label='Right middle frontal')
plt.legend()
