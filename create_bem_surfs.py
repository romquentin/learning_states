"""Create bem surfaces"""

import os.path as op
from config import path_data
from base import create_bem_surf
import sys
subject = sys.argv[1]

subjects_dir = op.join(path_data, 'freesurf_subjects')  # freesurfer sub folder
source_method = 'beamformer'
overwrite = True
# Create bem surface. recon-all need to be done at this point
create_bem_surf(subject=subject, subjects_dir=subjects_dir,
                overwrite=overwrite)
