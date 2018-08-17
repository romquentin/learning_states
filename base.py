import os
import os.path as op
from nose.tools import assert_true


def create_bem_surf(subject, subjects_dir=None, overwrite=False):  # from jr.meg # noqa
    from mne.bem import make_watershed_bem

    # Set file name ----------------------------------------------------------
    bem_dir = op.join(subjects_dir, subject, 'bem')
    src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')
    bem_fname = op.join(bem_dir, subject + '-5120-bem.fif')
    bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')

    # Create watershed BEM surfaces
    if overwrite or not op.isfile(op.join(bem_dir, subject + '-head.fif')):
        make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                           overwrite=True, volume='T1', atlas=False,
                           gcaatlas=False, preflood=None, show=True)
    # Setup source space
    if overwrite or not op.isfile(src_fname):
        from mne import setup_source_space
        files = ['lh.white', 'rh.white', 'lh.sphere', 'rh.sphere']
        for fname in files:
            if not op.exists(op.join(subjects_dir, subject, 'surf', fname)):
                raise RuntimeError('missing: %s' % fname)

        src = setup_source_space(subject=subject, subjects_dir=subjects_dir,
                                 spacing='oct6', surface='white',
                                 add_dist=True, n_jobs=-1, verbose=None)
        src.save(src_fname, overwrite=True)
    # Prepare BEM model
    if overwrite or not op.exists(bem_sol_fname):
        from mne.bem import (make_bem_model, write_bem_surfaces,
                             make_bem_solution, write_bem_solution)
        # run with a single layer model (enough for MEG data)
        surfs = make_bem_model(subject, conductivity=[0.3],
                               subjects_dir=subjects_dir)
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)
