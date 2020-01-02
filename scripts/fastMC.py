import sys
import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.core         import system_of_units as units
from invisible_cities.io.mcinfo_io import units_dict

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.mcsim.errmat     as errmat
import antea.mcsim.errmat3d   as errmat3d
import antea.mcsim.fastmc3d   as fmc
import antea.io.mc_io         as mcio

### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

table_folder = '/path/to/error/matrices/folder'
err_r_phot_file   = table_folder + '/errmat_r_phot_like.npz'
err_r_compt_file   = table_folder + '/errmat_r_compt_like.npz'
err_phi_phot_file = table_folder + '/errmat_phi_phot_like.npz'
err_phi_compt_file = table_folder + '/errmat_phi_compt_like.npz'
err_z_phot_file   = table_folder + '/errmat_z_phot_like.npz'
err_z_compt_file   = table_folder + '/errmat_z_compt_like.npz'
err_t_phot_file   = table_folder + '/errmat_t_phot_like.npz'
err_t_compt_file   = table_folder + '/errmat_t_compt_like.npz'

errmat_r_phot    = errmat.errmat(err_r_phot_file)
errmat_r_compt   = errmat.errmat(err_r_compt_file)
errmat_phi_phot  = errmat3d.errmat3d(err_phi_phot_file)
errmat_phi_compt = errmat3d.errmat3d(err_phi_compt_file)
errmat_z_phot    = errmat3d.errmat3d(err_z_phot_file)
errmat_z_compt   = errmat3d.errmat3d(err_z_compt_file)
errmat_t_phot    = errmat.errmat(err_t_phot_file)
errmat_t_compt   = errmat.errmat(err_t_compt_file)

start         = int(sys.argv[1])
numb_of_files = int(sys.argv[2])

for file_number in range(start, start+numb_of_files):
     
    folder   = '/path/to/sim/files'
    sim_file = folder + '/geant4_sim.{}.pet.h5'.format(file_number)
    out_file = folder + '/reco_sim.{}.h5'.format(file_number)

    try:
        particles = mcio.load_mcparticles(sim_file)
    except:
        print('File {} not found!'.format(sim_file))
        continue
    hits      = mcio.load_mchits(sim_file)

    events = particles.event_id.unique()

    reco = pd.DataFrame(columns=['event_id',
                                 'true_energy',
                                 'true_r1',
                                 'true_phi1',
                                 'true_z1',
                                 'true_t1',
                                 'true_r2',
                                 'true_phi2',
                                 'true_z2',
                                 'true_t2',
                                 'phot_like1',
                                 'phot_like2',
                                 'reco_r1',
                                 'reco_phi1',
                                 'reco_z1',
                                 'reco_t1',                             
                                 'reco_r2',
                                 'reco_phi2',
                                 'reco_z2',
                                 'reco_t2'])

    for evt in events:

        evt_df = fmc.simulate_reco_event(evt, hits, particles, errmat_r_phot,
                                         errmat_phi_phot, errmat_z_phot, errmat_t_phot, errmat_r_compt,
                                         errmat_phi_compt, errmat_z_compt, errmat_t_compt, 0.95)
        reco = pd.concat([reco, evt_df])
    

    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco, format='table', data_columns=True)
    store.close()
    
   
