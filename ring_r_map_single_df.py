
import os
import sys
import math
import tables as tb
import numpy  as np
import pandas as pd

import antea.database.load_db as db

from   invisible_cities.core.system_of_units_c import units

def lower_or_equal(f1, f2, allowed_error=1.e-6):
    return f1 <= f2 + allowed_error


def greater_or_equal(f1, f2, allowed_error=1.e-6):
    return f1 >= f2 - allowed_error


def from_cartesian_to_cyl_v(pos):
    cyl_pos = np.array([np.sqrt(pos[:,0]**2+pos[:,1]**2), np.arctan2(pos[:,1], pos[:,0]), pos[:,2]]).transpose()
    return cyl_pos

def find_SiPMs_over_thresholds(df, threshold):

    tot_charges_df = df.groupby(['event_id','sensor_id'])[['charge']].sum()
    return tot_charges_df[tot_charges_df.charge > threshold].reset_index()


def find_closest_sipm(given_pos, sns_positions, sipms):
    ### Find the closest SiPM to the true average point
    subtr = np.subtract(given_pos, sns_positions)

    distances = [np.linalg.norm(d) for d in subtr]
    min_dist = np.min(distances)
    min_sipm = np.isclose(distances, min_dist)
    closest_sipm = sipms[min_sipm]

    return closest_sipm


def find_given_particle_hits(p_ids, hits):
    return  hits[hits.particle_id.isin(p_ids)]


def assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx):

    sipms = DataSiPM_idx.loc[waveforms.sensor_id]
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = waveforms.charge

    sns_closest_pos = [np.array([find_closest_sipm(pos, sns_positions, sipms).X.values, find_closest_sipm(pos, sns_positions, sipms).Y.values, find_closest_sipm(pos, sns_positions, sipms).Z.values]).transpose()[0] for pos in true_pos]

    q1, q2     = [], []
    pos1, pos2 = [], []

    closest_pos = sns_closest_pos[0] ## I look at the first one, which always exists.
    ### The sensors on the same semisphere are grouped together, and those on the opposite side, too, only
    ### if two 511-keV interactions have been detected.
    for sns_pos, charge in zip(sns_positions, sns_charges):
        scalar_prod = sum(a*b for a, b in zip(sns_pos, closest_pos))
        if scalar_prod > 0.:
            q1.append(charge)
            pos1.append(sns_pos)
        elif len(sns_closest_pos) == 2:
            q2.append(charge)
            pos2.append(sns_pos)

    return q1, q2, pos1, pos2


### read sensor positions from database
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')


start     = int(sys.argv[1])
numb      = int(sys.argv[2])
threshold = int(sys.argv[3])


folder = '/data_extra2/paolafer/SimMC/pet/new_h5/'
file_full = folder + 'full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.{0:03d}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/r_map/full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.{0}_{1}_{2}'.format(start, numb, threshold)

true_r1 = []
true_r2 = []

touched_sipms1  = []
touched_sipms2  = []

var_phi1 = []
var_phi2 = []

var_z1 = []
var_z2 = []


for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        full_sns_response = pd.read_hdf(file_name, 'MC/waveforms')
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} boh'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/waveforms in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    particles = pd.read_hdf(file_name, 'MC/particles')
    hits      = pd.read_hdf(file_name, 'MC/hits')
    events    = particles.event_id.unique()

    sel_df = find_SiPMs_over_thresholds(full_sns_response, threshold)

    for evt in events[:]:

        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits[hits.event_id           == evt]
        primaries = evt_parts[evt_parts.primary == True]

        sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
        sel_name     = evt_parts.name == 'e-'
        sel_vol_name = evt_parts[sel_volume & sel_name]

        ids      = sel_vol_name.particle_id.values
        sel_hits = find_given_particle_hits(ids, evt_hits)
        energies = sel_hits.groupby(['particle_id'])[['energy']].sum()
        energies = energies.reset_index()
        energy_sel  = energies[greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]
        sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

        sel_all         = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]
        if len(sel_all) == 0: continue

        ### now that I have selected the event, let's calculate the true position(s)
        ids      = sel_all.particle_id.values
        sel_hits = find_given_particle_hits(ids, evt_hits)

        sel_hits = sel_hits.groupby(['particle_id'])
        true_pos = []
        for _, df in sel_hits:
            hit_positions = np.array([df.x, df.y, df.z]).transpose()
            true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

        waveforms = sel_df[sel_df.event_id == evt]
        if len(waveforms) == 0: continue

        q1, q2, pos1, pos2 = assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx)

        if len(pos1) > 0:
            pos_phi  = from_cartesian_to_cyl_v(np.array(pos1))[:,1]
            mean_phi = np.average(pos_phi, weights=q1)
            var_phi  = np.average((pos_phi-mean_phi)**2, weights=q1)

            pos_z  = np.array(pos1)[:,2]
            mean_z = np.average(pos_z, weights=q1)
            var_z  = np.average((pos_z-mean_z)**2, weights=q1)

            reco_cart = np.average(pos1, weights=q1, axis=0)

            var_phi1.append(var_phi)
            var_z1.append(var_z)
            touched_sipms1.append(len(pos1))
            r = np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2)
            true_r1.append(r)

        else:
            var_phi1.append(1.e9)
            var_z1.append(1.e9)
            touched_sipms1.append(1.e9)
            true_r1.append(1.e9)


        if len(pos2) > 0:
            pos_phi  = from_cartesian_to_cyl_v(np.array(pos2))[:,1]
            mean_phi = np.average(pos_phi, weights=q2)
            var_phi  = np.average((pos_phi-mean_phi)**2, weights=q2)

            pos_z  = np.array(pos2)[:,2]
            mean_z = np.average(pos_z, weights=q2)
            var_z  = np.average((pos_z-mean_z)**2, weights=q2)

            reco_cart = np.average(pos2, weights=q2, axis=0)

            var_phi2.append(var_phi)
            var_z2.append(var_z)
            touched_sipms2.append(len(pos2))
            r = np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2)
            true_r2.append(r)

        else:
            var_phi2.append(1.e9)
            var_z2.append(1.e9)
            touched_sipms2.append(1.e9)
            true_r2.append(1.e9)





a_true_r1 = np.array(true_r1)
a_true_r2 = np.array(true_r2)
a_var_phi1 = np.array(var_phi1)
a_var_phi2 = np.array(var_phi2)
a_var_z1 = np.array(var_z1)
a_var_z2 = np.array(var_z2)
a_touched_sipms1 = np.array(touched_sipms1)
a_touched_sipms2 = np.array(touched_sipms2)


np.savez(evt_file, a_true_r1=a_true_r1, a_true_r2=a_true_r2, a_var_phi1=a_var_phi1, a_var_phi2=a_var_phi2, a_var_z1=a_var_z1, a_var_z2=a_var_z2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2)
