import os
import sys

import numpy as np
import pandas as pd
import tables as tb

import antea.database.load_db as db
from   antea.utils.table_functions import load_rpos


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

start = int(sys.argv[1])
numb = int(sys.argv[2])
max_cut = int(sys.argv[3])

folder = '/data_extra2/paolafer/SimMC/pet/new_h5/'
file_full = folder + 'full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.{0:03d}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/reco_pos/full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.{0}_{1}_{2}'.format(start, numb, max_cut)

rpos_threshold = 4
rpos_file = '/home/paolafer/analysis/petalo/tables/r_table_iradius165mm_z140mm_depth3cm_pitch7mm_thr4pes.h5'

Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = "f{}pes200bins".format(rpos_threshold))


true_x1, true_y1, true_z1 = [], [], []
reco_x1, reco_y1, reco_z1 = [], [], []
true_x2, true_y2, true_z2 = [], [], []
reco_x2, reco_y2, reco_z2 = [], [], []

photo_response1, photo_response2 = [], []
touched_sipms1,  touched_sipms2  = [], []
event_ids = []

for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        full_sns_response = pd.read_hdf(file_name, 'MC/waveforms')
    except ValueError:
        print('File {0} not existing'.format(file_name))
        continue
    except OSError:
        print('File {0} not existing'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/waveforms in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    particles = pd.read_hdf(file_name, 'MC/particles')
    hits      = pd.read_hdf(file_name, 'MC/hits')

    sel_df_r      = find_SiPMs_over_thresholds(full_sns_response, threshold=rpos_threshold)
    sel_df_z_phi  = find_SiPMs_over_thresholds(full_sns_response, threshold=max_cut)

    events = particles.event_id.unique()

    for evt in events[:]:

        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits[hits.event_id           == evt]
        primaries = evt_parts[evt_parts.primary == True]
        #tot_energy = hits[hits.event_id == evt].energy.sum()

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

        ### now that I have selected the event, let's calculate the true position
        ids      = sel_all.particle_id.values
        sel_hits = find_given_particle_hits(ids, evt_hits)

        sel_hits = sel_hits.groupby(['particle_id'])
        true_pos = []
        for _, df in sel_hits:
            hit_positions = np.array([df.x, df.y, df.z]).transpose()
            true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

        if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.513):
            continue

        waveforms = sel_df_r[sel_df_r.event_id == evt]
        if len(waveforms) == 0: continue

        q1, q2, pos1, pos2 = assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx)
        var_phi1 = var_phi2 = None
        if len(pos1) > 0:
            pos1_phi = from_cartesian_to_cyl_v(np.array(pos1))[:,1]
            diff_sign = min(pos1_phi ) < 0 < max(pos1_phi)
            if diff_sign & (np.abs(np.min(pos1_phi))>np.pi/2.):
                pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
            mean_phi = np.average(pos1_phi, weights=q1)
            var_phi1 = np.average((pos1_phi-mean_phi)**2, weights=q1)
        if len(pos2) > 0:
            pos2_phi = from_cartesian_to_cyl_v(np.array(pos2))[:,1]
            diff_sign = min(pos2_phi ) < 0 < max(pos2_phi)
            if diff_sign & (np.abs(np.min(pos2_phi))>np.pi/2.):
       	        pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
            mean_phi = np.average(pos2_phi, weights=q2)
            var_phi2 = np.average((pos2_phi-mean_phi)**2, weights=q2)

        waveforms_z_phi = sel_df_z_phi[sel_df_z_phi.event_id == evt]
        if len(waveforms_z_phi) == 0: continue

        q1, q2, pos1, pos2 = assign_sipms_to_gammas(waveforms_z_phi, true_pos, DataSiPM_idx)

        if len(pos1) > 0 and var_phi1:
            reco_r    = Rpos(np.sqrt(var_phi1)).value
            reco_cart = np.average(pos1, weights=q1, axis=0)
            reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])

            reco_x1.append(reco_r * np.cos(reco_phi))
            reco_y1.append(reco_r * np.sin(reco_phi))
            reco_z1.append(reco_cart[2])
            true_x1.append(true_pos[0][0])
            true_y1.append(true_pos[0][1])
            true_z1.append(true_pos[0][2])
            photo_response1.append(sum(q1))
            touched_sipms1.append(len(q1))
            event_ids.append(evt)
        else:
            reco_x1.append(1.e9)
            reco_y1.append(1.e9)
            reco_z1.append(1.e9)
            true_x1.append(1.e9)
            true_y1.append(1.e9)
            true_z1.append(1.e9)
            photo_response1.append(1.e9)
            touched_sipms1.append(1.e9)
            event_ids.append(evt)

        if len(pos2) > 0 and var_phi2:
            reco_r    = Rpos(np.sqrt(var_phi2)).value
            reco_cart = np.average(pos2, weights=q2, axis=0)
            reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])

            reco_x2.append(reco_r * np.cos(reco_phi))
            reco_y2.append(reco_r * np.sin(reco_phi))
            reco_z2.append(reco_cart[2])
            true_x2.append(true_pos[1][0])
            true_y2.append(true_pos[1][1])
            true_z2.append(true_pos[1][2])
            photo_response2.append(sum(q2))
            touched_sipms2.append(len(q2))
        else:
            reco_x2.append(1.e9)
            reco_y2.append(1.e9)
            reco_z2.append(1.e9)
            true_x2.append(1.e9)
            true_y2.append(1.e9)
            true_z2.append(1.e9)
            photo_response2.append(1.e9)
            touched_sipms2.append(1.e9)


a_true_x1 = np.array(true_x1)
a_true_y1 = np.array(true_y1)
a_true_z1 = np.array(true_z1)
a_true_x2 = np.array(true_x2)
a_true_y2 = np.array(true_y2)
a_true_z2 = np.array(true_z2)

a_reco_x1 = np.array(reco_x1)
a_reco_y1 = np.array(reco_y1)
a_reco_z1 = np.array(reco_z1)
a_reco_x2 = np.array(reco_x2)
a_reco_y2 = np.array(reco_y2)
a_reco_z2 = np.array(reco_z2)

a_touched_sipms1 = np.array(touched_sipms1)
a_touched_sipms2 = np.array(touched_sipms2)

a_photo_response1 = np.array(photo_response1)
a_photo_response2 = np.array(photo_response2)

a_event_ids = np.array(event_ids)

np.savez(evt_file, a_true_x1=a_true_x1, a_true_y1=a_true_y1, a_true_z1=a_true_z1, a_true_x2=a_true_x2, a_true_y2=a_true_y2, a_true_z2=a_true_z2, a_reco_x1=a_reco_x1, a_reco_y1=a_reco_y1, a_reco_z1=a_reco_z1, a_reco_x2=a_reco_x2, a_reco_y2=a_reco_y2, a_reco_z2=a_reco_z2,  a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, a_photo_response1=a_photo_response1, a_photo_response2=a_photo_response2, a_event_ids=a_event_ids)
