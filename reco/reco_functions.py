import numpy as np
import pandas as pd
import tables as tb
import random
import sys
import os

from invisible_cities.core.system_of_units_c import units
from invisible_cities.evm.event_model import Waveform, MCParticle

from   antea.utils.table_functions import load_rpos
import antea.database.load_db as db

from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge

from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict


def sensor_position(h5in):
    """Returns dictionary that stores the position of all the sensors
    in cartesian coordinates
    """
    sipms    = h5in.root.MC.sensor_positions[:]
    sens_pos = {}
    for sipm in sipms:
        sens_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
    return sens_pos


def find_closest_sipm(given_pos, sensor_pos, sns_over_thr, charges_over_thr):
    ### Find the closest SiPM to the true average point
    min_dist = 1.e9
    min_sns = 0
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos = sensor_pos[sns_id]
        dist = np.linalg.norm(np.subtract(given_pos, pos))
        if dist < min_dist:
            min_dist = dist
            min_sns = sns_id

    return min_sns

def barycenter_3D(pos, qs):
    """pos = column np.array --> (matrix n x 3)
       ([x1, y1, z1],
        ...
        [xs, ys, zs])
       qs = vector (q1, ...qs) --> (1xn)
        """

    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge

    return np.average(pos, weights=qs, axis=0)


def calculate_phi_z_pos(pos, q, phi_z_thr):

    indices_over_thr = (q > phi_z_thr)
    q_over_thr       = q[indices_over_thr]
    pos_over_thr     = pos[indices_over_thr]

    cartesian_pos = barycenter_3D(pos_over_thr, q_over_thr)

    return np.arctan2(cartesian_pos[1], cartesian_pos[0]), cartesian_pos[2]


def calculate_mean_var_phi(phis, qs):
    diff_sign = min(phis ) < 0 < max(phis)
    if diff_sign & (np.abs(np.min(phis))>np.pi/2):
        phis[phis<0] = np.pi + np.pi + phis[phis<0]
    mean_phi = np.average(phis, weights=qs)
    var_phi  = np.average((phis-mean_phi)**2, weights=qs)

    return mean_phi, var_phi


def calculate_z_opening(zs):
    delta_z = max(zs) - min(zs)

    return delta_z


def calculate_phi_opening(phis):
    diff_sign = min(phis ) < 0 < max(phis)
    if diff_sign & (np.abs(np.min(phis))>np.pi/2):
        phis[phis<0] = np.pi + np.pi + phis[phis<0]

    return max(phis) - min(phis)


def filter_SiPMs_by_charge(sns_ids, tot_charges, threshold):

    indices_over_thr = (tot_charges > threshold)
    sns_over_thr = sns_ids[indices_over_thr]
    charges_over_thr = tot_charges[indices_over_thr]

    return sns_over_thr, charges_over_thr


def find_SiPMs_over_thresholds(current_charge, threshold):

    sns_dict = list(current_charge.values())[0]
    tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
    sns_ids = np.array(list(sns_dict.keys()))

    return filter_SiPMs_by_charge(sns_ids, tot_charges, threshold)


def reconstruct_pos(sipms_info: Sequence[Tuple[float, float, float, float]], r_thr: float, phi_z_thr: float, rpos_table: float) -> (float, float, float):
    sipms_info = np.array(sipms_info)
    q = sipms_info[:,3]

    ### first, find R
    sel_r   = q > r_thr
    q_r     = q[sel_r]
    sipms_r = sipms_info[sel_r]

    if len(q_r) == 0:
        return None, None, None

    pos_phi = np.arctan2(sipms_r[:,1], sipms_r[:,0])
    var_phi = None

    mean_phi, var_phi = calculate_mean_var_phi(pos_phi, q_r)
    reco_r   = rpos_table(np.sqrt(var_phi)).value

    ### Now, find z and phi
    sel_phi_z   = q > phi_z_thr
    q_phi_z     = q[sel_phi_z]
    sipms_phi_z = sipms_info[sel_phi_z]

    if len(q_phi_z) == 0:
        return None, None, None

    pos_phi_z =  sipms_phi_z[:, 0:3]

    reco_cart = barycenter_3D(pos_phi_z, q_phi_z)
    reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])
    reco_z    = reco_cart[2]

    return reco_r, reco_phi, reco_z


def find_phi_z_opening(sipms: Sequence[Tuple[float, float, float, float]])-> (float, float):
    """
    This function takes as an input a list of tetravectors (x, y, z, q) with
    the position and charge of the SiPMs under consideration and returns the spread
    of the sensor response in phi and z
    """
    phis = np.arctan2(sipms[:, 1], sipms[:, 0])
    zs   = sipms[:, 2]

    phi_spread = calculate_phi_opening(phis)
    z_spread   = calculate_z_opening(zs)

    return phi_spread, z_spread


def find_reco_pos(current_charge: Dict[int, Dict[int, Waveform]], r_threshold: float, zphi_threshold:float, rpos_table, db) -> (float, float, float):

    ### read sensor positions from database
    DataSiPM = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    ### first, find R
    sns_over_thr, charges_over_thr = find_SiPMs_over_thresholds(current_charge, r_threshold)

    if len(charges_over_thr) == 0:
        return None, None, None

    pos_phi = np.arctan2(DataSiPM_idx.loc[sns_over_thr].Y.values, DataSiPM_idx.loc[sns_over_thr].X.values)
    pos_phi = np.array(pos_phi)
    var_phi = None

    mean_phi, var_phi = calculate_mean_var_phi(pos_phi, charges_over_thr)

    ### Now, find z and phi
    sns_over_thr, charges_over_thr = find_SiPMs_over_thresholds(current_charge, zphi_threshold)

    if len(charges_over_thr) == 0:
        return None, None, None


    pos = np.array(DataSiPM_idx.loc[sns_over_thr].X.values, DataSiPM_idx.loc[sns_over_thr].Y.values, DataSiPM_idx.loc[sns_over_thr].Z.values).transpose()

    reco_r    = rpos_table(np.sqrt(var_phi)).value
    reco_cart = barycenter_3D(pos, charges_over_thr)
    reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])
    reco_z    = reco_cart[2]

    return reco_r, reco_phi, reco_z


def select_coincidences(current_charge: Dict[int, Dict[int, Waveform]], threshold, charge_range: Tuple[float, float], sens_pos: Dict[int, Tuple[float, float, float]], particle_dict: Dict[int, Sequence[MCParticle]]) -> (Sequence[Tuple[float, float, float, float]], Sequence[Tuple[float, float, float, float]]):

    sns_over_thr, charges_over_thr = find_SiPMs_over_thresholds(current_charge, threshold)
    if len(sns_over_thr) == 0:
        return [], [], None, None

    ### Find the SiPM with maximum charge. The set if sensors around it are labelled as 1
    ### The sensors on the opposite emisphere are labelled as 2.
    max_sns = sns_over_thr[np.argmax(charges_over_thr)]
    max_pos = sens_pos[max_sns]

    sipms1, sipms2 = [], []

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos = sens_pos[sns_id]
        scalar_prod = sum(a*b for a, b in zip(pos, max_pos))
        pos_q = (pos[0], pos[1], pos[2], charge)
        if scalar_prod > 0.:
            sipms1.append(np.array(pos_q))
        else:
            sipms2.append(np.array(pos_q))

    if len(sipms1) == 0 or len(sipms2) == 0:
        return [], [], None, None

    sipms1 = np.array(sipms1)
    sipms2 = np.array(sipms2)

    q1 = sum(sipms1[:,3])
    q2 = sum(sipms2[:,3])

    sel1 = (q1 > charge_range[0]) & (q1 < charge_range[1])
    sel2 = (q2 > charge_range[0]) & (q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], None, None


    ## find the first interactions of the primary gamma(s)
    tvertex_pos = tvertex_neg = -1
    min_pos, min_neg = None, None

    for _, part in particle_dict.items():
        if part.name == 'e-':
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                mother = particle_dict[part.mother_indx]
                if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                    if mother.p[1] > 0.:
                        if tvertex_pos < 0 or part.initial_vertex[3] < tvertex_pos:
                            min_pos     = part.initial_vertex[0:3]
                            tvertex_pos = part.initial_vertex[3]
                    else:
                        if tvertex_neg < 0 or part.initial_vertex[3] < tvertex_neg:
                            min_neg     = part.initial_vertex[0:3]
                            tvertex_neg = part.initial_vertex[3]


        elif part.name == 'gamma' and part.primary:
            if len(part.hits) > 0:
                if part.p[1] > 0.:
                    times         = [h.time for h in part.hits]
                    hit_positions = [h.pos for h in part.hits]
                    min_time      = min(times)
                    if min_time < tvertex_pos:
                        min_pos     = hit_positions[times.index(min_time)]
                        tvertex_pos = min_time
                else:
                    times         = [h.time for h in part.hits]
                    hit_positions = [h.pos for h in part.hits]
                    min_time      = min(times)
                    if min_time < tvertex_neg:
                        min_neg     = hit_positions[times.index(min_time)]
                        tvertex_neg = min_time

    if min_pos is None or min_neg is None:
        print("Cannot find two true gamma interactions for this event")
        return [], [], None, None


    pos_true1, pos_true2 = [], []
    scalar_prod = sum(a*b for a, b in zip(min_pos, max_pos))
    if scalar_prod > 0:
        pos_true1 = min_pos
        pos_true2 = min_neg
    else:
        pos_true1 = min_neg
        pos_true2 = min_pos

    return sipms1, sipms2, pos_true1, pos_true2
