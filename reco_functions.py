import numpy as np
import pandas as pd
import tables as tb
import random
import sys
import os

from invisible_cities.core.system_of_units_c import units

from   antea.utils.table_functions import load_rpos
import antea.database.load_db as db

from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge

from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict


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

def find_SiPMs_over_thresholds(current_charge, threshold):

    sns_ids = np.array(current_charge.columns.tolist())
    tot_charges = np.array([current_charge[sns_id].values[0] for sns_id in sns_ids])
    
    indices_over_thr = (tot_charges > threshold)
    sns_over_thr = sns_ids[indices_over_thr]
    charges_over_thr = tot_charges[indices_over_thr]

    return sns_over_thr, charges_over_thr


Rpos_table = load_rpos(rpos_file,
                       group = "Radius",
                       node  = "f{}pes200bins".format(rpos_threshold))


def find_reco_pos(current_charge, r_threshold, zphi_threshold):

    ### read sensor positions from database
    DataSiPM = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    ### first, find R
    sns_over_thr, charges_over_thr = find_SiPMs_over_thresholds(current_charge, r_threshold)
  
    if len(charges_over_thr) == 0:
        return

    q = []
    pos_phi = []

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):                       
        x = DataSiPM_idx.loc[sns_id].X
        y = DataSiPM_idx.loc[sns_id].Y
        z = DataSiPM_idx.loc[sns_id].Z
        
        pos_cyl = (np.sqrt(x*x + y*y), np.arctan2(y, x), z)
        q.append(charge)
        pos_phi.append(pos_cyl[1])

    var_phi = None

    mean_phi = np.average(pos_phi, weights=q)
    var_phi  = np.average((pos_phi-mean_phi)**2, weights=q)

    ### Now, find z and phi
    sns_over_thr, charges_over_thr = find_SiPMs_over_thresholds(current_charge, zphi_threshold)

    if len(charges_over_thr) == 0:
        return


    q   = []
    pos = []
                
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        x = DataSiPM_idx.loc[sns_id].X
        y = DataSiPM_idx.loc[sns_id].Y
        z = DataSiPM_idx.loc[sns_id].Z

        pos     = [x, y, z]
        pos_cyl = (np.sqrt(x*x + y*y), np.arctan2(y, x), z)

        pos.append(pos)
        q.append(charge)
                                         
        
    reco_r    = Rpos(np.sqrt(var_phi)).value 
    reco_cart = barycenter_3D(pos, q)
    reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])
    reco_z    = reco_cart[2]
