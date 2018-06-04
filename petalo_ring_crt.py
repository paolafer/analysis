import numpy as np
import pandas as pd
import tables as tb
import random
import sys
import os

from invisible_cities.core.system_of_units_c import units

from invisible_cities.io.mcinfo_io           import load_mcsensor_response
from invisible_cities.io.mcinfo_io           import read_mcsns_response
from invisible_cities.io.mcinfo_io           import read_mcinfo
import invisible_cities.reco.dst_functions as dstf

from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge


def find_closest_sipm(x, y, z, sensor_pos, sns_over_thr, charges_over_thr):
    ### Find the closest SiPM to the true average point
    min_dist = 1.e9
    min_sns = 0
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos = sensor_pos[sns_id]
        dist = np.linalg.norm(np.subtract((x, y, z), pos))
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

def calculate_charge_ratio(q_int, q_ext, r_thr):
    
    indices_int_over_thr = (q_int > r_thr)
    q_int_over_thr = q_int[indices_int_over_thr]
    indices_ext_over_thr = (q_ext > r_thr)
    q_ext_over_thr = q_ext[indices_ext_over_thr]
        
    if sum(q_int_over_thr) and sum(q_ext_over_thr):
        return sum(q_int_over_thr) / sum(q_ext_over_thr)
    else:
        return None

def calculate_phi_z_pos(pos, q, phi_z_thr):
    
    indices_over_thr = (q > phi_z_thr)
    q_over_thr       = q[indices_over_thr]
    pos_over_thr     = pos[indices_over_thr]
    
    cartesian_pos = barycenter_3D(pos_over_thr, q_over_thr)
    
    return np.arctan2(cartesian_pos[1], cartesian_pos[0]), cartesian_pos[2]

start = int(sys.argv[1])
numb = int(sys.argv[2])

base_file = '/home/paolafer/SimMC/pet/full_ring_depth3cm_pitch4mm.{}.pet.h5'
daq_file = "/home/paolafer/SimMC/pet/full_ring_depth3cm_pitch4mm_DAQ.{}.h5"

evt_file = '/home/paolafer/analysis/petalo/full_ring_depth3cm_pitch4mm_phot_CRT_{0}_{1}'.format(start,numb)

rpos_file = '/home/paolafer/analysis/r_table_var_thr_4mm.h5'
Rpos = dstf.load_rpos(rpos_file,
                      group = "Radius",
                      node  = "f5pes150bins")

rad_int = 100. # mm
rad_ext = 130. # mm
average_rad = (rad_int + rad_ext)/2.

rad_threshold = 5
phi_z_threshold = 5

speed_in_vacuum = 0.299792458 # mm / ps
ave_speed_in_LXe = 0.210 # mm / ps

time_diff = []

rads_1 = []
phis_1 = []
zs_1 = []
rads_2 = []
phis_2 = []
zs_2 = []

for ifile in range(start, start+numb):

    file_name = base_file.format(ifile)
    try:
        load_mcsensor_response(file_name, (0,1))
    except ValueError:
        continue
    except OSError:
         continue
    print('Analyzing file {0}'.format(file_name))

    with tb.open_file(file_name, mode='r') as h5in:

        sipms = h5in.root.MC.sensor_positions[:]
        sensor_pos = {}
        sensor_pos_cyl = {}
        for sipm in sipms:
            sensor_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
            sensor_pos_cyl[sipm[0]] = (np.sqrt(sipm[1]*sipm[1] + sipm[2]*sipm[2]), \
                                           np.arctan2(sipm[2], sipm[1]), sipm[3])  

### Read the SiPM response passed through the DAQ
        daq_file_name = daq_file.format(ifile)
        df_charge     = pd.read_hdf(daq_file_name, key='MC')
        df_tof        = pd.read_hdf(daq_file_name, key='MC_tof')   
        df_subthr_TDC = pd.read_hdf(daq_file_name, key='subth_TDC_L1')  
    
        events_in_file = len(h5in.root.MC.extents)
        for evt in range(events_in_file):
            this_event_dict = read_mcinfo(h5in, (evt, evt+1))
            this_event_wvf = read_mcsns_response(h5in, (evt, evt+1))
            event_number = h5in.root.MC.extents[evt]['evt_number']

            part_dict = list(this_event_dict.values())[0]

            energy1 = energy2 = 0.
            aveX1 = aveY1 = aveZ1 = 0.
            aveX2 = aveY2 = aveZ2 = 0.
            r1 = r2 = 0.
            phi1 = phi2 = 0.
    
            both = 0
            #interest = False
            
            for indx, part in part_dict.items():
                if part.name == 'e-' :
                    mother = part_dict[part.mother_indx]
                    if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                        if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6):
                            if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                                #interest = True
                                both += 1

                                if mother.p[1] > 0.:
                                    for h in part.hits:
                                        aveX1 += h.X * h.E
                                        aveY1 += h.Y * h.E
                                        aveZ1 += h.Z * h.E 
                                        energy1 += h.E                               
                                else:
                                    for h in part.hits:
                                        aveX2 += h.X * h.E
                                        aveY2 += h.Y * h.E
                                        aveZ2 += h.Z * h.E 
                                        energy2 += h.E

            if both == 2:
                print('Event number {}'.format(event_number))
                if energy1 != 0.:
                    aveX1 = aveX1 / energy1
                    aveY1 = aveY1 / energy1
                    aveZ1 = aveZ1 / energy1
                else:
                    print('Event {} to be inspected'.format(event_number))
                                
                if energy2 != 0.:
                    aveX2 = aveX2 / energy2
                    aveY2 = aveY2 / energy2
                    aveZ2 = aveZ2 / energy2
                else:
                    print('Event {} to be inspected'.format(event_number))
                    
                current_charge  = df_charge[evt:evt+1]
                current_tof     = df_tof[evt:evt+1]
                current_sub_TDC = df_subthr_TDC[evt:evt+1]
                
                over_thr = {}
                for sns_id in current_charge.columns.tolist():   
                    charge = current_charge[sns_id].values[0]
                    if charge > 0.:
                        over_thr[sns_id] = charge
                    
                min_sns1 = find_closest_sipm(aveX1, aveY1, aveZ1,
                                             sensor_pos, over_thr.keys(), over_thr.values())
                min_sns2 = find_closest_sipm(aveX2, aveY2, aveZ2,
                                             sensor_pos, over_thr.keys(), over_thr.values())

### Divide SiPMs in four subsets: 1/2 --> interaction labeled as 1/2, int/ext --> sensors in 
### internal or external face
                sipms_1_int = {}
                sipms_1_ext = {}
                sipms_2_int = {}
                sipms_2_ext = {}
    
                for sns_id, charge in over_thr.items():                       
                    pos     = sensor_pos[sns_id]
                    pos_cyl = sensor_pos_cyl[sns_id]
                    
                    pos_closest = sensor_pos[min_sns1]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.: 
                        if pos_cyl[0] > average_rad:
                            sipms_1_ext[pos] = charge
                        else:
                            sipms_1_int[pos] = charge
                            
                    pos_closest = sensor_pos[min_sns2]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.:
                        if pos_cyl[0] > average_rad:
                            sipms_2_ext[pos] = charge
                        else:
                            sipms_2_int[pos] = charge
                            
### Calculate radial positions                            
                ratio_1 = calculate_charge_ratio(np.array(list(sipms_1_int.values())), \
                                             np.array(list(sipms_1_ext.values())), rad_threshold)
                if ratio_1 == None:
                    print('No charge 1 over threshold for event {}'.format(event_number))
                    continue
                rad_1 = Rpos(ratio_1).value
    
                ratio_2 = calculate_charge_ratio(np.array(list(sipms_2_int.values())), \
                                             np.array(list(sipms_2_ext.values())), rad_threshold)
                if ratio_2 == None:
                    print('No charge 2 over threshold for event {}'.format(event_number))
                    continue
                rad_2 = Rpos(ratio_2).value

### Calculate phi and z positions
                if rad_1 > average_rad:
                    pos_1 = sipms_1_ext.keys()
                    q_1   = sipms_1_ext.values()
                else:
                    pos_1 = sipms_1_int.keys()
                    q_1   = sipms_1_int.values()
                    
                phi_1, z_1 = calculate_phi_z_pos(np.array(list(pos_1)), np.array(list(q_1)), phi_z_threshold)
                
                if rad_2 > average_rad:
                    pos_2 = sipms_2_ext.keys()
                    q_2   = sipms_2_ext.values()
                else:
                    pos_2 = sipms_2_int.keys()
                    q_2   = sipms_2_int.values()
                    
                phi_2, z_2 = calculate_phi_z_pos(np.array(list(pos_2)), np.array(list(q_2)), phi_z_threshold)
                
### Calculate also cartesian coordinates                
                pos_1 = np.array([rad_1 * np.cos(phi_1), rad_1 * np.sin(phi_1), z_1]) 
                pos_2 = np.array([rad_2 * np.cos(phi_2), rad_2 * np.sin(phi_2), z_2])

### Fiducialize
#                if rad_1 > 125. or rad_1 < 105. or z_1 > 17.5 or z_1 < -17.5:
#                    continue
#                if rad_2 > 125. or rad_2 < 105. or z_2 > 17.5 or z_2 < -17.5:
#                    continue

### Calculate time first photoelectrons
                tof = {}
                for sns_id in current_tof.columns.tolist():   
                    timestamp = current_tof[sns_id].values[0]
                    if timestamp > 0.:
                        tof[sns_id] = timestamp
                
                tof_1 = {}
                tof_2 = {}

                for sns_id, t in tof.items():                       
                    pos     = sensor_pos[sns_id]
                    pos_cyl = sensor_pos_cyl[sns_id]
                    
                    pos_closest = sensor_pos[min_sns1]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.: 
                        tof_1[sns_id] = t
                            
                    pos_closest = sensor_pos[min_sns2]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.:
                        tof_2[sns_id] = t
                        
                min_sns_1 = min(tof_1, key=tof_1.get)
                min_time_1 = tof_1[min_sns_1]
                min_sns_2 = min(tof_2, key=tof_2.get)
                min_time_2 = tof_2[min_sns_2]
                
### Distance between interaction point and sensor detecting first photon
                dist1 = np.linalg.norm(pos_1 - sensor_pos[min_sns_1])
                dist2 = np.linalg.norm(pos_2 - sensor_pos[min_sns_2])
        
### Distance of the interaction point from the centre of the system
                inter1 = np.linalg.norm(pos_1)
                inter2 = np.linalg.norm(pos_2)
        
### Calculate Delta t = 1/2 * (t1 - t2 - Delta d_g / c - Delta d_p / v_p)
                
                delta_t = 1/2 * (min_time_1 - min_time_2 - (inter1 - inter2) / speed_in_vacuum \
                                     - (dist1 - dist2) / ave_speed_in_LXe)
                time_diff.append(delta_t)

                rads_1.append(rad_1)
                rads_2.append(rad_2)
                phis_1.append(phi_1)
                phis_2.append(phi_2)
                zs_1.append(z_1)
                zs_2.append(z_2)


a_time_diff = np.array(time_diff)
a_rads_1 = np.array(rads_1)
a_rads_2 = np.array(rads_2)
a_phis_1 = np.array(phis_1)
a_phis_2 = np.array(phis_2)
a_zs_1 = np.array(zs_1)
a_zs_2 = np.array(zs_2)

np.savez(evt_file, a_time_diff=a_time_diff, a_rads_1=a_rads_1, a_rads_2=a_rads_2, a_phis_1=a_phis_1, a_phis_2=a_phis_2, a_zs_1=a_zs_1, a_zs_2=a_zs_2)
