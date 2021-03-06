import os
import sys
import math
import tables as tb
import numpy  as np

from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.io.mcinfo_io           import load_mcsensor_response
from   invisible_cities.io.mcinfo_io           import read_mcsns_response
from   invisible_cities.io.mcinfo_io           import read_mcinfo

import invisible_cities.reco.dst_functions as dstf
from invisible_cities.core.exceptions     import SipmEmptyList
from invisible_cities.core.exceptions     import SipmZeroCharge

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
        [x2, y2, z2]
        ...
        [xs, ys, zs])
       qs = vector (q1, q2...qs) --> (1xn)
        """

    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge
        
    return np.average(pos, weights=qs, axis=0)


start = int(sys.argv[1])
numb = int(sys.argv[2])
r_thr = int(sys.argv[3])
bins = int(sys.argv[4])


base_file = '/home/paolafer/SimMC/pet/full_ring_depth5cm_pitch4mm.{0}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/full_ring_depth5cm_pitch4mm_phot_reco_r_{0}_{1}_{2}_{3}'.format(start, numb, r_thr, bins)

rpos_file = '/home/paolafer/analysis/r_table_thr4pes_4mm_depth5cm.h5'

node_name = 'f{0}pes{1}bins'.format(r_thr, bins)
Rpos = dstf.load_rpos(rpos_file,
                      group = "Radius",
                      node  = node_name)

rpos_threshold = r_thr

events1 = []
events2 = []

true_r1 = []
true_r2 = []

reco_r1 = []
reco_r2 = []

rad_int = 100.
rad_ext = 130.
average_rad = (rad_int + rad_ext)/2.

evt_no_pos = 0

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
            sensor_pos_cyl[sipm[0]] = (np.sqrt(sipm[1]*sipm[1] + sipm[2]*sipm[2]), np.arctan2(sipm[2], sipm[1]), sipm[3])  
    
        events_in_file = len(h5in.root.MC.extents)
        for evt in range(events_in_file):

            this_event_dict = read_mcinfo(h5in, (evt, evt+1))
            this_event_wvf = read_mcsns_response(h5in, (evt, evt+1))
            event_number = h5in.root.MC.extents[evt]['evt_number']
            part_dict = list(this_event_dict.values())[0]

            energy1 = energy2 = 0.
            aveX1 = aveY1 = aveZ1 = 0.
            aveX2 = aveY2 = aveZ2 = 0.

            both = 0
            interest = False

            for indx, part in part_dict.items():
                if part.name == 'e-' :
                    mother = part_dict[part.mother_indx]
                    if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                        if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6):
                            if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                                interest = True
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
            
            
            if interest:
            
                if energy1 != 0.:
                    aveX1 = aveX1 / energy1
                    aveY1 = aveY1 / energy1
                    aveZ1 = aveZ1 / energy1
                else:
                    aveX1 = 1.e9
                    aveY1 = 1.e9
                    aveZ1 = 1.e9

                                
                if energy2 != 0.:
                    aveX2 = aveX2 / energy2
                    aveY2 = aveY2 / energy2
                    aveZ2 = aveZ2 / energy2
                else:
                    aveX2 = 1.e9
                    aveY2 = 1.e9
                    aveZ2 = 1.e9

            
            sns_dict = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
            sns_ids = np.array(list(sns_dict.keys()))
            
            ### extract the ratios with given threshold
            threshold = rpos_threshold
            indices_over_thr = (tot_charges > threshold)
            sns_over_thr = sns_ids[indices_over_thr]
            charges_over_thr = tot_charges[indices_over_thr]
            if len(charges_over_thr) == 0:
                evt_no_pos += 1
                continue

            q_1_int = []
            q_2_int = []
            q_1_ext = []
            q_2_ext = []

            if energy1:
                sns_closest1 = find_closest_sipm(aveX1, aveY1, aveZ1,
                                                 sensor_pos, sns_over_thr, charges_over_thr)
            if energy2:
                sns_closest2 = find_closest_sipm(aveX2, aveY2, aveZ2,
                                                 sensor_pos, sns_over_thr, charges_over_thr)
                
            for sns_id, charge in zip(sns_over_thr, charges_over_thr):                       
                pos     = sensor_pos[sns_id]
                pos_cyl = sensor_pos_cyl[sns_id]

                if energy1:
                    pos_closest = sensor_pos[sns_closest1]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.: 
                        if pos_cyl[0] > average_rad:
                            q_1_ext.append(charge)
                        else:
                            q_1_int.append(charge)                       
                if energy2:
                    pos_closest = sensor_pos[sns_closest2]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.:
                        if pos_cyl[0] > average_rad:
                            q_2_ext.append(charge)
                        else:
                            q_2_int.append(charge)
            
            ratio1 = ratio2 = None
            if sum(q_1_int) > 0. and sum(q_1_ext) > 0.:
                ratio1 = sum(q_1_int)/sum(q_1_ext)
            if sum(q_2_int) > 0. and sum(q_2_ext) > 0.:
                ratio2 = sum(q_2_int)/sum(q_2_ext)
            
                
            if ratio1:  
                reco_r = Rpos(ratio1).value 
                reco_r1.append(reco_r)
                true_r1.append(np.sqrt(aveX1*aveX1 + aveY1*aveY1))
                events1.append(event_number)
            else:
                reco_r1.append(1.e9)
                true_r1.append(1.e9)
                events1.append(event_number)
            if ratio2: 
                reco_r = Rpos(ratio2).value 
                reco_r2.append(reco_r)
                true_r2.append(np.sqrt(aveX2*aveX2 + aveY2*aveY2))
                events2.append(event_number)
            else:
                reco_r2.append(1.e9)
                true_r2.append(1.e9)
                events2.append(1.e9)


print('Events with no charge above threshold = {}'.format(evt_no_pos))

a_true_r1 = np.array(true_r1)
a_true_r2 = np.array(true_r2)

a_reco_r1 = np.array(reco_r1)
a_reco_r2 = np.array(reco_r2)

a_events1 = np.array(events1)
a_events2 = np.array(events2)

np.savez(evt_file, a_true_r1=a_true_r1, a_true_r2=a_true_r2, a_reco_r1=a_reco_r1, a_reco_r2=a_reco_r2, a_events1=a_events1, a_events2=a_events2)

