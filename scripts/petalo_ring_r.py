import os
import sys
import math
import tables as tb
import numpy  as np

from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.io.mcinfo_io           import load_mcsensor_response
from   invisible_cities.io.mcinfo_io           import read_mcsns_response
from   invisible_cities.io.mcinfo_io           import read_mcinfo

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

start = int(sys.argv[1])
numb = int(sys.argv[2])
nsteps = int(sys.argv[3])
thr_start = int(sys.argv[4])

base_file = '/home/paolafer/SimMC/pet/full_ring_depth3cm_pitch6mm.{0}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/full_ring_depth3cm_pitch6mm_phot_r_{0}_{1}_{2}'.format(start,numb,thr_start)

true_r1 = [[] for i in range(0, nsteps)]
true_r2 = [[] for i in range(0, nsteps)]

touched_sipms1_int  = [[] for i in range(0, nsteps)]
touched_sipms2_int  = [[] for i in range(0, nsteps)]
touched_sipms1_ext  = [[] for i in range(0, nsteps)]
touched_sipms2_ext  = [[] for i in range(0, nsteps)]
ratio1 = [[] for i in range(0, nsteps)]
ratio2 = [[] for i in range(0, nsteps)]

rad_int = 100.
rad_ext = 130.
average_rad = (rad_int + rad_ext)/2.

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
            r1 = r2 = 0.
            phi1 = phi2 = 0.
    
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
            
                for threshold in range(0, nsteps):                
                    indices_over_thr = (tot_charges > threshold + thr_start)
                    sns_over_thr = sns_ids[indices_over_thr]
                    charges_over_thr = tot_charges[indices_over_thr]
                
                    #max_sns = sns_over_thr[np.argmax(charges_over_thr)]

                    ### Find the closest SiPM to the true average point
                    if energy1:
                        min_sns1 = find_closest_sipm(aveX1, aveY1, aveZ1,
                                                     sensor_pos, sns_over_thr, charges_over_thr)
                    if energy2:
                        min_sns2 = find_closest_sipm(aveX2, aveY2, aveZ2,
                                                     sensor_pos, sns_over_thr, charges_over_thr)
                
                    ampl_1 = ampl_2 =  0.
                    count_1 = count_2 = 0
                    pos_1_int = []
                    pos_2_int = []
                    pos_1_ext = []
                    pos_2_ext = []
                    q_1_int = []
                    q_2_int = []
                    q_1_ext = []
                    q_2_ext = []
                    count_1_int = count_2_int = 0
                    count_1_ext = count_2_ext = 0
                
                    for sns_id, charge in zip(sns_over_thr, charges_over_thr):                       
                        pos     = sensor_pos[sns_id]
                        pos_cyl = sensor_pos_cyl[sns_id]
                        #pos_max = sensor_pos[max_sns]
                        if energy1:
                            pos_closest = sensor_pos[min_sns1]
                            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                            if scalar_prod > 0.:
                                ampl_1 += charge 
                                if pos_cyl[0] > average_rad:
                                    pos_1_ext.append(pos_cyl)
                                    q_1_ext.append(charge)
                                    count_1_ext += 1
                                else:
                                    pos_1_int.append(pos_cyl)
                                    q_1_int.append(charge)
                                    count_1_int += 1
                                    
                        if energy2:
                            pos_closest = sensor_pos[min_sns2]
                            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                            if scalar_prod > 0.:
                                ampl_2 += charge
                                if pos_cyl[0] > average_rad:
                                    pos_2_ext.append(pos_cyl)
                                    q_2_ext.append(charge)
                                    count_2_ext += 1
                                else:
                                    pos_2_int.append(pos_cyl)
                                    q_2_int.append(charge)
                                    count_2_int += 1
                
                
                    if ampl_1 != 0 and sum(q_1_int) != 0. and sum(q_1_ext) != 0:
                        ratio1[threshold].append(sum(q_1_int)/sum(q_1_ext))
                        touched_sipms1_int[threshold].append(count_1_int)
                        touched_sipms1_ext[threshold].append(count_1_ext)
                        r1 = np.sqrt(aveX1*aveX1 + aveY1*aveY1)
                        phi1 = np.arctan2(aveY1, aveX1)
                        true_r1[threshold].append(r1)
                    else:
                        ratio1[threshold].append(1.e9)
                        touched_sipms1_int[threshold].append(1.e9)
                        touched_sipms1_ext[threshold].append(1.e9)
                        true_r1[threshold].append(1.e9)
                    if ampl_2 != 0 and sum(q_2_int) != 0. and sum(q_2_ext) != 0: 
                        ratio2[threshold].append(sum(q_2_int)/sum(q_2_ext))
                        touched_sipms2_int[threshold].append(count_2_int)
                        touched_sipms2_ext[threshold].append(count_2_ext)
                        r2 = np.sqrt(aveX2*aveX2 + aveY2*aveY2)
                        true_r2[threshold].append(r2)
                    else:
                        ratio2[threshold].append(1.e9)
                        touched_sipms2_int[threshold].append(1.e9)
                        touched_sipms2_ext[threshold].append(1.e9)
                        true_r2[threshold].append(1.e9)
                        

a_true_r1_0 = np.array(true_r1[0])
a_true_r2_0 = np.array(true_r2[0])
a_ratio1_0 = np.array(ratio1[0])
a_ratio2_0 = np.array(ratio2[0])
a_touched_sipms1_int_0 = np.array(touched_sipms1_int[0])
a_touched_sipms2_int_0 = np.array(touched_sipms2_int[0])
a_touched_sipms1_ext_0 = np.array(touched_sipms1_ext[0])
a_touched_sipms2_ext_0 = np.array(touched_sipms2_ext[0])

a_true_r1_1 = np.array(true_r1[1])
a_true_r2_1 = np.array(true_r2[1])
a_ratio1_1 = np.array(ratio1[1])
a_ratio2_1 = np.array(ratio2[1])
a_touched_sipms1_int_1 = np.array(touched_sipms1_int[1])
a_touched_sipms2_int_1 = np.array(touched_sipms2_int[1])
a_touched_sipms1_ext_1 = np.array(touched_sipms1_ext[1])
a_touched_sipms2_ext_1 = np.array(touched_sipms2_ext[1])

a_true_r1_2 = np.array(true_r1[2])
a_true_r2_2 = np.array(true_r2[2])
a_ratio1_2 = np.array(ratio1[2])
a_ratio2_2 = np.array(ratio2[2])
a_touched_sipms1_int_2 = np.array(touched_sipms1_int[2])
a_touched_sipms2_int_2 = np.array(touched_sipms2_int[2])
a_touched_sipms1_ext_2 = np.array(touched_sipms1_ext[2])
a_touched_sipms2_ext_2 = np.array(touched_sipms2_ext[2])

a_true_r1_3 = np.array(true_r1[3])
a_true_r2_3 = np.array(true_r2[3])
a_ratio1_3 = np.array(ratio1[3])
a_ratio2_3 = np.array(ratio2[3])
a_touched_sipms1_int_3 = np.array(touched_sipms1_int[3])
a_touched_sipms2_int_3 = np.array(touched_sipms2_int[3])
a_touched_sipms1_ext_3 = np.array(touched_sipms1_ext[3])
a_touched_sipms2_ext_3 = np.array(touched_sipms2_ext[3])

a_true_r1_4 = np.array(true_r1[4])
a_true_r2_4 = np.array(true_r2[4])
a_ratio1_4 = np.array(ratio1[4])
a_ratio2_4 = np.array(ratio2[4])
a_touched_sipms1_int_4 = np.array(touched_sipms1_int[4])
a_touched_sipms2_int_4 = np.array(touched_sipms2_int[4])
a_touched_sipms1_ext_4 = np.array(touched_sipms1_ext[4])
a_touched_sipms2_ext_4 = np.array(touched_sipms2_ext[4])


np.savez(evt_file, a_true_r1_0=a_true_r1_0, a_true_r2_0=a_true_r2_0, a_true_r1_1=a_true_r1_1, a_true_r2_1=a_true_r2_1, a_true_r1_2=a_true_r1_2, a_true_r2_2=a_true_r2_2, a_true_r1_3=a_true_r1_3, a_true_r2_3=a_true_r2_3, a_true_r1_4=a_true_r1_4, a_true_r2_4=a_true_r2_4, a_ratio1_0=a_ratio1_0, a_ratio2_0=a_ratio2_0, a_touched_sipms1_int_0=a_touched_sipms1_int_0, a_touched_sipms2_int_0=a_touched_sipms2_int_0, a_touched_sipms1_ext_0=a_touched_sipms1_ext_0, a_touched_sipms2_ext_0=a_touched_sipms2_ext_0, a_ratio1_1=a_ratio1_1, a_ratio2_1=a_ratio2_1, a_touched_sipms1_int_1=a_touched_sipms1_int_1, a_touched_sipms2_int_1=a_touched_sipms2_int_1, a_touched_sipms1_ext_1=a_touched_sipms1_ext_1, a_touched_sipms2_ext_1=a_touched_sipms2_ext_1, a_ratio1_2=a_ratio1_2, a_ratio2_2=a_ratio2_2, a_touched_sipms1_int_2=a_touched_sipms1_int_2, a_touched_sipms2_int_2=a_touched_sipms2_int_2, a_touched_sipms1_ext_2=a_touched_sipms1_ext_2, a_touched_sipms2_ext_2=a_touched_sipms2_ext_2, a_ratio1_3=a_ratio1_3, a_ratio2_3=a_ratio2_3, a_touched_sipms1_int_3=a_touched_sipms1_int_3, a_touched_sipms2_int_3=a_touched_sipms2_int_3, a_touched_sipms1_ext_3=a_touched_sipms1_ext_3, a_touched_sipms2_ext_3=a_touched_sipms2_ext_3, a_ratio1_4=a_ratio1_4, a_ratio2_4=a_ratio2_4, a_touched_sipms1_int_4=a_touched_sipms1_int_4, a_touched_sipms2_int_4=a_touched_sipms2_int_4, a_touched_sipms1_ext_4=a_touched_sipms1_ext_4, a_touched_sipms2_ext_4=a_touched_sipms2_ext_4)

