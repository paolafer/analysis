import os
import sys
import math
import tables as tb
import numpy  as np

from   invisible_cities.core.system_of_units_c import units
from   antea.io.mc_io                          import read_mcsns_response
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
nsteps = int(sys.argv[3])
thr_start = int(sys.argv[4])
fid = int(sys.argv[5])

base_file = '/home/paolafer/SimMC/pet/full_ring_depth4mm_one_face.{0}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/full_ring_depth3cm_pitch4mm_one_face_phot_r_{0}_{1}_{2}_{3}fid'.format(start,numb,thr_start, fid)

true_r1 = [[] for i in range(0, nsteps)]
true_r2 = [[] for i in range(0, nsteps)]

touched_sipms1_ext  = [[] for i in range(0, nsteps)]
touched_sipms2_ext  = [[] for i in range(0, nsteps)]

var_phi1 = [[] for i in range(0, nsteps)]
var_phi2 = [[] for i in range(0, nsteps)]

var_z1 = [[] for i in range(0, nsteps)]
var_z2 = [[] for i in range(0, nsteps)]

z_min = -32
z_max = 32

for ifile in range(start, start+numb):

    file_name = base_file.format(ifile)
    try:
        read_mcsns_response(file_name, (0,1))
        print('Analysing file {}'.format(file_name))
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
            this_event_wvf = read_mcsns_response(file_name, (evt, evt+1))
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
                    
                    pos_1phi = []
                    pos_2phi = []
                    pos_1z = []
                    pos_2z = []
                    
                    q_1_ext = []
                    q_2_ext = []

                    pos_1 = []
                    pos_2 = []

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
                                pos_1.append(pos_cyl)
                                pos_1phi.append(pos_cyl[1])
                                pos_1z.append(pos_cyl[2])
                                q_1_ext.append(charge)
                                count_1_ext += 1
                                    
                        if energy2:
                            pos_closest = sensor_pos[min_sns2]
                            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                            if scalar_prod > 0.:
                                ampl_2 += charge
                                pos_2.append(pos_cyl)
                                pos_2phi.append(pos_cyl[1])
                                pos_2z.append(pos_cyl[2])
                                q_2_ext.append(charge)
                                count_2_ext += 1
                
                
                    if ampl_1 != 0 and sum(q_1_ext) != 0:
                        reco_cart = barycenter_3D(pos_1, q_1_ext)
                        if reco_cart[2] > z_min + fid and reco_cart[2] < z_max - fid:
                            mean_phi = np.average(pos_1phi, weights=q_1_ext)
                            var_phi = np.average((pos_1phi-mean_phi)**2, weights=q_1_ext)
                            mean_z = np.average(pos_1z, weights=q_1_ext)
                            var_z = np.average((pos_1z-mean_z)**2, weights=q_1_ext)
                            var_phi1[threshold].append(var_phi)
                            var_z1[threshold].append(var_z)
                            touched_sipms1_ext[threshold].append(count_1_ext)
                            r1 = np.sqrt(aveX1*aveX1 + aveY1*aveY1)
                            true_r1[threshold].append(r1)
                        else:
                            var_phi1[threshold].append(1.e9)
                            var_z1[threshold].append(1.e9)
                            touched_sipms1_ext[threshold].append(1.e9)
                            true_r1[threshold].append(1.e9)
                    else:
                        var_phi1[threshold].append(1.e9)
                        var_z1[threshold].append(1.e9)
                        touched_sipms1_ext[threshold].append(1.e9)
                        true_r1[threshold].append(1.e9)
                    if ampl_2 != 0 and sum(q_2_ext) != 0:
                        reco_cart = barycenter_3D(pos_2, q_2_ext)
                        if reco_cart[2] > z_min + fid and reco_cart[2] < z_max - fid:
                            mean_phi = np.average(pos_2phi, weights=q_2_ext)
                            var_phi = np.average((pos_2phi-mean_phi)**2, weights=q_2_ext)
                            mean_z = np.average(pos_2z, weights=q_2_ext)
                            var_z = np.average((pos_2z-mean_z)**2, weights=q_2_ext)
                            var_phi2[threshold].append(var_phi)
                            var_z2[threshold].append(var_z)
                            touched_sipms2_ext[threshold].append(count_2_ext)
                            r2 = np.sqrt(aveX2*aveX2 + aveY2*aveY2)
                            true_r2[threshold].append(r2)
                        else:
                            var_phi2[threshold].append(1.e9)
                            var_z2[threshold].append(1.e9)
                            touched_sipms2_ext[threshold].append(1.e9)
                            true_r2[threshold].append(1.e9)
                    else:
                        var_phi2[threshold].append(1.e9)
                        var_z2[threshold].append(1.e9)
                        touched_sipms2_ext[threshold].append(1.e9)
                        true_r2[threshold].append(1.e9)
                        

a_true_r1_0 = np.array(true_r1[0])
a_true_r2_0 = np.array(true_r2[0])
a_var_phi1_0 = np.array(var_phi1[0])
a_var_phi2_0 = np.array(var_phi2[0])
a_var_z1_0 = np.array(var_z1[0])
a_var_z2_0 = np.array(var_z2[0])
a_touched_sipms1_ext_0 = np.array(touched_sipms1_ext[0])
a_touched_sipms2_ext_0 = np.array(touched_sipms2_ext[0])

a_true_r1_1 = np.array(true_r1[1])
a_true_r2_1 = np.array(true_r2[1])
a_var_phi1_1 = np.array(var_phi1[1])
a_var_phi2_1 = np.array(var_phi2[1])
a_var_z1_1 = np.array(var_z1[1])
a_var_z2_1 = np.array(var_z2[1])
a_touched_sipms1_ext_1 = np.array(touched_sipms1_ext[1])
a_touched_sipms2_ext_1 = np.array(touched_sipms2_ext[1])

a_true_r1_2 = np.array(true_r1[2])
a_true_r2_2 = np.array(true_r2[2])
a_var_phi1_2 = np.array(var_phi1[2])
a_var_phi2_2 = np.array(var_phi2[2])
a_var_z1_2 = np.array(var_z1[2])
a_var_z2_2 = np.array(var_z2[2])
a_touched_sipms1_ext_2 = np.array(touched_sipms1_ext[2])
a_touched_sipms2_ext_2 = np.array(touched_sipms2_ext[2])

a_true_r1_3 = np.array(true_r1[3])
a_true_r2_3 = np.array(true_r2[3])
a_var_phi1_3 = np.array(var_phi1[3])
a_var_phi2_3 = np.array(var_phi2[3])
a_var_z1_3 = np.array(var_z1[3])
a_var_z2_3 = np.array(var_z2[3])
a_touched_sipms1_ext_3 = np.array(touched_sipms1_ext[3])
a_touched_sipms2_ext_3 = np.array(touched_sipms2_ext[3])

a_true_r1_4 = np.array(true_r1[4])
a_true_r2_4 = np.array(true_r2[4])
a_var_phi1_4 = np.array(var_phi1[4])
a_var_phi2_4 = np.array(var_phi2[4])
a_var_z1_4 = np.array(var_z1[4])
a_var_z2_4 = np.array(var_z2[4])
a_touched_sipms1_ext_4 = np.array(touched_sipms1_ext[4])
a_touched_sipms2_ext_4 = np.array(touched_sipms2_ext[4])


np.savez(evt_file, a_true_r1_0=a_true_r1_0, a_true_r2_0=a_true_r2_0, a_true_r1_1=a_true_r1_1, a_true_r2_1=a_true_r2_1, a_true_r1_2=a_true_r1_2, a_true_r2_2=a_true_r2_2, a_true_r1_3=a_true_r1_3, a_true_r2_3=a_true_r2_3, a_true_r1_4=a_true_r1_4, a_true_r2_4=a_true_r2_4, a_var_phi1_0=a_var_phi1_0, a_var_phi2_0=a_var_phi2_0, a_var_z1_0=a_var_z1_0, a_var_z2_0=a_var_z2_0, a_touched_sipms1_ext_0=a_touched_sipms1_ext_0, a_touched_sipms2_ext_0=a_touched_sipms2_ext_0, a_var_phi1_1=a_var_phi1_1, a_var_phi2_1=a_var_phi2_1, a_var_z1_1=a_var_z1_1, a_var_z2_1=a_var_z2_1, a_touched_sipms1_ext_1=a_touched_sipms1_ext_1, a_touched_sipms2_ext_1=a_touched_sipms2_ext_1,a_var_phi1_2=a_var_phi1_2, a_var_phi2_2=a_var_phi2_2, a_var_z1_2=a_var_z1_2, a_var_z2_2=a_var_z2_2, a_touched_sipms1_ext_2=a_touched_sipms1_ext_2, a_touched_sipms2_ext_2=a_touched_sipms2_ext_2, a_var_phi1_3=a_var_phi1_3, a_var_phi2_3=a_var_phi2_3, a_var_z1_3=a_var_z1_3, a_var_z2_3=a_var_z2_3, a_touched_sipms1_ext_3=a_touched_sipms1_ext_3, a_touched_sipms2_ext_3=a_touched_sipms2_ext_3, a_var_phi1_4=a_var_phi1_4, a_var_phi2_4=a_var_phi2_4, a_var_z1_4=a_var_z1_4, a_var_z2_4=a_var_z2_4, a_touched_sipms1_ext_4=a_touched_sipms1_ext_4, a_touched_sipms2_ext_4=a_touched_sipms2_ext_4)

