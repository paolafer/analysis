import os
import sys
import math
import tables as tb
import numpy  as np


from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.io.mcinfo_io           import read_mcinfo

from   antea.io.mc_io                          import read_mcsns_response
from   antea.io.mc_io                          import read_SiPM_bin_width_from_conf
from   antea.io.mc_io                          import go_through_file
from   antea.utils.table_functions             import load_rpos

from invisible_cities.core.exceptions     import SipmEmptyList
from invisible_cities.core.exceptions     import SipmZeroCharge

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
        [x2, y2, z2]
        ...
        [xs, ys, zs])
       qs = vector (q1, q2...qs) --> (1xn)
        """

    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge
        
    return np.average(pos, weights=qs, axis=0)

def find_SiPMs_over_thresholds(this_event_wvf, threshold):

    sns_dict = list(this_event_wvf.values())[0]
    tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
    sns_ids = np.array(list(sns_dict.keys()))
    
    indices_over_thr = (tot_charges > threshold)
    sns_over_thr = sns_ids[indices_over_thr]
    charges_over_thr = tot_charges[indices_over_thr]

    return sns_over_thr, charges_over_thr 


start = int(sys.argv[1])
numb = int(sys.argv[2])
max_cut = int(sys.argv[3])

base_file = '/home/paolafer/data/MC/pet/full_ring_iradius15cm_depth3cm_pitch4mm_one_face.{0}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/full_ring_iradius15cm_depth3cm_pitch4mm_one_face_phot_reco_pos_{0}_{1}_{2}'.format(start, numb, max_cut)

rpos_file = '/home/paolafer/analysis/petalo/tables/r_table_iradius15cm_thr2pes_4mm_depth3cm_one_face.h5'
rpos_threshold = 2
Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = "f{}pes200bins".format(rpos_threshold))

events = []

true_x1 = []
true_x2 = []
true_y1 = []
true_y2 = []
true_z1 = []
true_z2 = []

reco_x1 = []
reco_x2 = []
reco_y1 = []
reco_y2 = []
reco_z1 = []
reco_z2 = []

photo_response1_ext = []
photo_response2_ext = []

touched_sipms1_ext  = []
touched_sipms2_ext  = []


evt_no_pos = 0
evt_no = 0

for ifile in range(start, start+numb):

    file_name = base_file.format(ifile)
    try:
        read_mcsns_response(file_name, (0,1))
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

            bin_width   = read_SiPM_bin_width_from_conf(h5in)
            this_event_wvf = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
            
            event_number = h5in.root.MC.extents[evt]['evt_number']
            part_dict = list(this_event_dict.values())[0]

            energy1 = energy2 = 0.
           # aveX1 = aveY1 = aveZ1 = 0.
           # aveX2 = aveY2 = aveZ2 = 0.
            ave_true1 = []
            ave_true2 = []

           # both = 0
           # interest = False

            for indx, part in part_dict.items():
                if part.name == 'e-' :
                    mother = part_dict[part.mother_indx]
                    if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                        if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6):
                            if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                                #interest = True
                                #both += 1

                                if mother.p[1] > 0.:
                                    hit_positions = [h.pos for h in part.hits]
                                    energies = [h.E for h in part.hits]
                                    energy1 = sum(energies)
                                 #   for h in part.hits:
                                 #       aveX1 += h.X * h.E
                                 #       aveY1 += h.Y * h.E
                                 #       aveZ1 += h.Z * h.E 
                                 #       energy1 += h.E
                                    if energy1 != 0.:
                                        ave_true1 = np.average(hit_positions, axis=0, weights=energies)
                                    else:
                                        ave_true1 = np.array([1.e9, 1.e9, 1.e9])
                                 #   aveX1 = aveX1 / energy1
                                  #  aveY1 = aveY1 / energy1
                                   # aveZ1 = aveZ1 / energy1
                                 
                                else:
                                    hit_positions = [h.pos for h in part.hits]
                                    energies = [h.E for h in part.hits]
                                    energy2 = sum(energies)
                                #for h in part.hits:
                                #        aveX2 += h.X * h.E
                               #         aveY2 += h.Y * h.E
                                #        aveZ2 += h.Z * h.E 
                                #        energy2 += h.E
                                    if energy2 != 0.:
                                        ave_true2 = np.average(hit_positions, axis=0, weights=energies)
                                    else:
                                        ave_true2 = np.array([1.e9, 1.e9, 1.e9])
            
            
            if energy1 or energy2:

                sns_over_thr, charges_over_thr  = find_SiPMs_over_thresholds(this_event_wvf, rpos_threshold)
  
                if len(charges_over_thr) == 0:
                    evt_no_pos += 1
                    continue

                #max_sns = sns_over_thr[np.argmax(charges_over_thr)]

                pos_1phi = []
                pos_2phi = []

                q_1_ext = []
                q_2_ext = []

                if energy1:
                    sns_closest1 = find_closest_sipm(ave_true1,
                                                     sensor_pos, sns_over_thr, charges_over_thr)
                if energy2:
                    sns_closest2 = find_closest_sipm(ave_true2,
                                                     sensor_pos, sns_over_thr, charges_over_thr)
                
                for sns_id, charge in zip(sns_over_thr, charges_over_thr):                       
                    pos     = sensor_pos[sns_id]
                    pos_cyl = sensor_pos_cyl[sns_id]
                    #pos_max = sensor_pos[max_sns]
                    if energy1:
                        pos_closest = sensor_pos[sns_closest1]
                        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                        if scalar_prod > 0.: 
                            q_1_ext.append(charge)
                            pos_1phi.append(pos_cyl[1])
                                                 
                    if energy2:
                        pos_closest = sensor_pos[sns_closest2]
                        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                        if scalar_prod > 0.:
                            q_2_ext.append(charge)
                            pos_2phi.append(pos_cyl[1])
                        
            
                var_phi1 = var_phi2 = None
                if sum(q_1_ext) != 0:
                    mean_phi = np.average(pos_1phi, weights=q_1_ext)
                    var_phi1 = np.average((pos_1phi-mean_phi)**2, weights=q_1_ext)
                if sum(q_2_ext) != 0:
                    mean_phi = np.average(pos_2phi, weights=q_2_ext)
                    var_phi2 = np.average((pos_2phi-mean_phi)**2, weights=q_2_ext)
                

                sns_over_thr, charges_over_thr  = find_SiPMs_over_thresholds(this_event_wvf, max_cut)

                if len(charges_over_thr) == 0:
                    evt_no += 1
                    continue
                #max_sns = sns_over_thr[np.argmax(charges_over_thr)]

                ### Find the closest SiPM to the true average point
                if energy1:
                    min_sns1 = find_closest_sipm(ave_true1,
                                                 sensor_pos, sns_over_thr, charges_over_thr)
                if energy2:
                    min_sns2 = find_closest_sipm(ave_true2,
                                                 sensor_pos, sns_over_thr, charges_over_thr)


                pos_1_ext = []
                pos_2_ext = []

                q_1_ext = []
                q_2_ext = []
                
                for sns_id, charge in zip(sns_over_thr, charges_over_thr):
                    pos     = sensor_pos[sns_id]
                    pos_cyl = sensor_pos_cyl[sns_id]
                    #pos_max = sensor_pos[max_sns]

                    if energy1:
                        pos_closest = sensor_pos[min_sns1]
                        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                        if scalar_prod > 0.: 
                            pos_1_ext.append(pos)
                            q_1_ext.append(charge)
                                         
                    if energy2:
                        pos_closest = sensor_pos[min_sns2]
                        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                        if scalar_prod > 0.:
                            pos_2_ext.append(pos)
                            q_2_ext.append(charge)                       
            
                
                if sum(q_1_ext) != 0 and var_phi1:  
                    reco_r    = Rpos(np.sqrt(var_phi1)).value 
                    reco_cart = barycenter_3D(pos_1_ext, q_1_ext)
                    reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])
                                     
                    reco_x1.append(reco_r * np.cos(reco_phi))
                    reco_y1.append(reco_r * np.sin(reco_phi))
                    reco_z1.append(reco_cart[2])
                    true_x1.append(ave_true1[0])
                    true_y1.append(ave_true1[1])
                    true_z1.append(ave_true1[2])
                    photo_response1_ext.append(sum(q_1_ext))
                    touched_sipms1_ext.append(len(q_1_ext))
                    events.append(event_number)
                else:
                    reco_x1.append(1.e9)
                    reco_y1.append(1.e9)
                    reco_z1.append(1.e9)
                    true_x1.append(1.e9)
                    true_y1.append(1.e9)
                    true_z1.append(1.e9)
                    photo_response1_ext.append(1.e9)
                    touched_sipms1_ext.append(1.e9)
                    events.append(event_number)
                    
                if sum(q_2_ext) and var_phi2: 
                    reco_r    = Rpos(np.sqrt(var_phi2)).value 
                    reco_cart = barycenter_3D(pos_2_ext, q_2_ext)
                    reco_phi  = np.arctan2(reco_cart[1], reco_cart[0])
        
                    reco_x2.append(reco_r * np.cos(reco_phi))
                    reco_y2.append(reco_r * np.sin(reco_phi))
                    reco_z2.append(reco_cart[2])
                    true_x2.append(ave_true2[0])
                    true_y2.append(ave_true2[1])
                    true_z2.append(ave_true2[2])
                    photo_response2_ext.append(sum(q_2_ext))
                    touched_sipms2_ext.append(len(q_2_ext))
                else:
                    reco_x2.append(1.e9)
                    reco_y2.append(1.e9)
                    reco_z2.append(1.e9)
                    true_x2.append(1.e9)
                    true_y2.append(1.e9)
                    true_z2.append(1.e9)
                    photo_response2_ext.append(1.e9)
                    touched_sipms2_ext.append(1.e9)  


print('Events neglected because no good radial reconstruction = {}'.format(evt_no_pos))
print('Events neglected because no charges above threshold = {}'.format(evt_no))

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

a_touched_sipms1_ext = np.array(touched_sipms1_ext)
a_touched_sipms2_ext = np.array(touched_sipms2_ext)

a_photo_response1_ext = np.array(photo_response1_ext)
a_photo_response2_ext = np.array(photo_response2_ext)

a_events = np.array(events)


np.savez(evt_file, a_true_x1=a_true_x1, a_true_y1=a_true_y1, a_true_z1=a_true_z1, a_true_x2=a_true_x2, a_true_y2=a_true_y2, a_true_z2=a_true_z2, a_reco_x1=a_reco_x1, a_reco_y1=a_reco_y1, a_reco_z1=a_reco_z1, a_reco_x2=a_reco_x2, a_reco_y2=a_reco_y2, a_reco_z2=a_reco_z2,  a_touched_sipms1_ext_=a_touched_sipms1_ext, a_touched_sipms2_ext=a_touched_sipms2_ext, a_photo_response1_ext_=a_photo_response1_ext, a_photo_response2_ext=a_photo_response2_ext, a_events=a_events)

