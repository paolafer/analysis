import sys, os
import bisect

import time

import numpy             as np
import pandas            as pd
import tables            as tb

from functools import partial
from itertools import product

from   scipy.interpolate import griddata
import invisible_cities.core.system_of_units_c as system_of_units

from   invisible_cities.io.mcinfo_io import read_mchit_info

from invisible_cities.reco.tbl_functions import get_mc_info

import antea.database.load_db as db
from   antea.io.mc_io import read_mcsns_response
from   antea.io.mc_io import read_SiPM_bin_width_from_conf
from   antea.io.mc_io import go_through_file
from   antea.io.mc_io import mc_sns_response_writer
from   antea.io.mc_io import load_mchits, load_mcparticles

units = system_of_units.SystemOfUnits()

class PETALOparams:
    def __init__(self, **kwargs):

        self.w_s = 17.0 * units.eV
        self.update(**kwargs)

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

PETALO = PETALOparams()

class InterpolationProbList:

    def __init__(self,
                 xs, fs,
                 norm_strategy = None,
                 norm_opts     = {},
                 interp_method   = "nearest",
                 default_f       = 0):

        self._xs = [np.array( x, dtype=float) for x in xs]
        self._fs =  np.array(fs, dtype=float)
        self.interp_method   = interp_method
        self.default_f       = default_f

        self._init_interpolator(interp_method, default_f)

    @profile
    def __call__(self, *xs):
        """
        Compute the probability.
        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        # In order for this to work well both for arrays and scalars
        arrays = len(np.shape(xs)) > 1
        if arrays:
            xs = np.stack(xs, axis=1)

        value  = self._get_value(xs)

        return value if arrays else value[0]

    @profile
    def _init_interpolator(self, method, default_f):
        coordinates = np.array(list(product(*self._xs)))
        self._get_value = partial(griddata,
                                  coordinates,
                                  self._fs,
                                  method     = method,
                                  fill_value = default_f)

### Functions
@profile
def from_cartesian_to_cyl_v(positions):
    cyl_positions = np.array([np.sqrt(positions[:,0]**2+positions[:,1]**2), np.arctan2(positions[:,1], positions[:,0]), positions[:,2]]).transpose()
    #cyl_positions = np.array([(np.sqrt(pos[0]**2+pos[1]**2), np.arctan2(pos[1], pos[0]), pos[2]) for pos in positions])
    return cyl_positions

@profile
def find_closest_SiPM_v(cyl_positions):

    nhits = len(cyl_positions)

    # Select the closest SiPMs in phi.
    hit_phis = np.array(cyl_positions[:,1]).reshape(nhits,1)
    dist_phis = np.abs(hit_phis - sipm_phis.values)
    min_phis = np.min(dist_phis, axis=1)
    sel_phis = np.isclose(dist_phis,min_phis.reshape(nhits,1))

    hit_zs = np.array(cyl_positions[:,2]).reshape(nhits,1)
    dist_zs = np.abs(hit_zs - sipm_zs.values)
    min_zs = np.min(dist_zs, axis=1)
    sel_zs = np.isclose(dist_zs,min_zs.reshape(nhits,1))

    sel_phi_z = [DataSiPM[sphi & sz] for sphi,sz in zip(sel_phis,sel_zs)]

    return sel_phi_z

@profile
def distance_from_ref_v(points_cyl,closest_sipms):

    closest_positions_cyl = np.array([(cs.R.values[0], cs.Phi.values[0], cs.Z.values[0]) for cs in closest_sipms])
    deltas = points_cyl - closest_positions_cyl

    distances = [(p[0], d[1], d[2]) for p, d in zip(points_cyl, deltas)]

    return distances



### Files - temporary
#points_file  = '/home/paolafer/analysis/petalo/lut/scint_points_cyl_all.txt'
points_file  = '/Users/paola/PETALO/sim/lut/scint_points_cyl_all.txt'
#df_file_name = '/home/paolafer/analysis/petalo/lut/lut_cut_database_{0}_{1}_1000.hdf5'
df_file_name = '/Users/paola/PETALO/sim/lut/lut_cut_database_{0}_{1}_1000.hdf5'
#in_base_file_name  = '/data_extra2/paolafer/SimMC/pet/full_ring_fix-rot_iradius165mm_depth3cm_pitch7mm_z1.{0:03d}.pet.h5'
#in_base_file_name  = '/data_extra2/paolafer/SimMC/pet/lut/test.{0:03d}.pet.h5'
in_base_file_name  = '/Users/paola/PETALO/sim/lut/full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.{0:03d}.pet.h5'
#out_base_file_name = '/data_extra2/paolafer/SimMC/pet/full_ring_fix-rot_iradius165mm_depth3cm_pitch7mm_z1_lut_nearest_josh_final.{0:03d}.pet.h5'
#out_base_file_name = '/data_extra2/paolafer/SimMC/pet/lut/test_lut_nearest.{0:03d}.pet.h5'
#out_base_file_name = '/home/jrenner/petalo/lut/test_lut_nearest.{0:03d}.pet.h5'
out_base_file_name = '/Users/paola/PETALO/sim/lut/full_ring_iradius165mm_depth3cm_pitch7mm_new_h5_lut_sns_only.{0:03d}.pet.h5'

start = int(sys.argv[1])
numb = int(sys.argv[2])

#start_time = time.time()

### Read sensor positions from database
print("Building sensor position table...")
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

pos_1 = np.array([DataSiPM_idx.loc[1000].X, DataSiPM_idx.loc[1000].Y, DataSiPM_idx.loc[1000].Z]).reshape(1,3)
cyl_1 = from_cartesian_to_cyl_v(pos_1)
pos_2 = np.array([DataSiPM_idx.loc[1001].X, DataSiPM_idx.loc[1001].Y, DataSiPM_idx.loc[1001].Z]).reshape(1,3)
cyl_2 = from_cartesian_to_cyl_v(pos_2)
#print(cyl_2.shape)

sipm_z_pitch   = np.abs(DataSiPM_idx.loc[1000].Z - DataSiPM_idx.loc[1175].Z)
sipm_phi_pitch = np.abs(cyl_1[0][1] - cyl_2[0][1])
max_phi_number = DataSiPM.PhiNumber.max()
max_z_number   = DataSiPM.ZNumber.max()
sipm_rs   = np.sqrt(DataSiPM.X*DataSiPM.X + DataSiPM.Y*DataSiPM.Y)
sipm_phis = np.arctan2(DataSiPM.Y, DataSiPM.X)
sipm_zs   = DataSiPM.Z

DataSiPM['R']   = sipm_rs
DataSiPM['Phi'] = sipm_phis

sipms_phi_z = np.zeros([max_phi_number+1, max_z_number+1], dtype=int)
c = 0
for sns_id, phi, z in zip(DataSiPM.SensorID, DataSiPM.PhiNumber.values, DataSiPM.ZNumber.values):
    sipms_phi_z[phi][z] = sns_id
    c += 1

ProbList     = db.ProbabilityList('petalo', 0)
ProbList     = ProbList.drop(['PointID'], axis=1)

### Read scintillation point position
print("Reading scintillation point positions and building dataframe...")
scint_points = np.loadtxt(points_file, delimiter=' ')

dist_df = pd.DataFrame(columns=['pointID', 'r', 'dist_phi', 'dist_z'])
scint_point_df = pd.DataFrame(columns=['pointID', 'r', 'phi', 'z'])
c=0
for p in scint_points:
    dist_df.loc[c] = [c, p[0], p[1] - cyl_1[0][1], p[2] - cyl_1[0][2]]
    scint_point_df.loc[c] = [c, p[0], p[1], p[2]]
    c += 1

convert_dict = {'pointID': int, 'r': float, 'dist_phi': float, 'dist_z': float}
dist_df      = dist_df.astype(convert_dict)

convert_dict   = {'pointID': int, 'r': float, 'phi': float, 'z': float}
scint_point_df = scint_point_df.astype(convert_dict)

dist_phi_u, dist_z_u = np.unique(dist_df.dist_phi.values), np.unique(dist_df.dist_z.values)
r_u, phi_u, z_u      = np.unique(scint_point_df.r.values), np.unique(scint_point_df.phi.values), np.unique(scint_point_df.z.values)
#print("Distance dataframe:")
#print(dist_df)
#print("Unique distances (phi,z)")
#print(dist_phi_u)
#print(dist_z_u)
#print("Scintillation dataframe:")
#print(scint_point_df)
#print("Unique (r,phi,z)")
#print(r_u)
#print(phi_u)
#print(z_u)

### Join dataframes
bunch = 40
# first dataframe
f = 0
try:
    df = pd.read_hdf(df_file_name.format(f, bunch), 'lut')
except:
    print('File {0} not good'.format(df_file_name.format(f, bunch)))

full_db = df

for i in range(1, 157):
    try:
        df = pd.read_hdf(df_file_name.format(i*bunch, bunch), 'lut')
    except:
        print('File {0} not good'.format(df_file_name.format(i*bunch, bunch)))
        continue

    full_db = pd.concat([full_db, df], ignore_index=True)

f     = 6280
bunch = 20
try:
    df = pd.read_hdf(df_file_name.format(f, bunch), 'lut')
except:
    print('File {0} not good'.format(df_file_name.format(f, bunch)))

full_db = pd.concat([full_db, df], ignore_index=True)

full_db_sorted = full_db.sort_values(by=['pointID', 'step_phi', 'step_z'])
df_sel         = full_db_sorted[full_db_sorted.prob > 0.000015]

min_step_phi = df_sel.step_phi.values.min()
max_step_phi = df_sel.step_phi.values.max()
min_step_z   = df_sel.step_z.values.min()
max_step_z   = df_sel.step_z.values.max()
print("Found min_step_phi, max_step_phi, min_step_z, max_step_z")
print(min_step_phi, max_step_phi, min_step_z, max_step_z)

len_dict = (max_step_phi - min_step_phi + 1)*(max_step_z - min_step_z + 1)
step_array = np.zeros([len_dict, 2], dtype=int)
c = 0
for phi in range(min_step_phi, max_step_phi+1):
    for z in range(min_step_z, max_step_z+1):
        step_array[c][0] = phi
        step_array[c][1] = z
        c += 1
#step_dict     = {}
#step_dict_inv = {}
#c = 0
#for i in range(min_step_phi, max_step_phi+1):
#    for j in range(min_step_z, max_step_z+1):
#        step_dict[c] = (i, j)
#        step_dict_inv[(i, j)] = c
#        c += 1
#print("Created step array:")
#print(step_array)

point_p_a    = ProbList.values
interpolator = InterpolationProbList((r_u, dist_phi_u, dist_z_u), point_p_a, interp_method='nearest')
#print("Prob list values:")
#print(point_p_a)

@profile
def get_sensor_responses():

    file_sensor_response = {}

    for i in range(start, start+numb):
        in_file_name  = in_base_file_name.format(i)
        try:
            with tb.open_file(in_file_name, mode='r') as h5f:
                config = h5f.root.MC.configuration
        except:
            print('File {} not found'.format(in_file_name))

        out_file_name = out_base_file_name.format(i)
        print(in_file_name)

        my_hits_df = load_mchits(in_file_name)
        events = my_hits_df.evt_id.unique()

        end_time1 = time.time()

        for event_number in events:
            hits = my_hits_df[my_hits_df.evt_id == event_number]

            sensor_response = {}

            n_ph = hits.hit_energy/PETALO.w_s
            rnd_photons = np.random.poisson(n_ph)
            hpos = np.array([hits.hit_pos_x, hits.hit_pos_y, hits.hit_pos_z]).transpose()

            cyl_positions = from_cartesian_to_cyl_v(hpos)
            closest_SiPMs = find_closest_SiPM_v(cyl_positions)
            p_to_interpolate = distance_from_ref_v(cyl_positions,closest_SiPMs)
            prob_lists = interpolator(p_to_interpolate)

            #rnd_numbers = [np.random.rand(n_photons) for n_photons in rnd_photons]

            for prob_list, closest_sipm, rnd in zip(prob_lists, closest_SiPMs, rnd_photons):
                grad_prob = prob_list[0][1:] - prob_list[0][:-1]
                grad_prob = np.insert(grad_prob, 0, 0)
                pos_in_list = np.random.choice(len(grad_prob), size=rnd, p=grad_prob)

                #pos_in_list = np.array([bisect.bisect_left(prob_list[0], random) for random in n_phot])
                pos_in_list[pos_in_list==0] = 1
                sel = pos_in_list <= len(step_array)
                pos = pos_in_list[sel]
                if len(pos) == 0: continue

                steps     = step_array[pos-1]
                step_phis = steps[:, 0]
                step_zs   = steps[:, 1]

                phi_closest = closest_sipm.PhiNumber.values[0]
                z_closest   = closest_sipm.ZNumber.values[0]

                phi_sensors = phi_closest + step_phis
                phi_sensors[phi_sensors>max_phi_number] = phi_sensors[phi_sensors>max_phi_number] - max_phi_number - 1
                phi_sensors[phi_sensors<0] = phi_sensors[phi_sensors<0] + max_phi_number + 1

                z_sensors = z_closest + step_zs
                sel = (z_sensors<=max_z_number) & (z_sensors>=0)
                z_sensors   = z_sensors[sel]
                phi_sensors = phi_sensors[sel]
                sipm_ids = sipms_phi_z[phi_sensors, z_sensors]

                for sipm_id in sipm_ids:
                    try:
                        sensor_response[sipm_id] += 1
                    except:
                        sensor_response[sipm_id] = 1


            file_sensor_response[event_number] = sensor_response

        end_time2 = time.time()
        print('Time in the event: {}'.format(end_time2-end_time1))


        writer = mc_sns_response_writer(in_file_name)

        for evt in file_sensor_response.keys():
            writer(file_sensor_response, evt)
        writer.close_file()
        
        end_time3 = time.time()
        print('Time of writing to file: {}'.format(end_time3-end_time2))


# Run the function
get_sensor_responses()
