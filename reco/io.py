import os
import pandas as pd


def combine_datafiles(basedir, basefile, startfile, nfiles, outfile):
    """Read data files containing reconstructed coordinates and combine them into a single file.
    """

    dfs = []

    for ii in range(nfiles):

        fnum = startfile + ii

        #fname = "{}/phantom_NEMAlike_coincidences_{}.npz".format(basedir,fnum)
        fname = f"{basedir}/{basefile}.{fnum}.h5"
        print(fname)
        if os.path.isfile(fname):

            print("Adding file {}...".format(fname))

            #try:
            if True:
                df = pd.read_hdf(fname)
                dfs.append(df)
            #except:
            #    print('File {} not found, skipping...'.format(fname))
            #    continue

    df_reco_info = pd.concat(dfs)
    df_reco_info = df_reco_info[df_reco_info.true_r1 != 0]

    store = pd.HDFStore(outfile, "w", complib=str("zlib"), complevel=4)
    store.put('reco_info',  df_reco_info, format='table', data_columns=True)
    store.close()


def read_datafile(infile, n_coincidences):

    df = pd.read_hdf(infile)
    df = df.head(n_coincidences)

    true_r1   = df.true_r1.values
    true_phi1 = df.true_phi1.values
    true_z1   = df.true_z1.values
    true_t1   = df.true_t1.values

    reco_r1   = df.reco_r1.values
    reco_phi1 = df.reco_phi1.values
    reco_z1   = df.reco_z1.values
    reco_t1   = df.reco_t1.values

    true_r2   = df.true_r2.values
    true_phi2 = df.true_phi2.values
    true_z2   = df.true_z2.values
    true_t2   = df.true_t2.values

    reco_r2   = df.reco_r2.values
    reco_phi2 = df.reco_phi2.values
    reco_z2   = df.reco_z2.values
    reco_t2   = df.reco_t2.values

    event_ids = df.event_id.values

    return (event_ids, true_r1, true_phi1, true_z1, true_t1,
                       true_r2, true_phi2, true_z2, true_t2,
                       reco_r1, reco_phi1, reco_z1, reco_t1,
                       reco_r2, reco_phi2, reco_z2, reco_t2)
