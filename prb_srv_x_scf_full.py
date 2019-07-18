
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib import rc

rc('text', usetex=True)
pl.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',   
    r'\usepackage{sansmath}',  
    r'\sansmath'               
    r'\usepackage{siunitx}',   
    r'\sisetup{detect-all}',   
]  

import argparse, sys, os, itertools, pickle
import numpy as np

   


data_dirs = sorted(['data/'+pth for pth in next(os.walk("data/"))[1]])


# find all bin bn_mus, bn_sigs
bn_mus,bn_sigs = [],[]

for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        bn_mus.append(nsp['bn_mu'])
        bn_sigs.append(nsp['bn_sig'])

    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")

bn_mus = list(set(bn_mus))
bn_sigs = list(set(bn_sigs))

idmap = {0.0     : 0,
         0.00005 : 1,
         0.0001  : 2}




for bin_w in [1.]:#,10.,100.,1000.,10000.]:

    for bn_sig in bn_sigs:

        fig, axes = pl.subplots(3,2)
        fig.set_size_inches(7.35,8)
        
        for dpath in data_dirs:

            try:
                with open(dpath+'/namespace.p', 'rb') as pfile:
                    nsp=pickle.load(pfile)

                if nsp['bn_sig'] == bn_sig:

                    with open(dpath+'/lts.p', 'rb') as pfile:
                        lts_df=np.array(pickle.load(pfile))

                    # print(lts_df[:10])
                    # print('\n', lts_df[:-5], '\n')

                    # ind = np.lexsort((lts_df[:,0],lts_df[:,1]))
                    # lts = lts_df[ind]

                    lts_frame = lts_df[lts_df[:,1]>0]
                    lts = lts_frame[:,2]
                    #assert len(lts)==nsp['Nprocess']

                    bins = np.arange(1,np.max(lts),bin_w)

                    counts, edges = np.histogram(lts,
                                                 bins=bins,
                                                 density=False)

                    srv = 1. - np.cumsum(counts)/float(np.sum(counts))

                    centers = (edges[:-1] + edges[1:])/2.

                    ax = axes[(idmap[nsp['bn_mu']],0)]
                    ax.plot(centers, srv, '.', label='scf = %d'
                            %(int(nsp['scf'])))

                    with open(dpath+'/scfs.p', 'rb') as pfile:
                        scfs=np.array(pickle.load(pfile))

                    ax = axes[(idmap[nsp['bn_mu']],1)]
                    ax.plot(scfs)


            except FileNotFoundError:
                print(dpath[-4:], "reports: Error loading namespace")


        for ax in list(axes.flatten()):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            ax.set_xlabel('lifetime [steps]')
            ax.set_ylabel('relative frequency')

        for ax in list(axes.flatten())[::2]:
            ax.legend(frameon=False, prop={'size': 8})
            ax.set_xscale('log')
            ax.set_yscale('log')


        directory = 'figures/prb_srv/'

        if not os.path.exists(directory):
            os.makedirs(directory)




        fname = "prb-srv_Nsteps_bn-sig%.5f_binw%d_clt0_3x2" %(bn_sig, int(bin_w))
        # fname = "turnover_x_T_%ds" %(int(bin_w/second))
        # if not starters:
        #     fname+='_nostart'

        fig.tight_layout()

        fig.savefig(directory+'/'+fname+'.png', dpi=150,
                    bbox_inches='tight')


