
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

fig, ax = pl.subplots()

bin_w = 1

for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        with open(dpath+'/lts.p', 'rb') as pfile:
            lts_df=np.array(pickle.load(pfile))

        # discard synapses present at beginning
        lts_df = lts_df[lts_df[:,1]>0]

        # only take synapses grown in first half of simulation
        t_split = nsp['Nsteps']/2
        lts_df = lts_df[lts_df[:,3]<t_split]
        
        lts = lts_df[:,2] - lts_df[:,3]
        
        assert np.min(lts) > 0
        
        lts[lts>t_split]=t_split

        bins = np.arange(1,t_split+bin_w,bin_w)

        counts, edges = np.histogram(lts,
                                     bins=bins,
                                     density=False)

        srv = 1. - np.cumsum(counts)/float(np.sum(counts))

        label = str(nsp['bn_sig'])

        centers = (edges[:-1] + edges[1:])/2.
        ax.plot(centers, srv, '.', label=label)



    except FileNotFoundError:
        print(dpath[-4:], "reports: Error loading namespace")



    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('lifetime [steps]')
    ax.set_ylabel('relative frequency')



    directory = 'figures/prb_srv_single/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    pl.legend()

    fname = dpath[-4:]

    fig.savefig(directory+'/'+fname+'.png', dpi=150,
                bbox_inches='tight')


