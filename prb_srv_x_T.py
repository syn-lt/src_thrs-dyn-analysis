
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
from scipy import optimize

def powerlaw_func_s(t, gamma, s):
    return (t/s+1)**(-1*gamma)


data_dirs = sorted(['data/'+pth for pth in next(os.walk("data/"))[1]])

fig, ax = pl.subplots()
fig.set_size_inches(5.2,3)

fit=False
bin_w = 1

for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        # if nsp['bn_sig']==0.1:
        if True:

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

            # label = str(nsp['up_cap'])
            # label = r'$\mu=' + '%.2f' %(nsp['mu']) + '$, ' +\
            #         r'$\theta=' + '%.4f' %(nsp['theta']) + '$, ' +\
            #         r'$\sigma=' + '%.2f' %(nsp['sigma']) + '$'
            label = ''

            centers = (edges[:-1] + edges[1:])/2.


          
            # ax.plot(centers, srv, 'o', label=label,
            #         markeredgewidth=1,
            #         markerfacecolor='None')


            if fit:
                sep = 0
                prm, prm_cov = optimize.curve_fit(powerlaw_func_s,
                                                  centers[centers>sep],
                                                  srv[centers>sep],
                                                  p0=[1.5,100])

                label = label + ', $\gamma = %.4f$' %(prm[0]) 

            ax.plot(centers, srv, label=label)

            if fit:
                ax.plot(centers[centers>sep],
                        powerlaw_func_s(centers[centers>sep],*prm),
                        color='grey', linestyle='--')


    except FileNotFoundError:
        print(dpath[-4:], "reports: Error loading namespace")


    xs = np.logspace(0, np.log10(nsp['Nsteps']), num=1000)

    ax.plot(xs, (xs+1)**(-0.5), color='grey', linestyle='dashed')


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # ax.set_suptitle('c= 

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('simulation steps')
    ax.set_ylabel('survival probability')

    # ax.set_ylim(10**(-6),1)



directory = 'figures/prb_srv_single/'

if not os.path.exists(directory):
    os.makedirs(directory)

ax.legend(frameon=False, loc='lower left',
              prop={'size': 9})

fname = dpath[-4:]

fig.savefig(directory+'/'+fname+'.png', dpi=300,
            bbox_inches='tight')


