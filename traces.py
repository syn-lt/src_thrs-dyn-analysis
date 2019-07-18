
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
fig.set_size_inches(5.2,3)


for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        # if nsp['bn_sig']==0.1:
        if True:

            with open(dpath+'/record.p', 'rb') as pfile:
                df=np.array(pickle.load(pfile))


            # for i in range(1):
            for i in range(np.shape(df)[1]):
                ax.plot(df[:,i], color='grey')

                
    except FileNotFoundError:
        print(dpath[-4:], "reports: Error loading namespace")



    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # ax.set_suptitle('c=

    # ax.set_ylim(bottom=0., top=10.)

    # ax.set_yscale('log')

    ax.set_xlabel('simulation steps')
    ax.set_ylabel('survival probability')





directory = 'figures/traces/'

if not os.path.exists(directory):
    os.makedirs(directory)

ax.legend(frameon=False, loc='lower left',
              prop={'size': 9})

fname = dpath[-4:]

fig.savefig(directory+'/'+fname+'.png', dpi=300,
            bbox_inches='tight')


