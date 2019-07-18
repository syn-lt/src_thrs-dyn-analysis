
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


# def add_turnover(ax, bin_w, bpath, nsp, fit, starters):

#     label = str(int(nsp['T2']/second)) + ' s'

#     lifetime_distribution_loglog_linear_bins(ax, bpath, nsp,
#                                              bin_w=bin_w,
#                                              discard_t=0.,
#                                              with_starters=starters,
#                                              density=True,
#                                              label=label)
    

    
if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    data_dirs = sorted(['data/'+pth for pth in next(os.walk("data/"))[1]])

    # find all bin bn_mus
    bn_mus = []
    
    for dpath in data_dirs:
            
        try:
            with open(dpath+'/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)
            
            bn_mus.append(nsp['bn_mu'])

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")

    bn_mus = list(set(bn_mus))

    print(bn_mus)



    for bin_w in [1.,10.,100.,1000.,10000.]:
    
        for bn_mu in bn_mus:

            fig, ax = pl.subplots()

            for dpath in data_dirs:

                try:
                    with open(dpath+'/namespace.p', 'rb') as pfile:
                        nsp=pickle.load(pfile)

                    if nsp['bn_mu'] == bn_mu:

                        with open(dpath+'/lts.p', 'rb') as pfile:
                            lts_df=np.array(pickle.load(pfile))

                        lts = lts_df[:,2]

                        print(np.max(lts))

                        counts, edges = np.histogram(lts,
                                                     bins=np.arange(1,np.max(lts),bin_w),
                                                     # bins=10**np.linspace(0,5,150),
                                                     density=True)

                        centers = (edges[:-1] + edges[1:])/2.
                        ax.plot(centers, counts, '.', label=str(nsp['Nsteps']))



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



            directory = 'figures/lts_x/binw%d' %(int(bin_w))

            if not os.path.exists(directory):
                os.makedirs(directory)

            pl.legend()


            fname = "lt_x_Nsteps_bn-mu%.4f_binw%d" %(bn_mu, int(bin_w))
            # fname = "turnover_x_T_%ds" %(int(bin_w/second))
            # if not starters:
            #     fname+='_nostart'

            fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')


