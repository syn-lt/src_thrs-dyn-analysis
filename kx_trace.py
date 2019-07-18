
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


# # find all bin bn_mus, bn_sigs
# bn_mus,bn_sigs = [],[]

# for dpath in data_dirs:

#     try:
#         with open(dpath+'/namespace.p', 'rb') as pfile:
#             nsp=pickle.load(pfile)

#         bn_mus.append(nsp['bn_mu'])
#         bn_sigs.append(nsp['bn_sig'])

#     except FileNotFoundError:
#         print(bpath[-4:], "reports: No namespace data. Skipping.")

# bn_mus = list(set(bn_mus))
# bn_sigs = list(set(bn_sigs))

# print(bn_mus)
# print(bn_sigs)

for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)
        with open(dpath+'/KXt.p', 'rb') as pfile:
            kxt_df=np.array(pickle.load(pfile))

        print(kxt_df[:,0])


        fig, ax = pl.subplots()

        for i in range(len(kxt_df.T)):
            ax.plot(kxt_df[:,i], color='grey')



        directory = 'figures/kxt/'

        if not os.path.exists(directory):
                os.makedirs(directory)


        fname = "kxt-srv_Nsteps_bn-sig%.5f" %(nsp['bn_sig'])
        # fname = "turnover_x_T_%ds" %(int(bin_w/second))
        # if not starters:
        #     fname+='_nostart'

        fig.savefig(directory+'/'+fname+'.png', dpi=150,
                    bbox_inches='tight')

    except FileNotFoundError:
       print(dpath[-4:], "reports: Error loading namespace")

        

# for bin_w in [1.]:#,10.,100.,1000.,10000.]:

#     for bn_sig in bn_sigs:

#         fig, ax = pl.subplots()

#         for dpath in data_dirs:

#             try:
#                 with open(dpath+'/namespace.p', 'rb') as pfile:
#                     nsp=pickle.load(pfile)

#                 if nsp['bn_sig'] == bn_sig:

#                     with open(dpath+'/lts.p', 'rb') as pfile:
#                         lts_df=np.array(pickle.load(pfile))

#                     print(lts_df[:10])
#                     print('\n', lts_df[:-5], '\n')

#                     # ind = np.lexsort((lts_df[:,0],lts_df[:,1]))
#                     # lts = lts_df[ind]

#                     lts_frame = lts_df[lts_df[:,1]==0]
#                     lts = lts_frame[:,2]
#                     #assert len(lts)==nsp['Nprocess']

#                     bins = np.arange(1,np.max(lts),bin_w)

#                     counts, edges = np.histogram(lts,
#                                                  bins=bins,
#                                                  density=False)

#                     srv = 1. - np.cumsum(counts)/float(np.sum(counts))

#                     centers = (edges[:-1] + edges[1:])/2.
#                     ax.plot(centers, srv, '.', label=str(nsp['bn_sig']))



#             except FileNotFoundError:
#                 print(dpath[-4:], "reports: Error loading namespace")



#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')

#         ax.set_xscale('log')
#         ax.set_yscale('log')

#         ax.set_xlabel('lifetime [steps]')
#         ax.set_ylabel('relative frequency')



#         directory = 'figures/prb_srv/'

#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         pl.legend()


#         fname = "prb-srv_Nsteps_bn-sig%.5f_binw%d" %(bn_sig, int(bin_w))
#         # fname = "turnover_x_T_%ds" %(int(bin_w/second))
#         # if not starters:
#         #     fname+='_nostart'

#         fig.savefig(directory+'/'+fname+'.png', dpi=150,
#                     bbox_inches='tight')


