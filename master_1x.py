
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
from decimal import Decimal
import numpy as np
from scipy import optimize
from scipy.stats import norm, lognorm

def powerlaw_func_s(t, gamma, s):
    return (t/s+1)**(-1*gamma)

data_dirs = sorted(['data/'+pth for pth in next(os.walk("data/"))[1]])

fig, axes = pl.subplots(2,2)
fig.set_size_inches(8.2,5)

fit=False
bin_w = 1



def survival_probability(ax, nsp, lts_df):

    # # discard synapses present at beginning. why???
    # lts_df = lts_df[lts_df[:,1]>0]

    # only consider synapses grown in first half of simulation
    t_split = nsp['Nsteps']/2
    lts_df = lts_df[lts_df[:,3]<t_split]

    # if a synapse didn't die until end of simulation, the lifetime
    # until then is added. thus there is always a 'death time point',
    # even if it is at the end of the simulation
    lts = lts_df[:,2] - lts_df[:,3]

    assert (np.min(lts) > 0)

    # lifetimes can be longer than t_split, but this is not considered
    # here since some synapses don't have data for this time period
    lts[lts>t_split]=t_split

    bins = np.arange(1,t_split+bin_w,bin_w)

    counts, edges = np.histogram(lts,
                                 bins=bins,
                                 density=False)

    # for every bin, calculate the fraction of synapses
    # that are still alive at this point
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



def equilibrium_distribution(ax, nsp, lts_df, nbins=50):

    weights = lts_df[lts_df[:,4]==-1][:,5]

    counts, edges = np.histogram(weights,bins=nbins,density=True)
    centers = (edges[:-1] + edges[1:])/2.
    
    ax.plot(centers, counts)
    
    # floc, fscale = norm.fit(log_weights)
    # f_rv = norm(loc=floc, scale=fscale)
    # xs = np.linspace(start=np.min(log_weights),
    #                  stop=np.max(log_weights),
    #                  num = 1000)

    # ax.plot(xs, f_rv.pdf(xs), linestyle='dashed', color='grey')

        

def equilibrium_distribution_log(ax, nsp, lts_df, notation, nbins=50):

    weights = lts_df[lts_df[:,4]==-1][:,5]
    log_weights = np.log10(weights[weights>0])

    counts, edges = np.histogram(log_weights,bins=nbins,density=True)
    centers = (edges[:-1] + edges[1:])/2.

    label = '$'+ notation[nsp['var']] + '=' +  str(nsp[nsp['var']]) + '$'
    ax.plot(centers, counts, label=label)

    if fit:
        floc, fscale = norm.fit(log_weights)
        f_rv = norm(loc=floc, scale=fscale)
        xs = np.linspace(start=np.min(log_weights),
                         stop=np.max(log_weights),
                         num = 1000)

        ax.plot(xs, f_rv.pdf(xs), linestyle='dashed', color='grey')



def lifetime_distribution(ax, nsp, lts_df):
    
    # only consider synapses that died within simulation time
    lts_df = lts_df[lts_df[:,4]==1]

    # lifetime is death time point - start time point
    lts = lts_df[:,2] - lts_df[:,3]

    assert (np.min(lts) > 0)

    bins = np.arange(1, nsp['Nsteps']+bin_w, bin_w)

    counts, edges = np.histogram(lts,
                                 bins=bins,
                                 density=True)
    
    centers = (edges[:-1] + edges[1:])/2.
    ax.plot(centers, counts)

    
    

for dpath in data_dirs:

    try:
        with open(dpath+'/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        with open(dpath+'/lts.p', 'rb') as pfile:
            lts_df=np.array(pickle.load(pfile))

        params = ['Nsteps', 'Nprocess',
                  'an_mu', 'an_sig',
                  'bn_mu', 'bn_sig']

        if nsp['var'] in params:
            params.remove(nsp['var'])

        notation = {'Nsteps'   : r'N_{\text{steps}}',
                    'Nprocess' : r'N_{\text{processes}}',
                    'an_mu'    : r'\mu_a',
                    'an_sig'   : r'\sigma_a',
                    'bn_mu'    : r'\mu_b',
                    'bn_sig'   : r'\sigma_b'}

        survival_probability(axes[0,0], nsp, lts_df)
        equilibrium_distribution(axes[0,1], nsp, lts_df)

        lifetime_distribution(axes[1,0], nsp, lts_df)
        equilibrium_distribution_log(axes[1,1], nsp, lts_df, notation)
        
    except FileNotFoundError:
        print(dpath[-4:], "reports: Error loading namespace")



# add Brownian motion survival probability
xs = np.logspace(0, np.log10(nsp['Nsteps']), num=1000)
axes[0,0].plot(xs, (1+xs)**(-0.5), color='grey', linestyle='dashed')


for ax in axes.flatten():

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

fig.suptitle('$X_{n+1} = a_n X_n + b_n\,\,$ pruned at $c=' + \
              str(nsp['c']) + '$ with ' + r' $P_{\mathrm{prune}} = ' + \
              str(nsp['p_prune']) +'$') 



axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')

axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')


axes[0,0].set_xlabel('simulation steps')
axes[0,0].set_ylabel('survival probability')

axes[1,0].set_xlabel('lifetime')
axes[1,0].set_ylabel('probability density')


axes[0,0].legend(frameon=False, loc='lower left',
              prop={'size': 9})


axes[0,1].set_xlabel(r'$X(t_{\mathrm{max}})$')
axes[0,1].set_ylabel('probability density')


axes[1,1].set_xlabel(r'$\log_{10} X(t_{\mathrm{max}})$')
axes[1,1].set_ylabel('probability density')

# text = r'$\mu_{a}=' + '%.5E $' % Decimal(nsp['an_mu']) +\
#            '\n' + r'$\sigma_{a}=' + '%.2E $' % Decimal(nsp['an_sig']) +\
#            '\n' + r'$\mu_{b}=' + '%.5E $' % Decimal(nsp['bn_mu']) +\
#            '\n' + r'$\sigma_{b}=' + '%.2E $' % Decimal(nsp['bn_sig']) +\
#            '\n ---------------- ' +\
#            '\n' + r'$N_{\text{process}} = ' + str(nsp['Nprocess']) +'$' +\
#            '\n' + r'$N_{\text{steps}} = ' + str(nsp['Nsteps']) +'$'


text = ''
for j, prm in enumerate(params):
    if j>0: text += '\n'
    text += '$' + notation[prm] +'='+ str(nsp[prm]) + '$'

axes[0,1].text(1.1, 0.95, text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    linespacing = 1.95,
                    fontsize=10,
                    bbox={'boxstyle': 'square, pad=0.3',
                          'facecolor':'white', 'alpha':1,
                          'edgecolor':'none'},
                    transform = axes[0,1].transAxes,
                    clip_on=False)


axes[1,1].legend(bbox_to_anchor=(1.04,1), loc="upper left",
                 frameon=False)

# ax.set_ylim(10**(-6),1)


fig.tight_layout()
fig.subplots_adjust(top=0.92)

directory = 'figures/'

if not os.path.exists(directory):
    os.makedirs(directory)


fname = 'master_1x'

fig.savefig(directory+fname+'.png', dpi=300,
            bbox_inches='tight')


