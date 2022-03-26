# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:58:44 2021

@author: beatr
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
import pandas as pd


#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)

npzfile = np.load("sHDP_DGP0.npz")
thetareloaded = npzfile['theta']
#selected = npzfile['selected']
xreloaded = npzfile['x']
xireloaded = npzfile['xi']

for mm in selected:
    fig = plt.figure()
    ax = plt.axes()
    plt.xlim(0, 8)
    plt.yticks(range(6),labels=['','Pop1', 'Pop2', 'Pop3','Pop4',''])
    s = np.nanvar(xreloaded[:,mm])**(1/2)
    m = np.nanmean(xreloaded[:,mm])
    
    plt.plot(np.mean(thetareloaded[mm-1,5002:10001,0]) * s + m, 1, 'o', color='black')
    plt.plot(np.mean(thetareloaded[mm-1,5002:10001,1]) * s + m, 2, 'o', color='black')
    plt.plot(np.mean(thetareloaded[mm-1,5002:10001,2]) * s + m, 3, 'o', color='black')
    plt.plot(np.mean(thetareloaded[mm-1,5002:10001,3]) * s + m, 4, 'o', color='black')
    
    plt.plot([np.quantile(thetareloaded[mm-1,5002:10001,0], 0.025) * s + m,\
              np.quantile(thetareloaded[mm-1,5002:10001,0], 0.975) * s + m],[1,1], color='black')
    plt.plot([np.quantile(thetareloaded[mm-1,5002:10001,1], 0.025) * s + m,\
              np.quantile(thetareloaded[mm-1,5002:10001,1], 0.975) * s + m],[2,2], color='black')
    plt.plot([np.quantile(thetareloaded[mm-1,5002:10001,2], 0.025) * s + m,\
              np.quantile(thetareloaded[mm-1,5002:10001,2], 0.975) * s + m],[3,3], color='black')
    plt.plot([np.quantile(thetareloaded[mm-1,5002:10001,3], 0.025) * s + m,\
              np.quantile(thetareloaded[mm-1,5002:10001,3], 0.975) * s + m],[4,4], color='black')
    
    plt.axvline(1, 0, 5, color='green',linestyle="--", label='Pop1')
    #plt.axvline(1.317116885, 0, 5, color='green',linestyle="--", alpha=0.5, label='sample mean Pop1')
    
    plt.axvline(3, 0, 5, color='orange',linestyle=":", label='Pop2')
    #plt.axvline(3.012893694, 0, 5, color='orange',linestyle=":", alpha=0.5, label='sample mean Pop2')
    
    plt.axvline(5, 0, 5, color='red',linestyle="-.", label='Pop3')
    #plt.axvline(5.184102712, 0, 5, color='red',linestyle="-.", alpha=0.5, label='sample mean Pop3')
    
    plt.axvline(7, 0, 5, color='blue', label='Pop4')
    #plt.axvline(7.086706189, 0, 5, color='blue', alpha=0.5, label='sample mean Pop4')
    plt.legend()
    
    fig.savefig('confmean%s.png' % mm, dpi=300)
    
    
    
    cmap = sns.color_palette("GnBu", as_cmap=True)  
    
    xi_final = xireloaded[mm-1,:,:]
    
    
    tot_iter = 10001
    burnin = 5001
    ccp = np.ones((100,100)) * np.nan
    for i in range(100):
        for ii in range(i+1,100):
            ccp[i,ii] = np.count_nonzero(xi_final[5001:10001,i]==xi_final[5001:10001,ii])
            ccp[ii,i] = ccp[i,ii]
                           
    ccp = ccp / (tot_iter-burnin)
    
       
    P1=sns.heatmap(ccp, cmap=cmap, cbar=False, vmin = np.min(ccp), vmax= np.max(ccp), square=True,\
                   yticklabels=False, xticklabels=False)
    plt.axvline(0, 0.5, 1, linewidth=2, color='black')
    plt.axvline(50, 0.31,1, linewidth=2, color='black')
    plt.axvline(69, 0.22,0.5,linewidth=2, color='black')
    plt.axvline(78, 0,0.31, linewidth=2,color='black')
    plt.axvline(100, 0,0.22,linewidth=2, color='black')
    plt.axhline(0, 0, 0.5, linewidth=2,color='black')
    plt.axhline(50, 0, 0.69, linewidth=2,color='black')
    plt.axhline(69, 0.5,0.78,linewidth=2, color='black')
    plt.axhline(78, 0.69,1, linewidth=2,color='black')
    plt.axhline(100, 0.78,1,linewidth=2, color='black')
    fig =P1.get_figure()        
    fig.savefig('cluster_scenario0_natord%s' % mm, dpi=300)
    #overall order
    for i in range(100):
        ccp[i,i]=1
            
    df = pd.DataFrame(columns=np.arange(0,100))
    d = sch.distance.pdist(ccp)
    L = sch.linkage(d, method='ward')
    ind = sch.fcluster(L, t=0.7, criterion='distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    ccpor = ccp[:,columns]
    ccpor = ccpor[columns,:]
    
    for i in range(100):
        ccpor[i,i]=np.nan
        
    P2=sns.heatmap(ccpor, cmap=cmap, cbar=False, vmin = np.min(ccp), vmax= np.max(ccp), square=True,\
                   yticklabels=False,xticklabels=False)
    fig2 =P2.get_figure()
    fig2.savefig('cluster_scenario0_reord%s' % mm, dpi=300)  
    plt.show()