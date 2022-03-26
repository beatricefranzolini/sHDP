# -*- coding: utf-8 -*-
"""
Python code to run to reproduce summaries for real data (to run after main.py)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import seaborn as sns
import scipy.cluster.hierarchy as sch

M=10
tot_iter=10000
burnin = 5000
d=4

#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)

n = np.ones(d)  #statistical units per each population
n[0] = 50
n[1] = 19
n[2] = 9
n[3] = 22
n = n.astype(int)

#1.base dist for the systematic component
mu = np.zeros(M)  #loc parameter of base dist. of DP over means
s_th = np.ones(M) * 1 #var parameter of base dist. of DP over means

#2.base distribution for the idiosyncratic component (NIG)
k_eps = np.ones(M) * 1
alpha_eps = np.ones(M) * 2
beta_eps = np.ones(M) * 4

x = np.ones( (M, int(np.max(n)), d)) *np.nan
for j in range(d):
    x[:,0:n[j],j] = pd.read_excel('data.xlsx', sheet_name=j, header=None)
    
npzfile = np.load("sHDPreal_mainnew.npz")
T = npzfile['T']
theta = npzfile['theta']
xi_eps = npzfile['xi_eps']
sigma_eps = npzfile['sigma_eps']
alpham = npzfile['alpham']
gammam = npzfile['gammam']
phi_tab = npzfile['phi_tab']
sigma_tab = npzfile['sigma_tab']
nk = npzfile['nk']

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller
       
something = list(("C","GH","MP", "SP"))
models = []
for nn, p in enumerate(partition(something), 1):
    models.append((nn-1, sorted(p)))
   
check = np.zeros((M,15))
check_round = np.zeros((15,10))

for mm in range(M):

    check[mm,0] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
   
    check[mm,1] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
     
    check[mm,2] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
   
    check[mm,3] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
   
    check[mm,4] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)  
   
    check[mm,5] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)  
   
    check[mm,6] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
   
    check[mm,7] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,3,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)
     
    check[mm,8] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)
   
    check[mm,9] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) ) / \
     (tot_iter+1-burnin)    
   
    check[mm,10] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)
     
    check[mm,11] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)      
     
    check[mm,12] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)      
     
    check[mm,13] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]==theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,3,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,3,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)    
     
    check[mm,14] = np.sum(\
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,1,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,0,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,2,burnin:tot_iter+1]) & \
        (theta[mm,1,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1]) & \
        (theta[mm,2,burnin:tot_iter+1]!=theta[mm,3,burnin:tot_iter+1])) / \
     (tot_iter+1-burnin)

for mm in range(M):
    for mod in range(15):
        check_round[mod,mm] = round(check[mm,mod],3)
       
t = Table([models, check_round])
print(t)

#CI for locations#############################################################
for mm in range(M):
    fig = plt.figure()
    ax = plt.axes()
    plt.xlim(np.nanmin(x[mm,:,:]), np.nanmax(x[mm,:,:]))
    plt.yticks(range(6),labels=['','Pop1', 'Pop2', 'Pop3','Pop4',''])
    s = np.nanvar(x[mm,:,:])**(1/2)
    m = np.nanmean(x[mm,:,:])
    
    plt.plot(np.mean(theta[mm,0,burnin:tot_iter+1]) * s + m, 1, 'o', color='black')
    plt.plot(np.mean(theta[mm,1,burnin:tot_iter+1]) * s + m, 2, 'o', color='black')
    plt.plot(np.mean(theta[mm,2,burnin:tot_iter+1]) * s + m, 3, 'o', color='black')
    plt.plot(np.mean(theta[mm,3,burnin:tot_iter+1]) * s + m, 4, 'o', color='black')
    
    plt.plot([np.quantile(theta[mm,0,burnin:tot_iter+1], 0.025) * s + m,\
              np.quantile(theta[mm,0,burnin:tot_iter+1], 0.975) * s + m],[1,1], color='green',linestyle="--")
    plt.plot([np.quantile(theta[mm,1,burnin:tot_iter+1], 0.025) * s + m,\
              np.quantile(theta[mm,1,burnin:tot_iter+1], 0.975) * s + m],[2,2], color='orange',linestyle=":")
    plt.plot([np.quantile(theta[mm,2,burnin:tot_iter+1], 0.025) * s + m,\
              np.quantile(theta[mm,2,burnin:tot_iter+1], 0.975) * s + m],[3,3], color='red',linestyle="-.")
    plt.plot([np.quantile(theta[mm,3,burnin:tot_iter+1], 0.025) * s + m,\
              np.quantile(theta[mm,3,burnin:tot_iter+1], 0.975) * s + m],[4,4], color='blue')
    plt.title('E/A')
    fig.savefig('confmean%s.png' %mm, dpi=300)
#cluster####CI################################################################ 
cmap = sns.color_palette("GnBu", as_cmap=True)  

xi = xi_eps[mm,:,:,:]
temporaryxi = xi.reshape((200,10001), order = 'F')
xi_final = np.transpose(temporaryxi[~np.isnan(temporaryxi)].reshape((100,10001)))

tot_iter = 10001
burnin = 5001
ccp = np.ones((100,100)) * np.nan
for i in range(100):
    for ii in range(i+1,100):
        ccp[i,ii] = np.count_nonzero(xi_final[5001:10001,i]==xi_final[5001:10001,ii])
        ccp[ii,i] = ccp[i,ii]
                       
ccp = ccp / (tot_iter-burnin)

minmin = np.nanmin(ccp)
maxmax = np.nanmax(ccp)
P1=sns.heatmap(ccp, cmap=cmap, cbar=False, vmin = minmin, vmax= maxmax, square=True,\
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
fig.savefig('cluster_EA_unif', dpi=300)
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
    
P2=sns.heatmap(ccpor, cmap=cmap, cbar=False, vmin = minmin, vmax= maxmax, square=True,\
               yticklabels=False,xticklabels=False)
fig2 =P2.get_figure()
fig2.savefig('cluster_ord_EA_unif', dpi=300)  
plt.show()

d = 4
mm = 9
#density estimation
#sampling the predictive
howmanyfordens = 10000
y = np.ones(howmanyfordens) * np.nan
pop = np.ones((howmanyfordens,d)) * np.nan
for j in range(d):
    for l in range(howmanyfordens):
        sim_num = np.random.choice(range(tot_iter-burnin))
        p = np.ones(n[j]+1)*(1/(n[j]+gammam[mm,j,sim_num+burnin]))
        p[n[j]] = gammam[mm,j,sim_num+burnin]/(n[j]+gammam[mm,j,sim_num+burnin])
        obs_num = np.random.choice(range(n[j]+1), p=p)
        if obs_num < n[j]:
            y[l] = theta[mm,j,sim_num+burnin] +\
                    np.random.normal(xi_eps[mm,obs_num,j,sim_num+burnin],\
                                np.power(sigma_eps[mm,obs_num,j,sim_num+burnin],1/2))
        else:
            p2 = np.ones(201)
            p2[0:200] = nk[mm,:,sim_num+burnin]
            p2[200] = alpham[mm,sim_num+burnin]
            p2 = p2 / sum(p2)
            tab_num = np.random.choice(range(len(p2)),p=p2)
            if tab_num < 200:
                y[l]=theta[mm,j,sim_num+burnin] +\
                    np.random.normal(phi_tab[mm,tab_num,sim_num+burnin],\
                                np.power(sigma_tab[mm,tab_num,sim_num+burnin],1/2))
            else:    
                s2 = 1/np.random.gamma(alpha_eps[mm],\
                       beta_eps[mm])
                mul = np.random.normal(0, np.power(s2/1,1/2))
                y[l] = theta[mm,j,sim_num+burnin] +\
                    np.random.normal(mul,np.power(s2,1/2))
    pop[:,j] = y * np.nanvar(x[mm,:,:])**(1/2) + np.nanmean(x[mm,:,:])
   

df=pd.DataFrame(pop)
df.columns = ['control','gestional hypertension','mild preeclampsia','severe preeclampsia']
p1=sns.kdeplot(df['control'], shade=True, color="green",linestyle="--", label = 'C')
p1=sns.kdeplot(df['gestional hypertension'], shade=True, color="orange",linestyle=":",\
               label = 'G')
p1=sns.kdeplot(df['mild preeclampsia'], shade=True, color="red", linestyle="-.",\
               label = 'M')
p1=sns.kdeplot(df['severe preeclampsia'], shade=True, color="b",\
               label = 'S')
plt.legend()
plt.title('EA')
fig =p1.get_figure()

fig.savefig('EA_unif.png', dpi=300)