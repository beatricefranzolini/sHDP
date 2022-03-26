"""
Python code to run to reproduce simulation studies.
"""

import os
from pathlib import Path
import time

import numpy as np
from scipy.special import gamma
import math

import ANOVA_sHDP as an
import pandas as pd

#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)

#I. SIMULATE DATA################################################################
order = True #true to use the main prior (with unif=False below) 
unif = False #true to use the uniform independent prior 
# If both order and unif above are false the prior used is the mixture of DPs

S = 1 #number of simulation studies under the same DGP
d = 4 #number of distinct populations
M = 1 #number of response variables considered
mm = 0
n = np.ones(d)  #statistical units per each population
n[0] = 50
n[1] = 19
n[2] = 9
n[3] = 22
n = n.astype(int)
N = sum(n)
np.random.seed(0) #set seed for replicability 

s = np.sqrt(1) #standard dev of the DGP
x_sim = np.zeros((N, S+1))
#population labels:
x_sim[:,0] = np.concatenate((np.repeat(1, n[0]),\
                             np.repeat(2, n[1]),\
                             np.repeat(3, n[2]),\
                             np.repeat(4, n[3])))
  
#prepare table to collect posterior prob:
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

#generate data for S simulation studies:
k0 = np.zeros(S)
k1 = np.zeros(S)
k2 = np.zeros(S)
k3 = np.zeros(S)

for sim in range(S):
    """
    #DGP0
    k0[sim] = np.random.binomial(n[0], 0.5)
    k1[sim] = np.random.binomial(n[1], 0.5)
    k2[sim] = np.random.binomial(n[2], 0.5)
    k3[sim] = np.random.binomial(n[3], 0.5)
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.normal(0, np.sqrt(0.5), int(k0[sim])),\
                        np.random.normal(2, np.sqrt(0.5), n[0]-int(k0[sim])),\
                        np.random.normal(2, np.sqrt(0.5), int(k1[sim])),\
                        np.random.normal(4, np.sqrt(0.5), n[1]-int(k1[sim])),\
                        np.random.normal(4, np.sqrt(0.5), int(k2[sim])),\
                        np.random.normal(6, np.sqrt(0.5), n[2]-int(k2[sim])),\
                        np.random.normal(6, np.sqrt(0.5), int(k3[sim])),\
                        np.random.normal(8, np.sqrt(0.5), n[3]-int(k3[sim]))))
    """
    """
    #DGP1
    k0[sim] = np.random.binomial(n[0], 0.5)
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.normal(0, np.sqrt(0.5), n[0]-1),\
                        np.random.normal(4, np.sqrt(0.5), 1),\
                        np.random.normal(1, np.sqrt(0.5), n[1]),\
                        np.random.normal(1, np.sqrt(0.5), n[2]),\
                        np.random.normal(2, np.sqrt(0.5), n[3])))
        """
        
    """
    #DGP2
    k0[sim] = np.random.binomial(n[0], 0.5)
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.normal(-1, np.sqrt(0.5), int(k0[sim])),\
                        np.random.normal(1, np.sqrt(0.5), n[0]-int(k0[sim])),\
                        np.random.normal(1, np.sqrt(0.5), n[1]),\
                        np.random.normal(1, np.sqrt(0.5), n[2]),\
                        np.random.normal(2, np.sqrt(0.5), n[3])))
        
        """
    """
    #DGP3
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.normal(0, np.sqrt(0.5), n[0]),\
                        np.random.gamma(3, 1/3, n[1]),\
                        np.random.normal(1, np.sqrt(0.5), n[2]),\
                        np.random.normal(2, np.sqrt(0.5), n[3])))
        """
    """
    #DGP4
    k0[sim] = np.random.binomial(n[0], 0.7)
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.normal(-1, np.sqrt(0.5), int(k0[sim])),\
                        np.random.normal(1, np.sqrt(0.5), n[0]-int(k0[sim])),\
                        np.random.normal(1, np.sqrt(0.5), n[1]),\
                        np.random.normal(1, np.sqrt(0.5), n[2]),\
                        np.random.normal(2, np.sqrt(0.5), n[3])))  
          """
    #DGP5
    k0[sim] = np.random.binomial(n[3], 0.5)
    x_sim[:,sim+1]  = np.concatenate((
                        np.random.gamma(10, 1/10, n[0]),\
                        np.random.gamma(10, 1/10, n[1]),\
                        np.random.gamma(10, 1/10, n[2]),\
                        np.random.normal(0, np.sqrt(0.5), int(k0[sim])),\
                        np.random.normal(2, np.sqrt(0.5), int(n[3]-k0[sim]))))
        
with pd.ExcelWriter('simuldata1.xlsx') as writer:  
    df1 = pd.DataFrame(x_sim)
    df1.to_excel(writer, sheet_name='Simdata', index = False)
    '''
    df2 = pd.DataFrame((k0, k1, k2, k3))
    df2.to_excel(writer, sheet_name='trueclustering', index = False)
    '''

#randomly select 3 simulation studies
selected = np.random.choice(range(1, S+1), 3)
        
#II. SET HYPERPARAMETERS######################################################
    
#1.base dist for the systematic component
mu = np.zeros(M)  #loc param of base dist. of DP over means
s_th = np.ones(M) #var param of base dist. of DP over means (updated-EB later)

#2.base distribution for the idiosyncratic component (NIG)
k_eps = np.ones(M)
alpha_eps = np.ones(M) * 2
beta_eps = np.ones(M) * 4 

#3.hyperparameters for the concentrations
#omega:
a = 3
b = 3 
#alpha:
g = 3
h = 3
#gamma:
e = 3
f = 3 

#III. MCMC PARAMETERS#########################################################
    
verbose = 'T' #if set to T,every 500 iterations a message is displayed
tot_iter = 10000 #total numer of iterations (including burnin)
burnin = 5000

theta_final_value = np.ones((S, tot_iter+1, d)) * np.nan
xi_final = np.ones((S, tot_iter, N)) * np.nan
sigma_final = np.ones((S, tot_iter, N)) * np.nan

#IV. MCMC#####################################################################
for sim in range(1, S+1):
    print("simulation study number:" , sim)
    
    #x is a 3d array contaning data for a single simulation study
    x = np.ones( (M, int(np.max(n)), d)) * np.nan
    x[:, 0:n[0], 0] = x_sim[0:n[0], sim]
    x[:, 0:n[1], 1] = x_sim[(n[0]):(n[1]+n[0]), sim]
    x[:, 0:n[2], 2] = x_sim[(n[0]+n[1]):(n[2]+n[1]+n[0]), sim]
    x[:, 0:n[3], 3] = x_sim[(n[0]+n[1]+n[2]):(n[3]+n[2]+n[1]+n[0]), sim]
    #standardize data    
    x = an.standardrow(x)
    
    ###############################################################################
    #MCMC - INITIALIZATION:
    
    [theta, eps, c, C, nc, sign, k, K, nk,\
     xi_star, sigma_star, phi_tab, sigma_tab,\
     xi_eps, sigma_eps, z, t, T, nt, theta_tab,\
     alpham, gammam, omega, v] =\
         an.init(M, d, n, tot_iter, x)
      
    ###########################################################################
    #MCMC - SAMPLING
    start = time.time() 
    for numiter in range(1, tot_iter+1):
        if verbose=='T' and (numiter/500 == int(numiter/500)):
                print('iteration number:', numiter)
                end = time.time()
                print('time in seconds for 500 iter:', end - start) 
                start = time.time()   
        #"update" values
        T[:, numiter] = T[:,numiter-1]
        omega[numiter] = omega[numiter-1]
        
        theta[:, :, numiter] = theta[:, :, numiter-1]
        alpham[:, numiter] = alpham[:, numiter-1]
        gammam[:, :, numiter] = gammam[:, :, numiter-1]
        
        phi_tab[:, :, numiter] = phi_tab[:, :, numiter-1]
        sigma_tab[:, :, numiter] = sigma_tab[:, :, numiter-1]
        nk[:, :, numiter] = nk[:, :, numiter-1]
        
        #compute eps (idiosyncratic components) 
        for j in range(d):
            eps[:, :, j] =  x[:, :, j] -\
                            np.repeat( theta[:, j, numiter], np.max(n))\
                                                .reshape((M, np.max(n))) 
                                                
        ##1. sampling idiosyncratic components 
        #from s-HDP Mixture#####################                                      
        #1.a.sample cijm: ############################################
        [c, k, nc, nk, xi_star, sigma_star, phi_tab, sigma_tab] = \
            an.sample_c(n, d, mm, numiter, k_eps, alpha_eps, beta_eps,\
            c, k, nc, nk, xi_star, sigma_star, eps, K,\
            phi_tab, sigma_tab, alpham, gammam)
             
        #1.b. sample signij: ##########################################
        sign = an.sample_sign(n, d, mm, eps, c, xi_star, sigma_star, sign)
        
        #1.c.sample k_cjm: ############################################
        [k, nk, xi_star, sigma_star, phi_tab, sigma_tab] = \
            an.sample_k(n, d, mm, numiter, k_eps, alpha_eps, beta_eps,\
            c, k, nc, nk, xi_star, sigma_star, eps, K,\
            phi_tab, sigma_tab, alpham, gammam, sign)
         
        #1.d.sample phi_tab: ######################################### 
        [sigma_tab,phi_tab,sigma_star,xi_star,sigma_eps,xi_eps] =\
        an.sample_phitab(n,d,mm,numiter,c,nc,k,nk,sign,eps,\
                         alpha_eps,k_eps,beta_eps,\
                  sigma_tab,phi_tab,sigma_star,xi_star,sigma_eps,xi_eps)

        #1.e.sample gamma #############################################
        for j in range(d):    
            u = np.random.beta(gammam[mm,j,numiter]+1, n[j])
            C[mm,j] = np.size(np.unique(c[mm,~np.isnan(c[mm,:,j]),j]))
            C = C.astype(int)
            v = np.random.binomial(1, n[j]/(n[j]+gammam[mm,j,numiter]))
            gammam[mm,j,numiter] = np.random.gamma(e + C[mm,j] - v, \
                  1/(f - np.log(u)))
        
        #1.f.sample alpha #############################################
        
        u = np.random.beta(alpham[mm,numiter]+1,sum(C[mm,:]))
        K[mm] = np.size(np.unique(k[mm,~np.isnan(k[mm,:,:])]))
        va = np.random.binomial(1, sum(C[mm,:])*(h-math.log(u))/\
                 (sum(C[mm,:])*(h-math.log(u)) + g + K[mm] -1))
        alpham[mm,numiter] = np.random.gamma(g + K[mm] - va, \
              1/(h - math.log(u)))
            
#####2.sampling systematic components#########################################
        z[mm,:,:] = (x[mm,:,:] - xi_eps[mm,:,:,numiter])
                     
        #2.a.sample tj: ###############################################
        if unif:
            [t,nt,theta,theta_tab] =\
        an.sample_t_unif(n,d,mm,numiter,t,nt,z,theta_tab,theta,\
                    sigma_eps,s_th,mu,omega)
        else:
            if order:
                [t,nt,theta,theta_tab] =\
            an.sample_t_ord(n,d,mm,numiter,t,nt,z,theta_tab,theta,\
                        sigma_eps,s_th,mu,omega)
            else:
                [t,nt,theta,theta_tab] =\
            an.sample_t(n,d,mm,numiter,t,nt,z,theta_tab,theta,\
                        sigma_eps,s_th,mu,omega)
        
        #2.c sample theta_tab: ####################################################

        [theta_tab,theta] =\
        an.sample_thetatab(n,d,mm,numiter,t,nt,\
                           z,sigma_eps,s_th,mu,theta_tab,theta)          
        
        T[mm,numiter] = np.size(np.unique(t[mm]))

        #2.4 sample omega
        if unif==False:
            if order:
                tosampleom = np.random.gamma(a, 1/b, size=1000)
                p = np.zeros(1000)
                p = tosampleom**np.sum(T[:,numiter]) * gamma(tosampleom)**M /\
                    gamma(tosampleom + d)**M
                p = p / np.sum(p)
                omega[numiter] = np.random.choice(tosampleom, p = p)
            else:
                uuu = np.random.beta(omega[numiter]+1, d, M)
                prob_om = np.ones(M+1)
                    
                for kkkk in range(M+1):
                    prob_om[kkkk] = gamma(a+np.sum(T[:,numiter])-kkkk)*\
                    np.power(d,kkkk)*\
                    math.factorial(M)/math.factorial(M-kkkk)/math.factorial(kkkk)\
                    / np.power((b - np.sum(np.log(uuu))),a+np.sum(T[:,numiter])-kkkk)
                prob_om = prob_om / np.sum(prob_om)
                vvv = np.random.choice(range(len(prob_om)),p=prob_om)
                omega[numiter] = np.random.gamma(a + np.sum(T[:,numiter]) - vvv, \
                          1/(b - np.sum(np.log(uuu)))) 
    
    xi_eps = xi_eps[:,:,:,1:10001]
    sigma_eps = sigma_eps[:,:,:,1:10001]
    theta_final_value[sim-1,:,:] = np.transpose(theta[0, :, :])
    temporaryxi = xi_eps.reshape((200,10000), order = 'F')
    xi_final[sim-1,:,:]  = np.transpose(temporaryxi[~np.isnan(temporaryxi)].reshape((100,10000)))
    temporarysigma = sigma_eps.reshape((200,10000), order = 'F')
    sigma_final[sim-1,:,:]  = np.transpose(temporarysigma[~np.isnan(temporarysigma)].reshape((100,10000)))
           
            
        
    #save posterior prob.
    check = np.zeros((M,15))
    check_round = np.zeros((15,M))
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
    
    for mod in range(15):
        check_round[mod,mm] = round(check[mm,mod], 3)
    file1 = open("aggregatetheta.txt", "a")
    np.savetxt(file1, np.transpose(check_round))      
                             
    file1.close()
    
    if sim == selected[0]:
        with pd.ExcelWriter('selected0.xlsx') as writer:  
            df1 = pd.DataFrame(theta_final_value[sim-1,:,:])
            df1.to_excel(writer, sheet_name='theta', index = False)
            df2 = pd.DataFrame(xi_final[sim-1,:,:])
            df2.to_excel(writer, sheet_name='xi', index = False)
    
        
    if sim == selected[1]:
        with pd.ExcelWriter('selected1.xlsx') as writer:  
            df1 = pd.DataFrame(theta_final_value[sim-1,:,:])
            df1.to_excel(writer, sheet_name='theta', index = False)
            df2 = pd.DataFrame(xi_final[sim-1,:,:])
            df2.to_excel(writer, sheet_name='xi', index = False)
    
    if sim == selected[2]:
        with pd.ExcelWriter('selected2.xlsx') as writer:  
            df1 = pd.DataFrame(theta_final_value[sim-1,:,:])
            df1.to_excel(writer, sheet_name='theta', index = False)
            df2 = pd.DataFrame(xi_final[sim-1,:,:])
            df2.to_excel(writer, sheet_name='xi', index = False)

#save simulations in .npz format
np.savez("sHDP_DGP5_mainnew", x = x_sim, theta = theta_final_value, xi = xi_final, \
         sigma = sigma_final, selected = selected)  
#to recover simulations results
npzfile = np.load("sHDP_DGP5_mainnew.npz")
xreloaded = npzfile['x']
thetareloaded = npzfile['theta']
xireloaded = npzfile['xi']
sigmareloaded = npzfile['sigma']
selected = npzfile['selected']
    