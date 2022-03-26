"""
Python code to run to reproduce results on real data.
"""

import os
from pathlib import Path
import time 

import numpy as np
import pandas as pd
from scipy.special import gamma
import math

import ANOVA_sHDP as an


#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)

#I. IMPORT DATA################################################################
#use order and unif to set one of the three alternative priors
order = False 
unif = True
d = 4 #number of distinct populations
M = 10 #number of response variables considered
n = np.ones(d)  #statistical units per each population
n[0] = 50
n[1] = 19
n[2] = 9
n[3] = 22
n = n.astype(int)
#x is a 3d array contaning data
x = np.ones( (M, int(np.max(n)), d)) *np.nan
for j in range(d):
    x[:,0:n[j],j] = pd.read_excel('data.xlsx', sheet_name=j, header=None)
#standardize data    
x = an.standardrow(x)

#II. SET HYPERPARAMETERS#######################################################

#1.base dist for the systematic component
mu = np.zeros(M)  #loc parameter of base dist. of DP over means
s_th = np.ones(M) #var parameter of base dist. of DP over means

#2.base distribution for the idiosyncratic component (NIG)
k_eps = np.ones(M) * 1
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


#III. MCMC PARAMETERS##########################################################

verbose = 'T' #if set to T,every 500 iterations a message is displayed
np.random.seed(0) #set seed for replicability 
tot_iter = 10000 #total numer of iterations (including burnin)
burnin = 5000

###############################################################################
#MCMC - INITIALIZATION:

[theta,eps,c,C,nc,sign,k,K,nk,xi_star,sigma_star,phi_tab,sigma_tab,\
           xi_eps,sigma_eps,z,t,T,nt, theta_tab,alpham,gammam,omega,v] =\
     an.init(M,d,n,tot_iter,x)

###########################################################################
#MCMC - SAMPLING
print('MCMC simulations has started')
start = time.time()
for numiter in range(1,tot_iter+1):
    if verbose=='T':
        if (numiter/100==int(numiter/100)):
            print('iteration number:', numiter)
            end = time.time()
            print('time in seconds for last 100 iters:', end - start)
            start = time.time()
            
    #"update" some values for coding convenience
    T[:,numiter] = T[:,numiter-1]
    omega[numiter] = omega[numiter-1]
    
    theta[:,:,numiter] = theta[:,:,numiter-1]
    #file0 = open("thetareal.txt", "a")
    #np.savetxt(file0, np.transpose(theta[:,:,numiter]))      
    #file0.close()
    alpham[:,numiter] = alpham[:,numiter-1]
    gammam[:,:,numiter] = gammam[:,:,numiter-1]
    
    phi_tab[:,:,numiter] = phi_tab[:,:,numiter-1]
    sigma_tab[:,:,numiter] = sigma_tab[:,:,numiter-1]
    nk[:,:,numiter] = nk[:,:,numiter-1]
    
    #compute eps (idiosyncratic components) 
    for j in range(d):
        eps[:,:,j] =  x[:,:,j] - np.repeat( theta[:,j,numiter], np.max(n))\
                                            .reshape((M,np.max(n))) 
                                            
##1. sampling idiosyncratic components 
#from Hierarchical invariant Dirichlet Process Mixture#########################                                      
    for mm in range(M):#response variable
        #1.a.sample cijm: ############################################
        [c,k,nc,nk,xi_star,sigma_star,phi_tab,sigma_tab] = \
            an.sample_c(n,d,mm,numiter,k_eps,alpha_eps,beta_eps,\
            c,k,nc,nk,xi_star,sigma_star,eps,K,\
            phi_tab,sigma_tab,alpham,gammam)
             
        #1.b. sample signij: ##########################################
        sign = an.sample_sign(n,d,mm,eps,c,xi_star,sigma_star,sign)
        
        #1.c.sample k_cjm: ############################################
        [k,nk,xi_star,sigma_star,phi_tab,sigma_tab] = \
            an.sample_k(n,d,mm,numiter,k_eps,alpha_eps,beta_eps,\
            c,k,nc,nk,xi_star,sigma_star,eps,K,\
            phi_tab,sigma_tab,alpham,gammam,sign)
         
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
            tosampleom = np.random.gamma(a, 1/b, size=10000)
            p = tosampleom ** (np.sum(T[:,numiter])-M) *\
                ((tosampleom + 2) * (tosampleom ** 2 + tosampleom + 3)) ** (-M)
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
        
check = np.zeros((M,15))
check_round = np.zeros((15,M))     
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
    
    for mod in range(15):
        check_round[mod,mm] = round(check[mm,mod], 3)
    file1 = open("aggregatethetareal.txt", "a")
    np.savetxt(file1, np.transpose(check_round))      
    file1.close()
    
#save simulations in .npz format
np.savez("sHDPreal_unif", T = T, omega = omega, theta = theta, xi_eps = xi_eps, \
         sigma_eps= sigma_eps, alpham = alpham,\
         gammam = gammam, phi_tab = phi_tab, sigma_tab = sigma_tab, nk = nk)  

