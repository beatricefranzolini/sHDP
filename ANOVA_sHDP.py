# -*- coding: utf-8 -*-
"""
Python functions to estimate the model (need to run main.py and main_simul.py).
"""

import numpy as np
import copy
from scipy.stats import norm 
from scipy.stats import t as student
from scipy.special import loggamma
import math

#standardize data##############################################################
def standardrow(x):
    M = np.size(x,0)
    for mm in range(M):   
        x[mm,:,:] = (x[mm,:,:] - np.nanmean(x[mm,:,:])) \
        /np.nanvar(x[mm,:,:])**(1/2)
    return(x)
    
#initialize####################################################################
def init(M,d,n,tot_iter,x):
    ##1.initialization of conditional means in ANOVA#############################
    theta = np.ones( (M, d, tot_iter+1) ) * np.nan                  #SAVE SIMUL
    theta[:,:,0] = np.nanmean(x,axis=1)     
    #EXPL: theta[mm,j,numiter]=mean of pop j at iteration numiter for var mm
    
    ##2.initialization for idiosincratic components############################
    eps = np.ones( (M, max(n), d) ) * np.nan
    #EXPL: eps[mm,i,j]=idiosync. comp. of unit i in pop. j for var mm
    
    #c, C, nc refer to the underlying DP that induces the symmetric-DP (obs.)
    #c
    c = np.ones( (M, np.max(n), d) ) * np.nan #table index (like t later)
    for mm in range(M):
        for j in range(d):
            c[mm,0:n[j],j] = range(n[j]) #everyone sits alone 
    #c = c.astype(int) not to use otherwise problem with nan
    #EXPL: c[mm,i,j]=table index where i-th customer (from pop. j) sits(var mm)
    
    #C
    C = np.ones((M,d)) * np.nan #tot numb of tables
    for mm in range(M):    
        for j in range(d):
            C[mm,j] = np.size(np.unique(c[mm,~np.isnan(c[mm,:,j]),j]))
    C = C.astype(int)
    #C[mm,j]=number of components for density estimation of residuals in pop j
    
    #nc
    nc = np.zeros( (M, np.max(n), d)) #customers are classified by abs of xi
    for mm in range(M):
        for j in range(d):
            for cc in range(n[j]):
                nc[mm,cc,j] = 1 #each one alone, all tables occupied 
    nc = nc.astype(int)
    #EXPL: nc[mm,cc,j]=how many customers sit at table cc in restaurant j
    #NOTE: nc[mm,c[mm,i,j],j]=how many customers sit with i-th (i-th included) 
    
    sign = np.ones( (M, np.max(n), d) )*np.nan
    #EXPL: sign[mm,i,j]==-1 if xi_eps[mm,i,j]==-xi_tab[mm,c[mm,i,j],j]
    
    #k, K, nk refer to the underlying DP that induces the symmetric-DP (tables)
    #k
    k = np.ones( (M, np.max(n), d) ) * np.nan #dish index (like t later)
    for mm in range(M):
        temp = 0
        for j in range(d):
            k[mm,0:n[j],j] = range(temp,temp+n[j])#everyone eats someth. unique
            temp = temp + max(n)
    #k = k.astype(int) not to use otherwise problem with nan
    #EXPL: k[mm,cc,j]=dish index for cc-th table (from pop. j)
    #NOTE: k[mm,c[mm,i,j],j]=dish index eaten by unit i in pop j
         
    #K
    K = np.ones((M)) * np.nan #tot numb of dishes
    for mm in range(M):    
        K[mm] = np.size(np.unique(k[mm,~np.isnan(k[mm,:,:])]))
    K = K.astype(int)
    #K[mm]=number of distinct dishes in the whole franchise (for var mm)
    
    nk = np.zeros( (M, max(n)*d,tot_iter+1))                        #SAVE SIMUL
    for mm in range(M):
        nk[mm,:,0] = \
        ~np.isnan(k[mm,:,:].reshape(max(n)*d,order='F')).astype(int)+2  
    nk = nk.astype(int)
    #EXPL: nK[mm,kk]=how many tables eat dish kk
    #NOTE: nk[mm,k[mm,cc,j]]
    #NOTE: nk[mm,k[mm,c[mm,i,j]]]
    
    #values for table:
    xi_star = np.ones( (M, max(n), d) ) * np.nan
    sigma_star = np.ones( (M, max(n), d) ) 
    for j in range(d):
        xi_star[:,:,j] = x[:,:,j] - np.repeat( theta[:,j,0], np.max(n))\
                                                .reshape((M,np.max(n))) 
        #sigma_star[:,:,j] =  xi_star[:,:,j]**2           
    #EXPL: xi_star[mm,cc,j]=dish eaten at table cc in restaurant j (var mm)
    #EXPL: sigma_star[mm,cc,j] same as xi
    
    #values for dishes/orders                                       #SAVE SIMUL
    phi_tab = np.ones((M,max(n)*d,tot_iter+1))
    sigma_tab = np.ones((M,max(n)*d,tot_iter+1))
    phi_tab[:,:,0] = copy.deepcopy(xi_star.reshape(M,max(n)*d,order='F'))
    sigma_tab[:,:,0] = copy.deepcopy( sigma_star.reshape(M,max(n)*d,order='F'))  
    #EXPL: phi_tab[mm,kk]=dish with index kk (var mm)
    #EXPL: sigma_tab[mm,cc,j] same as phi
    
    #values for units
    xi_eps = np.ones( (M, max(n), d,tot_iter+1) ) * np.nan
    sigma_eps = np.ones( (M, max(n), d, tot_iter+1) ) *np.nan
    xi_eps[:,:,:,0] = xi_star                                       #SAVE SIMUL
    sigma_eps[:,:,:,0] = sigma_star                                 #SAVE SIMUL
    
    ##3.initialization for systematic components###############################
    z = np.ones( (M, int(np.max(n)), d) ) * np.nan
    #EXPL: z[mm,i,j]="realization of systematic component" of unit i in pop. j
        
    t = np.ones((M,d)) 
    for mm in range(M):
        t[mm] = range(d)  #table index (like c of first part) #all sat alone
    t = t.astype(int) 
    #EXP: t[mm,j]=table index where j-th pop sits
    
    T = np.ones((M,tot_iter+1)) * np.nan                            #SAVE SIMUL
    for mm in range(M):
        T[mm,0] = np.size(np.unique(t[mm]))  #tot numb. of tables
    
    nt = np.ones((M,d)) * np.nan  
    for tt in range(d):
        for mm in range(M):
            nt[mm,tt] = np.sum(t[mm]==tt)
    nt = nt.astype(int)
    #EXP: nt[mm,tt]=how many populations sit at table tt
    #NOTE: nt[mm,t[mm,j]]=how many populations sit with j-th (j-th included)
    
    theta_tab = np.ones((M,d))*np.nan 
    for mm in range(M):
        for tt in range(d):
            theta_tab[mm,tt] = theta[mm,np.where(t[mm]==tt)[0][0],0]
    #EXP: theta_tab[mm,tt]=dish served at table tt
    #NOTE: theta_tab[mm,t[mm,j]]==theta[mm,j,numiter]    
    
    ##4.initialization for concentration parameters############################
    alpham = np.ones((M,tot_iter+1))
    gammam = np.ones((M,d,tot_iter+1))
    omega = np.ones(tot_iter+1)          
    v = np.zeros(d)
    return(theta,eps,c,C,nc,sign,k,K,nk,xi_star,sigma_star,phi_tab,sigma_tab,\
           xi_eps,sigma_eps,z,t,T,nt, theta_tab,alpham,gammam,omega,v)     
#sample tables' indeces########################################################

def sample_c(n,d,mm,numiter,k_eps,alpha_eps,beta_eps,\
             c,k,nc,nk,xi_star,sigma_star,eps,K,\
             phi_tab,sigma_tab,alpham,gammam):
    
    for j in range(d):#restaurant
        
        for i in range(n[j]):#customer
            #remove unit i,j (eps_i,j stands up):
            if nc[mm,int(c[mm,i,j]),j]==1:#if she sits alone
                #reduce the number of tables
                nk[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] = \
                nk[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] - 1
                #remove index, 'cause no one else is at that table
                k[mm,int(c[mm,i,j]),j] = np.nan
            nc[mm,int(c[mm,i,j]),j] = nc[mm,int(c[mm,i,j]),j] - 1
            c[mm,i,j] = np.nan
            
            #compute full conditional of c[mm,i,j]:
            p = nc[mm,0:n[j],j]*(norm.pdf(eps[mm,i,j],  xi_star[mm,0:n[j],j], \
                                          sigma_star[mm,0:n[j],j]**(1/2)) +\
                                 norm.pdf(eps[mm,i,j], -xi_star[mm,0:n[j],j], \
                                          sigma_star[mm,0:n[j],j]**(1/2)))

            #p of sitting in a new table
            K[mm] = np.size(np.unique(k[mm,~np.isnan(k[mm,:,:])]))
            K.astype(int)
            temp = np.zeros(max(n)*d)
            temp[np.where(nk[mm,:,numiter]>0)] = \
                nk[mm,np.where(nk[mm,:,numiter]>0),numiter] / \
                (K[mm] + alpham[mm,numiter]) *\
                (norm.pdf(eps[mm,i,j],  \
                phi_tab[mm,np.where(nk[mm,:,numiter]>0),numiter], \
                sigma_tab[mm,np.where(nk[mm,:,numiter]>0),numiter]**(1/2)) +\
                norm.pdf(eps[mm,i,j], \
                -phi_tab[mm,np.where(nk[mm,:,numiter]>0),numiter], \
                sigma_tab[mm,np.where(nk[mm,:,numiter]>0),numiter]**(1/2)))

            #compute h_bar
            alpha_n = alpha_eps[mm] + 1/2
            k_n = k_eps[mm] + 1
            beta_n = beta_eps[mm] + \
                     k_eps[mm] * eps[mm,i,j]**2 / (2*(k_n))
            logp =  loggamma(alpha_n) - loggamma(alpha_eps[mm]) +\
                alpha_eps[mm]*math.log(beta_eps[mm]) -\
                alpha_n*math.log(beta_n) +\
                1/2*(math.log(k_eps[mm])-math.log(k_n)) - \
                1/2*math.log(2*math.pi)
            temp = np.append(temp, 2 * alpham[mm,numiter] / \
                    (K[mm] + alpham[mm,numiter]) * math.exp(logp))
            
            p = np.append(p,gammam[mm,j,numiter] * np.nansum(temp)) 

            p = p / sum(p) 

            #sample for the full conditional of c[mm,i,j]:
            c_sampled = np.random.choice( range(len(p)) , p = p)
         
            if c_sampled < n[j]: #sits to an "old" table
                
                c[mm,i,j] = c_sampled
                nc[mm,int(c[mm,i,j]),j] = nc[mm,int(c[mm,i,j]),j] + 1
                
            else: #sits to a "new" table
                
                c[mm,i,j] = np.setdiff1d(range(n[j]+1),c[mm,~np.isnan(c[mm,:,j]),j])[0]
                        
                    
                nc[mm,int(c[mm,i,j]),j] = 1
                
                #sample order for the new table
                temp = temp / sum(temp)
                k_sampled = np.random.choice(range(len(temp)),p=temp)
                
                if k_sampled < max(n)*d: #order an "old" dish
                    k[mm,int(c[mm,i,j]),j] = k_sampled
                    nk[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] = \
                    nk[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] + 1
                
                else: #new dish ordered
                    
                    #look for a new order index:

                    k[mm,int(c[mm,i,j]),j] = np.setdiff1d(range(max(n)*d+1),k[mm,~np.isnan(k[mm,:,:])])[0]

                    nk[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] = 1
                        
                    #new dish
                    sigma_tab[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] = \
                        1 / np.random.gamma( alpha_n, 1/beta_n ) 
                                             
                    #new loc parameter
                    phi_tab[mm,int(k[mm,int(c[mm,i,j]),j]),numiter]  = \
                        np.random.normal(eps[mm,i,j] / k_n,\
                        (sigma_tab[mm,int(k[mm,int(c[mm,i,j]),j]),numiter] \
                           / (k_n))**(1/2))
                        
                sigma_star[mm, int(c[mm,i,j]), j ] =  \
                    sigma_tab[mm,int(k[mm,int(c[mm,i,j]),j]),numiter]  
                xi_star[mm, int(c[mm,i,j]), j ] =  \
                    phi_tab[mm,int(k[mm,int(c[mm,i,j]),j]),numiter]
    
    return(c,k,nc,nk,xi_star,sigma_star,phi_tab,sigma_tab)

def samplec_DP(n,d,mm,numiter,k_eps,alpha_eps,beta_eps,\
             c,k,nc,nk,xi_star,sigma_star,eps,K,\
             phi_tab,sigma_tab,alpham,gammam,xi_eps, sigma_eps):
    
    for j in range(d):#restaurant
        
        for i in range(n[j]):#customer
            #remove unit i,j (eps_i,j stands up):
            nc[mm,int(c[mm,i,j])] = nc[mm,int(c[mm,i,j])] - 1
            c[mm,i,j] = np.nan
            p= []
            p = np.concatenate((p, nc[mm,:]*(norm.pdf(eps[mm,i,j], \
                                                                xi_star[mm,:], \
                                                  sigma_star[mm,:]**(1/2)))))
                    
            p = np.concatenate((p, [alpham[mm,numiter] * student.pdf(eps[mm,i,j], df = 2 * alpha_eps[mm],\
                                            loc = 0, \
                        scale = (beta_eps[mm] * (k_eps[mm] + 1) / (alpha_eps[mm] * k_eps[mm]))**(1/2))]),axis=0)
            p = p/np.sum(p)
            c_sampled = np.random.choice( range(len(p)) , p = p)
            if c_sampled < np.sum(n): #sits to an "old" table
                c[mm,i,j] = c_sampled
                nc[mm,int(c[mm,i,j])] = nc[mm,int(c[mm,i,j])] + 1
            else: #sits to a "new" table
                c[mm,i,j] = np.setdiff1d(range(np.sum(n)+1),c[mm,~np.isnan(c[mm,:])])[0]
                nc[mm,int(c[mm,i,j])] = 1
                
                aDP = alpha_eps[mm] + 1 / 2
                bDP = beta_eps[mm] + k_eps[mm] / (2 * (k_eps[mm] + 1)) * eps[mm,i,j]
                sDP = 1 / np.random.gamma(aDP, 1/bDP)
                
                mDP = 1 / (1  + k_eps[mm] ) * eps[mm,i,j] 
                tDP = 1 * 1/sDP + k_eps[mm] * 1/sDP
                xi_star[mm, int(c[mm,i,j])] = np.random.normal(mDP, tDP ** (-1/2))
                sigma_star[mm, int(c[mm,i,j])] = sDP
            xi_eps[mm,i,j,numiter] = xi_star[mm, int(c[mm,i,j])]
            sigma_eps[mm,i,j,numiter] = sigma_star[mm, int(c[mm,i,j])]
    return(c,k,nc,nk,xi_star,sigma_star,phi_tab,sigma_tab,xi_eps, sigma_eps)

#sample tables' side (sign)####################################################   
def sample_sign(n,d,mm,eps,c,xi_star,sigma_star,sign):
    for j in range(d):
        sign[mm,0:n[j],j] = np.ones(n[j])
        for i in range(n[j]):
            p3 = np.zeros(2)
            p3[0] = norm.pdf(eps[mm,i,j], xi_star[mm,int(c[mm,i,j]),j], \
                                sigma_star[mm,int(c[mm,i,j]),j]**(1/2))
            p3[1] = norm.pdf(eps[mm,i,j], -xi_star[mm,int(c[mm,i,j]),j],\
                                sigma_star[mm,int(c[mm,i,j]),j]**(1/2))
        
            p3 = p3 / sum(p3)
                
            sign_sampled = np.random.choice([False,True],p=p3)
                
            if sign_sampled:
                sign[mm,i,j]=-1  
    return(sign)

#sample dishes' indeces########################################################    
def sample_k(n,d,mm,numiter,k_eps,alpha_eps,beta_eps,\
             c,k,nc,nk,xi_star,sigma_star,eps,K,\
             phi_tab,sigma_tab,alpham,gammam,sign):  
    
    for j in range(d):            
        for cc in np.where(nc[mm,:,j]>0)[0]:
            #remove table cc,j:
            nk[mm,int( k[mm,cc,j]),numiter] = \
            nk[mm,int( k[mm,cc,j]),numiter] - 1
            k[mm,cc,j] = np.nan

            sume = np.sum(sign[mm,np.where(c[mm,:,j] == cc)[0],j] *\
                          eps[mm,np.where(c[mm,:,j] == cc)[0],j])
            sume2 = np.sum(eps[mm,c[mm,:,j] == cc,j]**2)
            
            #compute full conditional:
            p2 = np.zeros(max(n)*d)
            K[mm] = np.size(np.unique(k[mm,~np.isnan(k[mm,:,:])]))
            K.astype(int)
            p2[np.where(nk[mm,:,numiter]>0)] = \
                nk[mm,np.where(nk[mm,:,numiter]>0),numiter] *\
                (2*math.pi*\
                 sigma_tab[mm,np.where(nk[mm,:,numiter]>0),numiter])\
                 **(-nc[mm,cc,j]/2)*np.exp(-1/ \
                (2*sigma_tab[mm,np.where(nk[mm,:,numiter]>0),numiter])*\
                (sume2-2*phi_tab[mm,np.where(nk[mm,:,numiter]>0),numiter]*\
                 sume+nc[mm,cc,j]*\
                 phi_tab[mm,np.where(nk[mm,:,numiter]>0),numiter]**2))
                        
            alpha_n = alpha_eps[mm] + nc[mm,cc,j]/2
            k_n = k_eps[mm] + nc[mm,cc,j]
            beta_n = beta_eps[mm] + \
                +1/2*(sume2 - nc[mm,cc,j]*(sume/nc[mm,cc,j])**2)+\
            (k_eps[mm] * sume**2/nc[mm,cc,j])/ (2*(k_n))
            logp =  loggamma(alpha_n) - loggamma(alpha_eps[mm]) +\
                alpha_eps[mm]*math.log(beta_eps[mm]) -\
                alpha_n*math.log(beta_n) +\
                1/2*(math.log(k_eps[mm])-math.log(k_n)) - \
                nc[mm,cc,j]/2*math.log(2*math.pi)
            p2 = np.append(p2, alpham[mm,numiter] * math.exp(logp))

            p2 = p2 / sum(p2) 

            #sample for the full conditional:
            k_sampled = np.random.choice(range(len(p2)),p=p2)
    
            if k_sampled < max(n)*d: #order an "old" dish
                k[mm,cc,j] = k_sampled
                nk[mm,int(k[mm,cc,j]),numiter] = \
                nk[mm,int(k[mm,cc,j]),numiter] + 1
            
            else: #new dish ordered
            #look for a new order index:

                k[mm,cc,j] = np.setdiff1d(range(max(n)*d+1),k[mm,~np.isnan(k[mm,:,:])])[0]

                nk[mm,int(k[mm,cc,j]),numiter] = 1
                    
                #new dish
                sigma_tab[mm,int(k[mm,cc,j]),numiter] = \
                    1 / np.random.gamma( alpha_n, 1/beta_n ) 
                                         
                #new loc parameter
                
                phi_tab[mm,int(k[mm,cc,j]),numiter]  = \
                    np.random.normal(sume / k_n,\
                    (sigma_tab[mm,int(k[mm,cc,j]),numiter] \
                       / (k_n))**(1/2))
                    
            sigma_star[mm, cc, j ] =  \
                sigma_tab[mm,int(k[mm,cc,j]),numiter] 
            xi_star[mm, cc, j ] =  \
                phi_tab[mm,int(k[mm,cc,j]),numiter]
                    
    return(k,nk,xi_star,sigma_star,phi_tab,sigma_tab)
    
def sample_phitab(n,d,mm,numiter,c,nc,k,nk,sign,eps,alpha_eps,k_eps,beta_eps,\
                  sigma_tab,phi_tab,sigma_star,xi_star,sigma_eps,xi_eps):
    
    for kk in np.where(nk[mm,:,numiter]>0)[0]:
        
        n_n = np.sum(nc[mm,k[mm,:,:]==kk])
        sume = 0
        sume2 = 0
        for j in range(d):
            for cc in np.where(k[mm,:,j]==kk)[0]:
                sume = sume + np.sum(sign[mm,c[mm,:,j] == cc,j] *\
                eps[mm,c[mm,:,j] == cc,j])
                sume2 = sume2 + np.sum(eps[mm,c[mm,:,j] == cc,j]**2)
                   
        alpha_n = alpha_eps[mm] + n_n/2
        k_n = k_eps[mm] + n_n
        beta_n = beta_eps[mm] + \
                +1/2*(sume2 - n_n*(sume/n_n)**2)+\
            (k_eps[mm] * sume**2/n_n)/ (2*(k_n))   
        
        sigma_tab[mm,kk,numiter] = \
                    1 / np.random.gamma( alpha_n, 1/beta_n ) 
        phi_tab[mm,kk,numiter]  = \
                    np.random.normal(sume / k_n,\
                    (sigma_tab[mm,kk,numiter] \
                       / (k_n))**(1/2))
                    
        for j in range(d):
            for cc in np.where(k[mm,:,j]==kk)[0]:
                sigma_star[mm, cc, j ] =  \
                        sigma_tab[mm,kk,numiter] 
                xi_star[mm, cc, j ] =  \
                        phi_tab[mm,kk,numiter]
                        
                xi_eps[mm,c[mm,:,j] == cc,j,numiter] = \
                sign[mm,c[mm,:,j] == cc,j]*phi_tab[mm,kk,numiter]
                sigma_eps[mm,c[mm,:,j] == cc,j,numiter] = \
                sigma_tab[mm,kk,numiter]

    return(sigma_tab,phi_tab,sigma_star,xi_star,sigma_eps,xi_eps)    


def sample_t(n,d,mm,numiter,t,nt,z,theta_tab,theta,sigma_eps,s_th,mu,omega):
    for j in range(d):
        #remove pop j (theta_j stands up):
        nt[mm,t[mm,j]] = nt[mm,t[mm,j]] - 1
        t[mm,j] = -99999 #sort of nan for int

        #compute full conditional of tj:
        p = np.zeros(d+1)
        for tt in np.where(nt[mm,:]>0)[0]: #for each table
            p[tt] = nt[mm,tt] *np.prod(norm.pdf( z[mm,0:n[j],j], \
                theta_tab[mm,tt], sigma_eps[mm,0:n[j],j,numiter]**(1/2))) 

        sumz = np.nansum(z[mm,:,j] / sigma_eps[mm,:,j,numiter])
        sumz2 = np.nansum(z[mm,:,j]**2 / sigma_eps[mm,:,j,numiter])
        
        
        s_new = (np.nansum( sigma_eps[mm,:,j,numiter]**(-1)) +\
                         s_th[mm]**(-1))**(1/2)
        s_prod = np.nanprod(sigma_eps[mm,:,j,numiter]**(1/2)) *\
            (s_th[mm]**(1/2))

        p[d] = omega[numiter]*1 / ( (2*math.pi)**(n[j]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)

        p = p/sum(p)
        
        t_sampled = np.random.choice(range(len(p)),p=p)
        if t_sampled<d:
            t[mm,j] = t_sampled
            nt[mm,t_sampled] = nt[mm,t_sampled] + 1
            theta[mm,j, numiter] = theta_tab[mm,t_sampled]
        else:
            #look for an empty table 
 
            t[mm,j] = np.setdiff1d(range(d),t[mm,t[mm,:]!=-99999])[0]

            nt[mm,t[mm,j]] = 1
            #sample new value of theta sampling k first 
            varpost = 1 / (np.nansum( sigma_eps[mm,:,j,numiter]**(-1)) +\
                             s_th[mm]**(-1))
            mupost = (sumz + \
                      mu[mm]/s_th[mm])\
                      * varpost
            theta_tab[mm,t[mm,j]] = np.random.normal( mupost, varpost**(1/2) )    
            theta[mm,j, numiter] = theta_tab[mm,t[mm,j]]   
    return(t,nt,theta,theta_tab)

def sample_t_ord(n,d,mm,numiter,t,nt,z,theta_tab,theta,sigma_eps,s_th,mu,omega):
    #j=0 t[mm, 0]=0
    #j=1 t[mm, 1]=...
    p = np.zeros(2)
    p[0] = omega[numiter]**2 + 3 * omega[numiter] + 6
    p[1] = omega[numiter]**3 + 2 * omega[numiter]**2 + 2*omega[numiter]
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[1],1], \
                theta[mm, 0, numiter], sigma_eps[mm,0:n[1],1,numiter]**(1/2)))
        
    sumz = np.nansum(z[mm,:,1] / sigma_eps[mm,:,1,numiter])
    sumz2 = np.nansum(z[mm,:,1]**2 / sigma_eps[mm,:,1,numiter])
    
    
    s_new = (np.nansum( sigma_eps[mm,:,1,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,1,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[1]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
        
    t[mm, 1] = np.random.choice(range(len(p)),p=p)
    
    if t [mm, 1] == 1:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,1,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 1, numiter] =  np.random.normal( mupost, varpost**(1/2) )
        #j=2 t[mm, 2]=...
        p = np.zeros(2)
        p[0] = omega[numiter] + 2
        p[1] = omega[numiter]**2 + omega[numiter] 
    else: 
        theta[mm, 1, numiter] = theta[mm, 0, numiter]
        #else -> t [mm, 1] == 0
        #j=2 t[mm, 2]=...
        p = np.zeros(2)
        p[0] = 2 * omega[numiter] + 6
        p[1] = omega[numiter]**2 + omega[numiter] 
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[2],2], \
                theta[mm, 1, numiter], sigma_eps[mm,0:n[2],2,numiter]**(1/2)))
        
    sumz = np.nansum(z[mm,:,2] / sigma_eps[mm,:,2,numiter])
    sumz2 = np.nansum(z[mm,:,2]**2 / sigma_eps[mm,:,2,numiter])
    
    s_new = (np.nansum( sigma_eps[mm,:,2,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,2,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[2]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
        
    t[mm, 2] = t[mm, 1] + np.random.choice(range(len(p)),p=p)
    if t[mm, 2] != t[mm, 1]:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,2,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 2, numiter] =  np.random.normal( mupost, varpost**(1/2) )
        #j=3 t[mm, 3]=...
        p = np.zeros(2)
        p[0] = 1
        p[1] = omega[numiter] 
    else: 
        theta[mm, 2, numiter] = theta[mm, 1, numiter]
        if theta[mm, 1, numiter] == theta[mm, 0, numiter]:
            p = np.zeros(2)
            p[0] = 3
            p[1] = omega[numiter]
        else:
            p = np.zeros(2)
            p[0] = 2
            p[1] = omega[numiter]
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[3],3], \
                theta[mm, 2, numiter], sigma_eps[mm,0:n[3],3,numiter]**(1/2)))
    sumz = np.nansum(z[mm,:,3] / sigma_eps[mm,:,3,numiter])
    sumz2 = np.nansum(z[mm,:,3]**2 / sigma_eps[mm,:,3,numiter])
    
    s_new = (np.nansum( sigma_eps[mm,:,3,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,3,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[3]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
    t[mm, 3] = t[mm, 2] + np.random.choice(range(len(p)),p=p)
    if t[mm, 3] != t[mm, 2]:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,3,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 3, numiter] =  np.random.normal( mupost, varpost**(1/2) )
    else: 
        theta[mm, 3, numiter] = theta[mm, 2, numiter]
    #compute nt
    nt[mm,:] = 0 
    for tt in range(max(t[mm,:])+1):
        nt[mm,tt] = np.sum(t[mm,:]==tt)
        theta_tab[mm,tt] = theta[mm,np.where(t[mm]==tt)[0][0],0]
            
    return(t,nt,theta,theta_tab)  

def sample_t_unif(n,d,mm,numiter,t,nt,z,theta_tab,theta,sigma_eps,s_th,mu,omega):
    #j=0 t[mm, 0]=0
    #j=1 t[mm, 1]=...
    p = np.zeros(2)
    p[0] = 1
    p[1] = 1
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[1],1], \
                theta[mm, 0, numiter], sigma_eps[mm,0:n[1],1,numiter]**(1/2)))
        
    sumz = np.nansum(z[mm,:,1] / sigma_eps[mm,:,1,numiter])
    sumz2 = np.nansum(z[mm,:,1]**2 / sigma_eps[mm,:,1,numiter])
    
    
    s_new = (np.nansum( sigma_eps[mm,:,1,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,1,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[1]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
        
    t[mm, 1] = np.random.choice(range(len(p)),p=p)
    
    if t [mm, 1] == 1:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,1,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 1, numiter] =  np.random.normal( mupost, varpost**(1/2) )
    else: 
        theta[mm, 1, numiter] = theta[mm, 0, numiter]
        #else -> t [mm, 1] == 0
    #j=2 t[mm, 2]=...
    p = np.zeros(2)
    p[0] = 1
    p[1] = 1 
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[2],2], \
                theta[mm, 1, numiter], sigma_eps[mm,0:n[2],2,numiter]**(1/2)))
        
    sumz = np.nansum(z[mm,:,2] / sigma_eps[mm,:,2,numiter])
    sumz2 = np.nansum(z[mm,:,2]**2 / sigma_eps[mm,:,2,numiter])
    
    s_new = (np.nansum( sigma_eps[mm,:,2,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,2,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[2]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
        
    t[mm, 2] = t[mm, 1] + np.random.choice(range(len(p)),p=p)
    if t[mm, 2] != t[mm, 1]:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,2,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 2, numiter] =  np.random.normal( mupost, varpost**(1/2) )
    else: 
        theta[mm, 2, numiter] = theta[mm, 1, numiter]
        
    p = np.ones(2)
    p[0] = p[0] * np.prod(norm.pdf( z[mm,0:n[3],3], \
                theta[mm, 2, numiter], sigma_eps[mm,0:n[3],3,numiter]**(1/2)))
    sumz = np.nansum(z[mm,:,3] / sigma_eps[mm,:,3,numiter])
    sumz2 = np.nansum(z[mm,:,3]**2 / sigma_eps[mm,:,3,numiter])
    
    s_new = (np.nansum( sigma_eps[mm,:,3,numiter]**(-1)) +\
                     s_th[mm]**(-1))**(1/2)
    s_prod = np.nanprod(sigma_eps[mm,:,3,numiter]**(1/2)) *\
        (s_th[mm]**(1/2))
    p[1] = p[1] * 1 / ( (2*math.pi)**(n[3]/2) * s_new * s_prod )*\
            math.exp(-1/2 *(sumz2 + (mu[mm]**2/s_th[mm]))+\
         1/2 *(sumz  + (mu[mm]/s_th[mm]))**2 /\
                 s_new**2)
    p = p/sum(p)
    t[mm, 3] = t[mm, 2] + np.random.choice(range(len(p)),p=p)
    if t[mm, 3] != t[mm, 2]:
        varpost = 1 / (np.nansum( sigma_eps[mm,:,3,numiter]**(-1)) +\
                             s_th[mm]**(-1))
        mupost = (sumz + mu[mm] / s_th[mm]) * varpost   
        theta[mm, 3, numiter] =  np.random.normal( mupost, varpost**(1/2) )
    else: 
        theta[mm, 3, numiter] = theta[mm, 2, numiter]
    #compute nt
    nt[mm,:] = 0 
    for tt in range(max(t[mm,:])+1):
        nt[mm,tt] = np.sum(t[mm,:]==tt)
        theta_tab[mm,tt] = theta[mm,np.where(t[mm]==tt)[0][0],0]
            
    return(t,nt,theta,theta_tab)
    
    
def sample_thetatab(n,d,mm,numiter,t,nt,z,sigma_eps,s_th,mu,theta_tab,theta):
    for tt in np.where(nt[mm,:]>0)[0]:

        sumz = np.nansum(z[mm,:,t[mm,:]==tt] / \
                         sigma_eps[mm,:,t[mm,:]==tt,numiter])
        s_temp =  np.nansum( sigma_eps[mm,:,t[mm,:]==tt,numiter]**(-1))
            
        varpost = 1 / (s_temp +1/s_th[mm])
        mupost = (sumz + mu[mm]/s_th[mm]) * varpost
        theta_tab[mm,tt] = np.random.normal( mupost, varpost**(1/2) )        
            
        theta[mm,t[mm,:]==tt, numiter] = theta_tab[mm,tt]
    return(theta_tab,theta)