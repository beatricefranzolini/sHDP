# -*- coding: utf-8 -*-
"""
Python code to run to get simulation studies summaries (to run after main_simul.py).
"""

import os
from pathlib import Path

import numpy as np
from astropy.table import Table

#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)
############################POPULATION LEVEL##################################
#Get Table 1
#our model
theta_final = np.loadtxt('aggregatetheta.txt')
#nested
#theta_final = pd.read_csv('aggregatetheta_nested_s1.txt', sep=';', header = None)

n_simstud = theta_final.shape[0]

#theta_final = theta_final[:,0:15]

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
# map counts
MAPcounts = np.zeros(15) * np.nan
for i in range(15): 
    MAPcounts[i] = np.count_nonzero( \
                theta_final[:,i] == np.max(theta_final, axis = 1) )
#average post prob
avpostp = np.mean(theta_final, axis = 0)
#median post prob
mepostp = np.median(theta_final, axis = 0)
tsHDP = Table([models, MAPcounts, np.round(avpostp, 3), np.round(mepostp, 3)])
print(tsHDP)





