#This file creates risk plots over time using the evaluated risk, checkin events and infection events per cluster.
# Sirisha Rambhatla, September 2020

import pyarrow
import gc
import pyarrow.parquet as pq
import pandas as pd
import pickle

import numpy as np
import seaborn as sns
#import geopandas as gpd
import matplotlib.pyplot as plt
#from geopy.distance import great_circle, distance
from sklearn.cluster import KMeans
from scipy import stats
from tqdm import tqdm

import random

#import folium
import json
import os
#from folium import plugins
#from folium.plugins import HeatMap

import datetime
from pytz import timezone
from argparse import ArgumentParser
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--city', type=str, default='LA')
parser.add_argument('--clusters', type=int, default=15)
parser.add_argument('--resroot', type=str, default='<add default path>')

# python3 create_risk_inf_plots.py --city LA --clusters 15 

args = parser.parse_args()
city = args.city
month = 'dec'
NUM_COLORS = args.clusters
data_folder = args.resroot #+ month+'_'+city+ '_' + str(NUM_COLORS) + '/'

folder = data_folder + 'dec' + '_' + city + '_' + str(NUM_COLORS) + '/'

input_mobility_dec = glob(data_folder + 'dec' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_checkin_events_*.csv')[0]
input_infections_dec = glob(data_folder + 'dec' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_inf_events_*.csv')[0]
input_risk_dec = glob(data_folder + 'dec' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'risk.csv')[0]

input_mobility_jan = glob(data_folder + 'jan' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_checkin_events_*.csv')[0]
input_infections_jan = glob(data_folder + 'jan' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_inf_events_*.csv')[0]
input_risk_jan = glob(data_folder + 'jan' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'risk.csv')[0]

input_mobility_mar = glob(data_folder + 'mar' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_checkin_events_*.csv')[0]
input_infections_mar = glob(data_folder + 'mar' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'cluster_inf_events_*.csv')[0]
input_risk_mar = glob(data_folder + 'mar' + '_' + city + '_' + str(NUM_COLORS) + '/'+ 'risk.csv')[0]

infections_dec = pd.read_csv(input_infections_dec)
mobility_dec = pd.read_csv(input_mobility_dec)
risk_dec = pd.read_csv(input_risk_dec, header=None)

infections_jan = pd.read_csv(input_infections_jan)
mobility_jan = pd.read_csv(input_mobility_jan)
risk_jan = pd.read_csv(input_risk_jan, header=None)

infections_mar = pd.read_csv(input_infections_mar)
mobility_mar = pd.read_csv(input_mobility_mar)
risk_mar = pd.read_csv(input_risk_mar, header=None)


### infections
fig,ax = plt.subplots(1)

days_dec = np.arange(1,32)
days_jan = np.arange(1,32)
days_mar = np.arange(1,infections_mar.shape[0]+1)

# Average
plt.plot(days_dec, infections_dec.values[:,1:].mean(axis=1), linestyle='-', lw=4, color='red')
plt.plot(days_jan, infections_jan.values[:,1:].mean(axis=1), linestyle='-.' , lw=4, color='blue')
plt.plot(days_mar, infections_mar.values[:,1:].mean(axis=1), linestyle=':' , lw=4,  color='black')

# Specific
#plt.plot(days_dec, infections_dec[str(CLUSTER_NUM)], linestyle='-', lw=4, color='red')
#plt.plot(days_jan, infections_jan[str(CLUSTER_NUM)], linestyle='-.' , lw=4, color='blue')
#plt.plot(days_mar, infections_mar[str(CLUSTER_NUM)], linestyle=':' , lw=4,  color='black')

plt.legend(['Dec', 'Jan', 'Mar'], fontsize=14)

plt.xlabel('Days', size = 20)
plt.ylabel('Avg. Infections', size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)



plt.locator_params(axis="x", nbins=10)
plt.locator_params(axis="y", nbins=9)
plt.grid()

ax.autoscale(enable=True, axis='both', tight=True)
plt.savefig(folder + "avg_inf.png", bbox_inches='tight')
#ax.autoscale_view()


### Risk
fig,ax = plt.subplots(1)

# Specific
#plt.plot(days_dec, risk_dec.loc[CLUSTER_NUM], linestyle='-', lw=4, color='red')
#plt.plot(days_jan, risk_jan.loc[CLUSTER_NUM], linestyle='-.' , lw=4, color='blue')
#plt.plot(days_mar, risk_mar.loc[CLUSTER_NUM], linestyle=':' , lw=4,  color='black')

# Average
plt.plot(days_dec, risk_dec.mean(), linestyle='-', lw=4, color='red')
plt.plot(days_jan, risk_jan.mean(), linestyle='-.' , lw=4, color='blue')
plt.plot(days_mar, risk_mar.mean(), linestyle=':' , lw=4,  color='black')

plt.legend(['Dec', 'Jan', 'Mar'], fontsize=14)


plt.xlabel('Days', size = 20)
plt.ylabel('Avg. Risk', size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)



plt.locator_params(axis="x", nbins=10)
plt.locator_params(axis="y", nbins=9)
plt.grid()

ax.autoscale(enable=True, axis='both', tight=True)
plt.savefig(folder + "avg_risk.png", bbox_inches='tight')

### Mobility

fig,ax = plt.subplots(1)


# Specific
# total_mob_dec = mobility_dec[mobility_dec['Unnamed: 0']==CLUSTER_NUM].iloc[0].values[2:] #+ \
#            #     mobility_dec[mobility_dec['Unnamed: 0']==CLUSTER_NUM].iloc[1].values[2:] + \
#            #     mobility_dec[mobility_dec['Unnamed: 0']==CLUSTER_NUM].iloc[2].values[2:] 

# total_mob_jan = mobility_jan[mobility_jan['Unnamed: 0']==CLUSTER_NUM].iloc[0].values[2:] #+ \
#                # mobility_jan[mobility_jan['Unnamed: 0']==CLUSTER_NUM].iloc[1].values[2:] + \
#                # mobility_jan[mobility_jan['Unnamed: 0']==CLUSTER_NUM].iloc[2].values[2:] 

# total_mob_mar = mobility_mar[mobility_mar['Unnamed: 0']==CLUSTER_NUM].iloc[0].values[2:] #+ \
#               #  mobility_mar[mobility_mar['Unnamed: 0']==CLUSTER_NUM].iloc[1].values[2:] + \
#               #  mobility_mar[mobility_mar['Unnamed: 0']==CLUSTER_NUM].iloc[2].values[2:] 

# plt.plot(days_dec, total_mob_dec, linestyle='-', lw=4, color='red')
# plt.plot(days_jan, total_mob_jan, linestyle='-.' , lw=4, color='blue')
# plt.plot(days_mar, total_mob_mar, linestyle=':' , lw=4,  color='black')


# # Average
plt.plot(mobility_dec[mobility_dec['Mobility type']=='self_traffic'].mean().values[1:], linestyle='-', lw=4, color='red')
plt.plot(mobility_jan[mobility_jan['Mobility type']=='self_traffic'].mean().values[1:], linestyle='-.' , lw=4, color='blue')
plt.plot(mobility_mar[mobility_mar['Mobility type']=='self_traffic'].mean().values[1:], linestyle=':' , lw=4,  color='black')

plt.legend(['Dec', 'Jan', 'Mar'], fontsize=14, )


plt.xlabel('Days', size = 20)
plt.ylabel('Avg. Mobility Density', size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)



plt.locator_params(axis="x", nbins=10)
plt.locator_params(axis="y", nbins=9)
plt.grid()

ax.autoscale(enable=True, axis='both', tight=True)
plt.savefig(folder + "avg_mob.png", bbox_inches='tight')