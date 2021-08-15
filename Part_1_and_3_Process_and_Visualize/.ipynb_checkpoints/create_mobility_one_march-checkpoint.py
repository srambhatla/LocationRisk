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

parser = ArgumentParser()
parser.add_argument('--month', type=str, default='dec')
parser.add_argument('--city', type=str, default='LA')
parser.add_argument('--clusters', type=int, default=15)
parser.add_argument('--infect_stat_folder', type=str, default='/tank/users/sirishar/analyze/data/infect_stats/')
parser.add_argument('--latent_days', type=int, default=5)
#parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resroot', type=str, default='/home/users/sirishar/analyze/data/')


args = parser.parse_args()




# python3 compute_OD_matrix.py /
# --month 'dec' /
# --city 'LA' /
# --clusters 15 /
# --infect_stat_file '/tank/users/sirishar/SpreadSimulation-v4-master/tests/test_1036800_432000_3600_0.0001_1_1000_1_0_0.1/infect_stat_res.txt' /
# --seed 0 /
# --resroot '/home/users/sirishar/analyze/data/'

# python3 create_mobility.py --month 'dec' --city 'LA' --clusters 15 --infect_stat_file '/tank/users/sirishar/SpreadSimulation-v4-master/tests/test_1036800_432000_3600_0.0001_1_1000_1_0_0.1/infect_stat_res.txt' --resroot '/home/users/sirishar/analyze/data/'


# Set args
#input_filename = '/tank/users/sirishar/SpreadSimulation-v4-master/tests/test_1036800_432000_3600_0.0001_1_1000_1_0_0.1/infect_stat_res.txt'
month = args.month
city = args.city
NUM_COLORS = args.clusters
input_filename = args.infect_stat_folder + city + '_' + month + '_infect_stat_res.txt'
data_folder = args.resroot + month + '_' + city + '_' + str(NUM_COLORS)+ '/'
latent_days = args.latent_days
time_step = 24*60*60

os.mkdir(data_folder)


print('Loading data')
#month = 'dec'
#city = 'LA'
if month != 'mar':
    df = pq.read_table("/secure/dataset/derived/sorted_and_filtered_"+month+"_"+city+".parquet").to_pandas() 

    min_time = int(np.floor(min(df.timestamp)/time_step)*time_step)
    max_time = int(np.ceil(max(df.timestamp)/time_step)*time_step)
else:
    df = pq.read_table("/secure/dataset/derived/sorted_and_filtered_march_"+city+".parquet").to_pandas() 

    min_time = int(np.floor(min(df.timestamp)/time_step)*time_step)
    max_time = int(np.ceil(max(df.timestamp)/time_step)*time_step)
#else:
#     # we call the week 0 data as df to keep rest of things simple!
#     df = pq.read_table("/secure/dataset/derived/sorted_and_filtered_"+month+'0'+"_"+city+".parquet").to_pandas() 
#     min_time = int(np.floor(min(df.timestamp)/time_step)*time_step)
#     df_w1 = pq.read_table("/secure/dataset/derived/sorted_and_filtered_"+month+'1'+"_"+city+".parquet").to_pandas() 
#     df_w2 = pq.read_table("/secure/dataset/derived/sorted_and_filtered_"+month+'2'+"_"+city+".parquet").to_pandas() 
#     max_time = int(np.ceil(max(df_w2.timestamp)/time_step)*time_step)
# #df = pq.read_table("/tank/users/sirishar/data/sorted_and_filtered_"+month+"_"+city+".parquet").to_pandas() 


NUM_DAYS = int((max_time - min_time)/time_step)
time = np.arange(min_time, max_time, time_step)

print('kmeans..')
if month == 'dec':
    print('It is the month of December so Ill run KMeans')
    # Load data pts
    data_pts = list(zip(df['lat'].values[0:100000], df['lon'].values[0:100000]))

    # fit kmeans
    kmeans = KMeans(n_clusters=NUM_COLORS, random_state=0).fit(data_pts) 
    
    pickle.dump(kmeans, open(data_folder + "kmeans_" + city + ".pkl", "wb"))
else:
    print('Loading kmeans from pickle file (december data)')
    kmeans = pickle.load(open(args.resroot + 'dec' + '_' + city + '_' + str(NUM_COLORS)+ '/' + "kmeans_" + city + ".pkl", "rb"))

# Cluster Centers
np.save(data_folder + 'cluster_centers_'+ month + '_' + city + '_' + str(NUM_COLORS), kmeans.cluster_centers_)

np.random.seed(0)

print('Identify users')
sample_prop = 0.1

if sample_prop < 1:
#    if month != 'mar':
    usrs = df['id'].unique()
    usrs = np.random.choice(usrs, size=int(usrs.shape[0]*sample_prop))
    df = df.loc[df['id'].isin(usrs)]
    gc.collect()
#     else:
#         # week 0 
#         usrs = df['id'].unique()
#         usrs = np.random.choice(usrs, size=int(usrs.shape[0]*sample_prop))
#         df = df.loc[df['id'].isin(usrs)]
#         gc.collect()
        
#         #week 1
#         usrs = df_w1['id'].unique()
#         usrs = np.random.choice(usrs, size=int(usrs.shape[0]*sample_prop))
#         df_w1 = df_w1.loc[df_w1['id'].isin(usrs)]
#         gc.collect()
        
#         # week 2
#         usrs = df_w2['id'].unique()
#         usrs = np.random.choice(usrs, size=int(usrs.shape[0]*sample_prop))
#         df_w2 = df_w2.loc[df_w2['id'].isin(usrs)]
#         gc.collect()
 
    
# Use first 100k for kmeans
#data_pts = list(zip(df['lat'].values[0:100000], df['lon'].values[0:100000])) 



tqdm.pandas()

# function to evaluate cluster membership given a dataframe
def calc_cluster(df):
    if df.shape[0] > 1:
        return kmeans.predict(list(zip(df['lat'].values, df['lon'].values)))
    else:
        return kmeans.predict([(df['lat'].values[0], df['lon'].values[0])])


print('Form OD matrix')
#begin_time = df['timestamp'].min()
OD = np.zeros(shape=(NUM_DAYS, NUM_COLORS, NUM_COLORS))

def add_od(trajectory):
    clusters = calc_cluster(trajectory)
    indx = np.array(((trajectory['timestamp']-min_time)//(time_step))[:-1]), clusters[:-1], clusters[1:]
    OD[indx]+=1
    return 0

print('calculate OD')
#if month != 'mar': 
df.groupby('id').progress_apply(add_od)
#else:
#    df.groupby('id').progress_apply(add_od)
#    df_w1.groupby('id').progress_apply(add_od)
#    df_w2.groupby('id').progress_apply(add_od)

#print(OD)
np.save(data_folder + 'OD_matrix_'+ month + '_' + city + '_' + str(NUM_COLORS), OD)


def get_mob_inf_clusters(day_start, day_end):
    mob_inf_day = []
    OD_sel = OD[day_start:day_end, :, :]
    inf_record_sel = np.squeeze(np.array(inf_records.copy())).T[:,day_start:day_end]
    non_self_tr_days = OD_sel - [np.diag(np.diag(X)) for X in OD_sel]
    
    for c in range(0, NUM_COLORS):
        mob_to_c = []
        temp = non_self_tr_days.copy()
            
        for ii in range(0, non_self_tr_days.shape[0]):
            temp[ii, c,:] = 0
    
            mob_to_c.append(sum(np.squeeze(temp[ii, :,c])))
            temp[ii, :,c] = 0


        sum_inf = [sum(sum(X)) for X in temp.copy()]
        infections_minus_c = inf_record_sel.copy()
        infections_minus_c[c,:] = 0 
        infections_minus_c = sum(infections_minus_c)


        mob_inf_day.append(sum([(x/max(y,1))*z for x,y,z in zip(mob_to_c, sum_inf, infections_minus_c)]))
    return mob_inf_day

def get_mob_inf_clusters_with_self(day_start, day_end):
    mob_inf_day = []
    self_mob_inf_day = []
    OD_sel = OD[day_start:day_end, :, :]
    inf_record_sel = np.squeeze(np.array(inf_records)).T[:,day_start:day_end]
    non_self_tr_days = OD_sel - [np.diag(np.diag(X)) for X in OD_sel]
    self_tr_days = [np.diag(np.diag(X)) for X in OD_sel]

    temp_s = self_tr_days.copy()
    for c in range(0, NUM_COLORS):
        mob_to_c = []
        mob_self_c = []
        temp_o = non_self_tr_days.copy()
    
        #print(np.linalg.norm(temp))
    
            
        for ii in range(0, non_self_tr_days.shape[0]):
            temp_o[ii, c,:] = 0
    
            mob_to_c.append(sum(np.squeeze(temp_o[ii, :,c])))
            temp_o[ii, :,c] = 0

        #print(np.linalg.norm(temp))
        sum_inf = [sum(sum(X)) for X in temp_o.copy()]
        infections_minus_c = inf_record_sel.copy()    
        infections_minus_c[c,:] = 0 
        infections_minus_c = sum(infections_minus_c)
    
        #Keep on c's infections
        infections_of_c = inf_record_sel[c,:].copy()
    
    
    
        self_mob_inf_day.append(sum([(x/max(y,1))*z for x,y,z in zip(np.array(temp_s)[:,c, c].copy(), mob_to_c,infections_of_c ) ]))
        mob_inf_day.append(sum([(x/max(y,1))*z for x,y,z in zip(mob_to_c, sum_inf, infections_minus_c)]))
    
        #self_mob_inf_day.append(sum([(x/max(y,1)) for x,y,z in zip(np.array(temp_s)[:,c, c].copy(), mob_to_c,infections_of_c ) ]))
        #mob_inf_day.append(sum([(x/max(y,1)) for x,y,z in zip(mob_to_c, sum_inf, infections_minus_c)]))
    
    return self_mob_inf_day, mob_inf_day
print('Loading infections from ' + input_filename)
# Infections
chunk = pd.read_csv(input_filename, names=['time', 'infected_id', 'infected_by_id', 'lon', 'lat' ], delim_whitespace=True)

inf_records = []

i = 0

#mobility_type = ['checkins', 'self_traffic', 'to_traffic', 'from_traffic']*NUM_COLORS
mobility_type = ['self_traffic', 'to_traffic', 'from_traffic', 'inf_mob_from_other', 'inf_mob_self']*NUM_COLORS
#mobility_type = ['self_traffic', 'to_traffic', 'from_traffic']*NUM_COLORS
#mobility_type = ['checkins']*NUM_COLORS

locs = list(np.repeat(np.arange(0, NUM_COLORS), len(mobility_type)/NUM_COLORS))

df_event_rec = pd.DataFrame(index=locs)
df_event_rec['Mobility type'] = mobility_type


print('Process all indices')
for t in time:
    event_day = []
    
    day_start = int(np.floor(t/time_step)*time_step)
    day_end =  int(np.floor(t/time_step)*time_step) + time_step
    
    # For infection events
    day_events_inf = chunk[(chunk['time'] >= day_start) & (chunk['time'] < day_end)]
    
    
    day_inf_events_by_clus = np.zeros([1,NUM_COLORS])
    
    if not day_events_inf.empty:
        clus_inf_day = calc_cluster(day_events_inf)
    
        for clus in np.unique(clus_inf_day):
            day_inf_events_by_clus[0,clus] = len(np.where(clus_inf_day == clus)[0])
    
    inf_records.append(day_inf_events_by_clus)
    
    
#     #For checkin events
#     day_events = df[(df['timestamp'] >= day_start) & (df['timestamp'] < day_end)]
    
#     day_events_by_clus = np.zeros([1,NUM_COLORS])
    
#     if not day_events.empty:
#         clus_day = calc_cluster(day_events)

#         for clus in np.unique(clus_day):
#             day_events_by_clus[0,clus] = len(np.where(clus_day == clus)[0])
    
#     event_day.append(np.squeeze(day_events_by_clus))
    
                           
    # Load daily OD matrix
    OD_day = np.squeeze(OD[i,:,:])
    self_traffic_day = np.diag(OD_day)
    non_self_traffic = OD_day - np.diag(self_traffic_day)
    
    # Traffic related 
    event_day.append(self_traffic_day)
                           
    #TO traffic
    event_day.append(np.sum(non_self_traffic,0))
                           
    # From Traffic
    event_day.append(np.sum(non_self_traffic,1))
    
    inf_mob = np.zeros(NUM_COLORS)
    self_inf_mob = np.zeros(NUM_COLORS)

    #if i > latent_days:
    if i > 0:
     #day_start = max(0, i-12)
        #day_end = max(0, i-latent_days)
        day_start = max(0,i)
        day_end = max(0,i+1)
        if day_start<day_end:
            #inf_mob = get_mob_inf_clusters(day_start, day_end)
            inf_mob, self_inf_mob = get_mob_inf_clusters_with_self(day_start, day_end)
            
        
    event_day.append(np.array(inf_mob))
    event_day.append(np.array(self_inf_mob))
        
                      
    
    df_event_rec[t] = np.ravel(np.array(event_day), order='F')
    i = i + 1
print('Save csv...')    
# Save infections csv
inf_rec_out_file = data_folder + "cluster_inf_events_" + str(NUM_COLORS)+ str('_') + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) + '.csv'

inf_records = np.array(inf_records)
df_inf_rec = pd.DataFrame(np.squeeze(inf_records), columns=np.arange(0,NUM_COLORS), index=time)   
df_inf_rec.to_csv(inf_rec_out_file)

# Save events csv
event_rec_out_file = data_folder + "cluster_checkin_events_" + str(NUM_COLORS)+ str('_') + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) + '.csv'

# events_records = np.array(events_records)
# df_event_rec = pd.DataFrame([ mobility_type, np.squeeze(events_records)], columns=[ 'type', np.arange(0,NUM_COLORS)], index=mob_t_index)
# df_event_rec.to_csv(event_rec_out_file)

#df_event_rec = pd.DataFrame(events_records, columns=list(np.arange(0,NUM_COLORS)), index=mob_t_index)
#df_event_rec.insert(0, 'Mobility type', mobility_type)
df_event_rec.to_csv(event_rec_out_file)


# # Self traffic csv
# self_traffic_rec_out_file = data_folder + "cluster_self_traffic_" + str(NUM_COLORS)+ str('_') + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) + '.csv'

# self_traffic_records = np.array(self_traffic_records)
# df_self_traffic = pd.DataFrame(np.squeeze(self_traffic_records), columns=np.arange(0,NUM_COLORS), index=time)
# df_self_traffic.to_csv(self_traffic_rec_out_file)


# # to trafficic csv
# to_traffic_rec_out_file = data_folder + "cluster_to_traffic_" + str(NUM_COLORS)+ str('_') + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) + '.csv'

# to_traffic_records = np.array(to_traffic_records)
# df_to_traffic = pd.DataFrame(np.squeeze(to_traffic_records), columns=np.arange(0,NUM_COLORS), index=time)
# df_to_traffic.to_csv(to_traffic_rec_out_file)


# # from traffic csv
# from_traffic_rec_out_file = data_folder + "cluster_from_traffic_" + str(NUM_COLORS)+ str('_') + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) + '.csv'

# from_traffic_records = np.array(from_traffic_records)
# df_from_traffic = pd.DataFrame(np.squeeze(from_traffic_records), columns=np.arange(0,NUM_COLORS), index=time)
# df_from_traffic.to_csv(from_traffic_rec_out_file)

print('Done saving!')
