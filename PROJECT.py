#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import obspy as obs
from obspy.clients.fdsn.client import Client
from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
from scipy.signal import butter, filtfilt
import matplotlib.pylab as pl
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import fastparquet
import scipy.signal as sp
import functions 
from obspy.taup import plot_travel_times,TauPyModel
from obspy.taup.tau import plot_ray_paths
from obspy.clients.iris import Client as Client_dist
import pandas as pd
import os
from obspy import read
import obspy.signal as sig

client = Client('IRIS')
eventtime = UTCDateTime('2010-02-27 06:34:11')

eq_list = client.get_events(
    minmagnitude=8.5,
    starttime=eventtime-10,
    endtime=eventtime+10)


for event in eq_list.events:
    for origin in event.origins:
        eq_time = origin.time
        eq_depth = origin.depth
        eq_lat = origin.latitude
        eq_lon  = origin.longitude
        
        
for event in eq_list.events:
    for magnitude in event.magnitudes:
        eq_mag = magnitude.mag


#RECHERCHE DE TRACES BROADBAND VENANT DE TOUS LES ARRAYS  AUTOUR DE L'EQ !!  CETTE CELLULE SERT À LES RÉCUP, PAS ENCORE À PLOT!
compute_distance = Client_dist() #on utilise un outil qui calculera la distance angulaire qui est  nécessaire pour obspy pour calculer le travel time 
# plt.close('all')

# #parameters
pd.read_csv('stations_TA.csv')


# ## Now we know the stations, so we download their traces

# In[9]:

data = pd.read_csv('stations_TA.csv')
network_list_clean = list(data['Network'].values)
station_list_clean = list(data['Station'].values)
latitudes_list_clean = list(data['Latitude'].values)
longitudes_list_clean = list(data['Longitude'].values)
distances_list_clean = list(data['Distance'].values)
azimuth_list_clean  = list(data['Azimuth'].values)
channels = 'BHZ,HHZ,EHZ,SHZ'

################### TÉLÉCHARGEMENT DES TRACES ET ON LES RASSEMBLE DANS DES LISTES POUR POUVOIR TOUT PLOT AVEC LES COULEURS À LA FIN WAOUW 
start_delay = -30
duration = 1800
pad = 20 #télécharger pad secondes avant et après la trace pour s'assurer que elle soit pas affectée par effet de bord 
output_type = 'VEL'
starttime_obspy = eq_time+start_delay-pad 
endtime_obspy = eq_time+start_delay+duration+pad

bad_station_indexes = []

for i in tqdm(range(len(station_list_clean))):
    if os.path.exists(f'/home/parisnic/traces/{station_list_clean[i]}.mseed')==False: 
        try:
            st = client.get_waveforms(network=network_list_clean[i], station=station_list_clean[i], location='*', channel=channels, attach_response=True,
                                              starttime=starttime_obspy,endtime=endtime_obspy)
            st.remove_response(output=output_type)
            st.detrend('demean')            
            tr = st[0]
            tr.write(f"/home/parisnic/traces/{station_list_clean[i]}.mseed", format="MSEED")              
        except:
            bad_station_indexes.append(i)
    
    
try:
    for bad_index in bad_station_indexes:
        print(f'{network_list_clean[bad_index]}.{station_list_clean[bad_index]} could not be downloaded')
except:
    print('All stations were downloaded correctly! yey')
                
for bad_index in sorted(bad_station_indexes, reverse=True):#c'est l'heure de supprimer les mauvaises stations de la liste 
    network_list_clean.pop(bad_index) #on supprime la station de la liste pour avoir une liste mise à jour!
    station_list_clean.pop(bad_index)
    latitudes_list_clean.pop(bad_index)
    longitudes_list_clean.pop(bad_index)
    distances_list_clean.pop(bad_index)
    azimuth_list_clean.pop(bad_index)



#on fait à présent un tableau avec les traces, on resample les traces si besoin et on les filtres si l'on souhaite
time_list = []
arr_list = []
color_list = []
fs_list_clean = []

##### parameters to resmaple####
fs = 40 #frequency at which we will resample all the traces 

##### parameters to remove the padding and  we can also change the parameters to only keep a specific part of the traces ######
start_delay = 500 
duration = 1200

filtered = 'bandpass'
freq = [1,5]
order = 2 #order du filtre  -> filtfilt donc sera doublé!

for i in range(len(station_list_clean)): #on a supprimé les mauvaises stations et téléchargé les traces, avec le minimum de processing possible pour les garder en 
    #bon état
    st = read(f'/home/parisnic/traces/{station_list_clean[i]}.mseed')

    if st[0].stats.sampling_rate>fs: #on resample si trop hf 
                st.resample(fs, window='hann', no_filter=True, strict_length=False)

    tr = st[0]
    fs_list_clean.append(tr.stats.sampling_rate)
            
    if filtered!=None:
        st.filter(filtered,freqmin=freq[0],freqmax=freq[1],corners=order,zerophase=True) #fs_list_clean[i],
                    
    arr = tr.data
    time_arr = (tr.times(reftime=eq_time)) #on doit à présent chercher dans time_arr où se trouve     
    indexes_to_keep = np.where((time_arr>=start_delay) & (time_arr<=start_delay+duration))
    time_arr = time_arr[indexes_to_keep]
    arr = arr[indexes_to_keep]
    arr_list.append(list(arr))
    time_list.append(list(time_arr))
    color_list.append(int(azimuth_list_clean[i]))


#it's resampled but sometimes there is an additionnal sample at some stations which is troublesome, so we remove this additionnal sample at the beginning! 

min_nt = 99999999999999999999
for time in time_list:
    if len(time) < min_nt:
        min_nt = len(time)


arr_list_good = []
time_list_good = []

for i in range(len(time_list)):
    if len(time_list[i]) != min_nt:
        time_list_good.append(time_list[i][1:])
        arr_list_good.append(arr_list[i][1:])
    else:
        time_list_good.append(time_list[i])
        arr_list_good.append(arr_list[i])


# Create a grid of potential sources : we use lat lon coordinates intead of cartesian coordinates for the gridsearch to make it easier
x    = np.linspace(-74,-70, 50)
y    = np.linspace(-38, -33, 50)
x, y = np.meshgrid(x, y)

# Depth of the source (assumed to be known)
z = eq_depth/1000 #converted to km 

# Initialize the RMS for each potential source
rms = np.zeros((x.shape[0], x.shape[1]))

# Initialize the stacks for each potential source
obs = np.array(arr_list_good) #normalement devrait avoir taille  nr, nt 
nt = len(obs[0,:]) #number of samples : should be the same for all traces since we decimated them
stacks = np.zeros((nt, x.shape[0], x.shape[1]))

#on utilise le ray tracing 1D d'obspy pour estimerle travel time entre la source et chacune des stations 
model = TauPyModel(model='iasp91')


#just need to compute the time with 1D raytracing with obspy ()
for i in tqdm(range(x.shape[0]),leave=False): #looping over potential sources  
    for j in range(x.shape[1]):
        dist = np.zeros(len(distances_list_clean))
        ttime = np.zeros_like(dist)
        for k in range(len(longitudes_list_clean)): # looping over receivers and computing their distance to the potnetial source location which is important to  shift the traces accordingly 
            dist_km = gps2dist_azimuth(latitudes_list_clean[k],longitudes_list_clean[k],y[i,j],x[i,j])[0]/1000
            dist[k] = kilometers2degrees(dist_km) #  computing the distance between a potential source and the receivers 
            ttime[k] = model.get_ray_paths(source_depth_in_km=eq_depth/1000, distance_in_degree=dist[k], phase_list=['P'])[0].time
        obs_shifted = functions.shift(obs, ttime, fs_list_clean)
        stacks[:,i,j] = np.sum(obs_shifted,axis=0)
        rms[i,j]  = np.sqrt(np.sum((stacks[:,i,j])**2)) 

np.save('rms.npy',rms)
np.save('stacks.npy',stacks)