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
from matplotlib.animation import FuncAnimation



run_folder = 'run_time_cross'
cluster = True #juste pour qu'il sache où chercher les fichiers 

##### DOWNLOADING PARAMETERS ################
start_delay_dl = -30
duration_dl = 1800
pad = 20 #télécharger pad secondes avant et après la trace pour s'assurer que elle soit pas affectée par effet de bord 


##### parameters to remove the padding and  we can also change the parameters to only keep a specific part of the traces ######
start_delay = 500 
duration = 800
delta_static = 30 #delta static permet de réduire le shifting vers la gauche pour pouvoir avoir du temps avant l'eq 
# les traces seront donc shiftées de travel time - start time - delta_static    ça sert en plus d'assurance pour que le shift avec les CC ne fasse par partir le truc en vrille
    
##### parameters to resmaple and to process all the traces ####
fs = 40 #frequency at which we will resample all the traces 
filtered = 'bandpass' #type of fitlered applied
freq = [1,5] #frequencies for filtering 
order = 2 #order du filtre  -> filtfilt donc sera doublé!

#####  GRID OF SOURCES ##### 
x = np.linspace(-74,-70, 20)
y = np.linspace(-38, -31, 20)

##################" WINDOW PARAMETERS ####### 
plage = 10*fs # nombre de points d ela plage  # -> 2400 = 60s    -> doit faire attention à ce que le la plage doit diviseur de la durée du signal (et attention en + avec overlap)
overlap = plage//2 #avoir un overlap de 50% -> pas encore implémenté ... 


###### cross correlation parameters ####
cross_duration = 8 #duraction of the cross correlation 
cross_anticipation = cross_duration//2 # because it seems that the estimated travel time is overestimated, some P waves have already arrived, so this term allows to take them into account as well.  


######################## d
try:
    os.mkdir(f'{run_folder}')
except:
    pass

if cluster==True:
    datapath = f'/home/parisnic/traces/'
else:
    datapath = f'/media/parisnic/STOCKAGE/traces/' 

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
output_type = 'VEL'
starttime_obspy = eq_time+start_delay_dl-pad 
endtime_obspy = eq_time+start_delay_dl+duration_dl+pad

bad_station_indexes = []

for i in tqdm(range(len(station_list_clean))):
    if os.path.exists(datapath + station_list_clean[i] + '.mseed')==False:
        try:
            st = client.get_waveforms(network=network_list_clean[i], station=station_list_clean[i], location='*', channel=channels, attach_response=True,
                                              starttime=starttime_obspy,endtime=endtime_obspy)
            st.remove_response(output=output_type)
            st.detrend('demean')            
            tr = st[0]
            tr.write(datapath + station_list_clean[i] + ".mseed", format="MSEED")              
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


for i in range(len(station_list_clean)): #on a supprimé les mauvaises stations et téléchargé les traces, avec le minimum de processing possible pour les garder en 
    #bon état
    st = read(datapath + station_list_clean[i] + '.mseed')

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
x, y = np.meshgrid(x, y)

# Depth of the source (assumed to be known)
z = eq_depth/1000 #converted to km 

# Initialize the RMS for each potential source
# Initialize the stacks for each potential source
obs = np.array(arr_list_good) #normalement devrait avoir taille  nr, nt 
nt = len(obs[0,:]) #number of samples : should be the same for all traces since we decimated them

nl = nt//(plage-overlap)-1 #nnumber of time windows : depends on duration, fs, plage and overlap 
rms = np.zeros((nl,x.shape[0],x.shape[1]))
stacks = np.zeros((nl,plage, x.shape[0], x.shape[1]))

#on utilise le ray tracing 1D d'obspy pour estimerle travel time entre la source et chacune des stations 
model = TauPyModel(model='iasp91')



#avec les cross correlations on va pouvoir trouver de combien on doit shift chaque trace par rapport à la trace de référence == trace 0 
# par contre l'effet de ça c'est qu'on perd un peu la notion de temps 0 pour l'eq ...  mais c'est handled normalement ...

#maintenant on cherche de combien de temps corriger chacune des traces par rapport à ref avec cross corr en se basant sur loc de 'épicentre fournit par iris : 

#################################### CROSS CORRELATION POUR CORRECTION DE TTIME EMPIRIQUE ######

n_corr = np.zeros(len(distances_list_clean)) #le nombre d'échantillons par lequel il faudra shifter les traces par rapport à la ref pour avoir la meilleure cohérence de la P!
polarity = np.ones(len(distances_list_clean)) #pareil on détermine la meilleur epolarité avec les cross correlations  -> on condière que polarite de k=0 est 1 

### s'occupe de traiter la ref #### 
trace_ref = functions.normalize_trace(obs[0,:]) #on prend la ref, on la normalize et on la shift 
ttime = model.get_ray_paths(source_depth_in_km=eq_depth/1000, distance_in_degree=distances_list_clean[0], phase_list=['P'])[0].time
n_shift = int((ttime-start_delay-delta_static)*fs)
trace_ref = functions.shift(trace_ref,n_shift)
#####
### maintenant on va faire la même pour toutes les traces et on va aussi perform les cross correlations ### 
for k in range(1,len(distances_list_clean)): #pas beoisn de shifter la première trace obviously
    #doing similar processing as for the reference trace : traces should theoritically be aligned
    trace = functions.normalize_trace(obs[k,:])
    ttime = model.get_ray_paths(source_depth_in_km=eq_depth/1000, distance_in_degree=distances_list_clean[k], phase_list=['P'])[0].time
    n_shift = int((ttime-start_delay-delta_static)*fs)
    trace = functions.shift(trace,n_shift)
    ### but since there is lateral heterogeneity, we need to perform the cross correlations ... 
    corr_pos = np.correlate(trace_ref[(delta_static-cross_anticipation)*fs:(delta_static-cross_anticipation+cross_duration)*fs],
                        trace[(delta_static-cross_anticipation)*fs:(delta_static-cross_anticipation+cross_duration)*fs], mode='same') 
    corr_neg = np.correlate(trace_ref[(delta_static-cross_anticipation)*fs:(delta_static-cross_anticipation+cross_duration)*fs],
                        -trace[(delta_static-cross_anticipation)*fs:(delta_static-cross_anticipation+cross_duration)*fs], mode='same') 
    
    pol_pos = np.max(np.abs(corr_pos))
    pol_neg = np.max(np.abs(corr_neg))
    
    if pol_pos >= pol_neg:
        # n_corr[k] = np.argmax(np.abs(corr_pos))-len(corr_pos)//2 #en gros si c'est au milieu de la corr le shift est nul, et donc on shift de la pos - longueur de corr /2 
        polarity[k] = 1.
    else:
        # n_corr[k] = np.argmax(np.abs(corr_neg))-len(corr_neg)//2 #en gros si c'est au milieu de la corr le shift est nul, et donc on shift de la pos - longueur de corr /2 
        polarity[k] = -1.
    

# on fait plus la correction avec cross corrélation, on devrait avoir un truc similaire à avant si la polarité n'est pas inversée 
np.save(f'{run_folder}/polarity.npy',polarity)
    
##################################################################################################################
    
for i in range(x.shape[0]): #looping over potential sources  
    for j in range(x.shape[1]):
        for k in range(len(longitudes_list_clean)): # looping over receivers and computing their distance to the potnetial source location which is important to  shift the traces accordingly 
            dist_km = gps2dist_azimuth(latitudes_list_clean[k],longitudes_list_clean[k],y[i,j],x[i,j])[0]/1000
            dist = kilometers2degrees(dist_km) #  computing the distance between a potential source and the receivers 
            ###getting trael time 
            ttime = model.get_ray_paths(source_depth_in_km=eq_depth/1000, distance_in_degree=dist, phase_list=['P'])[0].time
            
            ### we now know how much to shift the trace 
            n_shift = int((ttime-start_delay-delta_static)*fs+n_corr[k]) #on a calculé au préalable le shift empirique attendu grâce aux cross-corr 

            # polarity = functions.handle_polarity(y[i,j],x[i,j],latitudes_list_clean[k],longitudes_list_clean[k]) # 
            
            trace = polarity[k]*functions.normalize_trace(obs[k,:])
            obs_shifted = functions.shift(trace,n_shift)
            for l in range(nl):
                starting_idx = l*(plage-overlap)
                stacks[l,:,i,j] += obs_shifted[starting_idx:starting_idx+plage]
        
        for l in range(nl):
            rms[l,i,j]  = np.sqrt(np.sum((stacks[l,:,i,j])**2)) 
            

## définition du temps en output (aurait pu le baser sur la time liste good et soustraire le start_delay 
times_to_save = np.zeros((nl,2)) #pour mettre le tbeg et tend de chacune window 

for i in range(nl):
    times_to_save[i,0] = (plage-overlap)/fs*i  - delta_static #comme on corrige traces pour aligner à delta_static, ça veut dire que premier échantillon à 0-delta_static 
    times_to_save[i,1] = times_to_save[i,0]+plage/fs #plage pour avoir la durée de la window ajoutée au début de la window  
    

np.save(f'{run_folder}/rms.npy',rms)
# np.save(f'{run_folder}/stacks.npy',stacks)
np.save(f'{run_folder}/x.npy',x)
np.save(f'{run_folder}/y.npy',y)
np.save(f'{run_folder}/times.npy',times_to_save)
# np.save(f'{run_folder}/latitudes_list_clean.npy',data['Latitude'].values)
# np.save(f'{run_folder}/longitudes_list_clean.npy',data['Longitude'].values)


### on profite à présent du run pour aussi en faire des figures et extraire la courbe de vitesse parce que why not c'est pas le temps que ça prend 
#pourrait aussi faire une finite diff pour calculer l'évolution de la vitesse avec le temps  mais on a déjà une estimation pas trop mal de la vitesse avec 1er order

times_fig = np.mean(times_to_save, axis = 1)

fig, ax = plt.subplots()
ax.set_title('RMS map for various source locations ')
ax.set_xlabel('lon (°)')
ax.set_ylabel('lat (°)')
ax.set_aspect('equal')  # Make sure the aspect ratio is equal

# Initialize pcolormesh
im = ax.pcolormesh(x,y,rms[0,:,:], cmap='turbo',vmin=np.min(rms),vmax=np.max(rms))
text_annotation = ax.text(-73.5,-37.5,f't={times_fig[0]}s',color='red', fontsize=15)
ax.scatter(eq_lon, eq_lat, marker='*', s=30, color='green')
fig.colorbar(im, ax=ax)


# Function to update the pcolormesh for each time step
def update(frame):
    text_annotation.set_text(f't={times_fig[frame]}s')
    im.set_array(rms[frame,:,:].ravel())
    
# Create an animation
ani = FuncAnimation(fig, update, frames=len(times_fig), repeat=False)

# Save the animation as an MP4 video
ani.save(f'{run_folder}/animation_pcolormesh.mp4', writer='ffmpeg')


###############################

fig, ax = plt.subplots()
ax.set_title('RMS map for various source locations ')
ax.set_xlabel('lon (°)')
ax.set_ylabel('lat (°)')
ax.set_aspect('equal')  # Make sure the aspect ratio is equal

contours = ax.contourf(x,y,rms[0,:,:], cmap='turbo',levels=np.linspace(np.min(rms), np.max(rms), 20))
ax.scatter(eq_lon, eq_lat, marker='*', s=30, color='green')
text_annotation = ax.text(-73.5,-37.5,f't={times_fig[0]}s',color='red', fontsize=15)
fig.colorbar(contours, ax=ax)


# Function to update the pcolormesh for each time step
def update(frame):
    global contours
    text_annotation.set_text(f't={times_fig[frame]}s')
    # Update the pcolormesh with the data at the current time step
    contours.collections.clear()  # Clear the old contour collections
    contours = ax.contourf(x, y, rms[frame, :, :], cmap='turbo', levels=np.linspace(np.min(rms), np.max(rms), 20))
    
# Create an animation
ani = FuncAnimation(fig, update, frames=len(times_fig), repeat=False)

# Save the animation as an MP4 video
ani.save(f'{run_folder}/animation_contourf.mp4', writer='ffmpeg')


###################################"

#à voir comment ça behave puisque maintenant on use  la vraie position de la source au lieu de la première trouvée 

xx = np.zeros(len(times_fig))
yy = np.zeros(len(times_fig))
zz = np.zeros(len(times_fig))
distances = np.zeros(len(times_fig))

for i in range(len(times_fig)):
    idx = np.unravel_index(np.argmax(np.abs(rms[i,:,:]), axis=None), rms[i,:,:].shape)
    xx[i] = x[0,idx[0]] #on lui donne des coordonnées au lieu des indices 
    yy[i] = y[idx[1],0]
    zz[i] = rms[i,idx[0],idx[1]]    
    distances[i] = gps2dist_azimuth(yy[i],xx[i],eq_lat,eq_lon)[0]/1000 #calcul distance par rapport à première position e, kliomètres 
    

### we only select idexes where the rms is larger than 50% of the max rms in zz 
selected_idx = np.where(zz>=0.5*np.max(zz))[0]
selected_idx = np.append(0,selected_idx) #rajoute le 0 au cas ou 

weights = np.ones(len(times_fig[selected_idx]))
weights[0] = 99999999 #to make sure the fit passes through the 0!
fit = np.polyfit(times_fig[selected_idx],distances[selected_idx],1,w=weights )


fig, ax = plt.subplots()
ax.set_title('Distance of the rupture front to the epicenter as a function of time')
ax.set_ylabel('Distance to *epicenter* (km))')
ax.set_xlabel('Time since origin (s)')
ax.plot(times_fig[selected_idx], np.polyval(fit,times_fig[selected_idx]))
ax.plot(times_fig[selected_idx],distances[selected_idx])
ax.text(0.05,0.9, f'v={round(fit[0],1)} km/s', fontsize=12, transform=ax.transAxes)
