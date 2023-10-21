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
    if os.path.exists(f'/media/parisnic/STOCKAGE/traces/{station_list_clean[i]}.mseed')==False: 
        try:
            st = client.get_waveforms(network=network_list_clean[i], station=station_list_clean[i], location='*', channel=channels, attach_response=True,
                                              starttime=starttime_obspy,endtime=endtime_obspy)
            st.remove_response(output=output_type)
            st.detrend('demean')            
            tr = st[0]
            tr.write(f"/media/parisnic/STOCKAGE/traces/{station_list_clean[i]}.mseed", format="MSEED")              
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
    st = read(f'/media/parisnic/STOCKAGE/traces/{station_list_clean[i]}.mseed')

    if st[0].stats.sampling_rate>fs: #on resample si trop hf 
                st.resample(fs, window='hann', no_filter=True, strict_length=False)

    tr = st[0]
    fs_list_clean.append(tr.stats.sampling_rate)
            
    if filtered!=None:
        st.filter(filtered,freqmin=freq[0],freqmax=freq[1],corners=order,zerophase=True) #fs_list_clean[i],
            
#     if filtered=='bandpass':
#         b,a = butter(order, [freq[0],freq[1]], btype='bandpass', output='ba', fs=fs_list_clean[i])
#         arr = filtfilt(b,a,arr)

#     elif filtered=='highpass':
#         b,a = butter(order, freq, btype='highpass', output='ba', fs=fs_list_clean[i])
#         arr = filtfilt(b,a,arr)
        
    arr = tr.data
    time_arr = (tr.times(reftime=eq_time)) #on doit à présent chercher dans time_arr où se trouve     
    indexes_to_keep = np.where((time_arr>=start_delay) & (time_arr<=start_delay+duration))
    time_arr = time_arr[indexes_to_keep]
    arr = arr[indexes_to_keep]
    arr_list.append(list(arr))
    time_list.append(list(time_arr))
    color_list.append(int(azimuth_list_clean[i]))


# In[11]:


# # ici on vérifie rapidement des spectrogrames histoire de voir la gamme de fréquence à conserver afin de conserver plutôt la P
# #sinon on peut juste mute les autres arrivées en se basant sur loi de vitesse en focntion de la distance
# plt.close('all')

# trace_number = 2
# order = 2
# freq = 2

# b,a = butter(order, freq, btype='highpass', output='ba', fs=fs_list_clean[trace_number])
# arr = arr_list[trace_number]
# arr = filtfilt(b,a,arr)

# plt.figure()
# plt.plot(time_list[trace_number], arr)


# f, t, Z = sp.stft(arr, fs=fs_list_clean[trace_number], window='hann', nperseg=100, noverlap=50, nfft=None, 
#                             detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum')

# plt.figure()
# plt.pcolormesh(t,f,np.abs(Z))


# In[12]:


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



# In[9]:


plt.close('all')

zoom = None #[0,60,30,45] #x0, x1, y0, y1  
scaling = 'float'

arr_list_transformed = []
for i in range(len(station_list_clean)):
    if scaling=='float':
        arr_transformed = 3e4*np.array(arr_list_good[i]) + distances_list_clean[i]    
    elif scaling =='distance': 
        arr_transformed = 5e2* 4*np.pi*(distances_list_clean[i]) * np.array(arr_list_good[i]) + distances_list_clean[i] #+ dist clean pour faire offset
    elif scaling == 'binary':
        max_amp = np.max(np.abs(np.array(arr_list_good[i])))  
        coeff = 3/max_amp
        arr_transformed = coeff*np.array(arr_list_good[i]) + distances_list_clean[i]
    else:
        arr_transformed = np.array(arr_list_good[i]) + distances_list_clean[i] #pas de scaling, donc on verra rien sans zoom   
    
    arr_list_transformed.append(arr_transformed)

fig, ax = plt.subplots(figsize=(15,15))
lc = functions.multiline(time_list_good, arr_list_transformed, color_list, cmap='jet', lw=1)
axcb = fig.colorbar(lc)
axcb.set_label('Azimuth from eq(°)')
ax.set_title('Broadband section')
ax.set_xlabel(f'Time since origin time ({eq_time})')
ax.set_ylabel('Offset (°)')

for i, distance in enumerate(distances_list_clean):
    ax.text(time_list_good[i][0],distance,f'{network_list_clean[i]}.{station_list_clean[i]}')

if zoom!=None:
    ax.set_xlim(zoom[0],zoom[1])
    ax.set_ylim(zoom[2],zoom[3])
        
# plt.savefig(f'{eq}_broadband.png', dpi=300, bbox_inches='tight')


# We will now mute the later part of the signals to make sure we don't get other waves or noise

# In[13]:


#muting the later parts of the signals 

# print(len(arr_list_good[0]))
# np.array(arr_list_good)


# # We now have traces ready to be used for the inversion process

# In[14]:


# model = TauPyModel(model='iasp91')
# print( model.get_ray_paths(source_depth_in_km=eq_depth/1000, distance_in_degree=distances_list_clean[0], phase_list=['P'])[0].time) 


# ## version avec rms sur full duration

# In[15]:


# Create a grid of potential sources : we use lat lon coordinates intead of cartesian coordinates for the gridsearch to make it easier
x    = np.linspace(-74,-70, 25)
y    = np.linspace(-38, -33, 25)
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


# In[109]:


# plt.figure()
# plt.plot(stacks[:,0,0])
# plt.plot(arr_list_good[0])


# In[25]:


#on plot à présent la rms matrix pour voir si on a bien un alignement de forte amplitude 
plt.figure()
plt.title('RMS map for various source locations')
plt.xlabel('lon (°)')
plt.ylabel('lat (°)')
plt.pcolormesh(x,y,rms)
# bon on voit que c'est pas fou mais on va quand même essayer de voir ... 


# In[33]:


#maintenant à chaque pas de temps on cherche où l'énergie est la plus forte 

# Initialize a time vector for locations and stack amplitude
xx = np.zeros(nt)
yy = np.zeros(nt)
zz = np.zeros(nt)

#à cherche à présent dans matrice de stack pour chaque time step à quelle position l'énergie est la plus forte -> on trouve alors la position de la source à chaque pas de temps 
#il nous  faut donc une matrice de taille (length signal, 2 ) histoire de pouvoir store x et y 

for i in range(nt):
    idx = np.unravel_index(np.argmax(np.abs(stacks[i,:,:]), axis=None), stacks[i,:,:].shape)
    xx[i] = x[0,idx[0]] #on lui donne des coordonnées au lieu des indices 
    yy[i] = y[idx[1],0]
    zz[i] = stacks[i,idx[0],idx[1]] #zz est l'maplitude associée au point de coordonnées x et y qui a la plus forte amplitude dans le stack à temps i 


# In[34]:


max_amplitude_total = np.max(np.abs(stacks))
time_mod = np.array(time_list_good[0])

xx_select = []
yy_select = []
time_select = []

for i in range(nt): #on boucle encore sur les pas de temps et on vérifie si les amplitudes sont supérieures au threshold fourni 
    if np.abs(zz[i])>0.8*max_amplitude_total: 
        xx_select.append(xx[i])
        yy_select.append(yy[i])
        time_select.append(time_mod[i])
        
xx_select = np.array(xx_select)
yy_select = np.array(yy_select)
time_select = np.array(time_select)

plt.figure()
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.title('Time evolution of the source location')
plt.scatter(xx_select,yy_select,c=time_select,cmap='jet')


# In[36]:


#calcule les distances par rapport à la première position xx et yy selectionnés; car le premier selectionné est à la position où le séisme a eu lieu
distances = np.zeros(len(xx_select))
for i in range(len(distances)):
    distances[i] = gps2dist_azimuth(yy_select[i],xx_select[i],yy_select[0],xx_select[0])[0]/1000 
# print(distances)


#calcul des distances intersources sélectionnées pour pouvoir calculer des vitesses locales le long de la faille !!
short_distances = distances[1:]-distances[:-1] #difference between the distance of neighbour points = short distances 
short_times = time_select[1:]-time_select[:-1]


plt.figure()
plt.title('distance from south western point vs time')
plt.xlabel('time (s)')
plt.ylabel('distance from the center (km)')
plt.plot(time_select,distances) #slope is the velocity

#on a une évolution linéaire de la distance avec le temps, donc vitesse constante et on peut calculer la vitesse avec un polyfit pour avoir la slpe :
print('The rupture velocity is ', np.polyfit(time_select,distances,1)[0], 'km/s.')


# ## Version avec RMS sur fenêtres de temps

# In[25]:


x    = np.linspace(-120.75, -120.25, 50)
y    = np.linspace(35.75, 36.0, 50)
x, y = np.meshgrid(x, y)

z = eq_depth/1000 #converted to km 

obs = np.array(arr_list_good) #normalement devrait avoir taille  nr, nt 
nt = len(obs[0,:]) #number of samples : should be the same for all traces since we decimated them

vp = 5.71 #trouvé avec la slope des arrivées sur la broad section 

plage = 250 # nombre de points d ela plage 
rms = np.zeros((nt//plage,x.shape[0], x.shape[1])) #auatant de carte de rms que de plages différentes 
stacks = np.zeros((nt//plage,plage, x.shape[0], x.shape[1])) #pour chaque  carte de rms il faut fournir la carte de stacks

#on doit calculer la map de rms sur des plages de temps

print('Computing the evolution of the rms map, please wait...')
for l in tqdm(range(nt//plage)): #looping over the different ranges, so the different maps 
    for i in range(x.shape[0]): #looping over sources 
        for j in range(x.shape[1]):
            dist = np.zeros(len(distances_list_clean))
            for k in range(len(longitudes_list_clean)): # looping over receivers
                dist[k] = gps2dist_azimuth(latitudes_list_clean[k],longitudes_list_clean[k],y[i,j],x[i,j])[0]/1000 #  computing the distance between a potential source and the receivers 
            ttime = dist/vp
            obs_shifted = shift(obs[:,l*plage:(l+1)*plage], ttime)
            stacks[l,:,i,j] = np.sum(obs_shifted, axis=0)
            rms[l,i,j]  = np.sqrt(np.sum((stacks[l,:,i,j])**2)) 


# In[35]:


plt.figure()
plt.title('RMS map for various source locations')
plt.xlabel('lon (°)')
plt.ylabel('lat (°)')
plt.pcolormesh(x,y,rms[2,:,:],vmin=np.min(rms),vmax=np.max(rms)) # using vmin and vmax to keep the same colorbar for all  maps 
plt.colorbar()


# In[ ]:





# In[ ]:





# In[ ]:




