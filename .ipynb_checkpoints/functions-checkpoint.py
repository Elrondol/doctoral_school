import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from typing import List



def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def normalize_trace(trace):
    return trace/np.max(np.abs(trace))
    

def handle_polarity():
    az_mecha = 178 #azimuth de la faille
    # if az < az_mecha
    #     return 1
    # else:
    #     return -1
    return 1
    

def shift(obs, tshift, fs_list_clean):
    obs_shifted = np.zeros((len(obs[:,0]),len(obs[0,:])))
    #looping over the decievers  for a given source     
    for i in range(len(obs[:,0])):
        n_shift = int(tshift[i]*fs_list_clean[i]) #on sait de combien on doit shift la trace 
        polarity = handle_polarity()   # on sait si la trace doit être changée depolarisée
        trace = polarity*normalize_trace(obs[i,:])
        obs_shifted[i,:] = np.roll(trace,-n_shift) #doit shift de moins n pour que ça aille vers la gauche  
    return obs_shifted 
