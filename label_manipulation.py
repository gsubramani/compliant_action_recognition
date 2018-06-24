import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d


def median_filter_labels(labels,window):

    if window%2 == 0:
        window = window = 1
    label_names_uniques,labels_ints = np.unique(labels,return_inverse=True)
    labels_ints_filtered = medfilt(labels_ints,1001)
    filtered_labels = [label_names_uniques[int(val)] for val in labels_ints_filtered]
    return filtered_labels

def fill_closest(labels,replace = ""):
    unknown_samples = [ii for ii,label in enumerate(labels) if label == replace]
    known_samples = [ii for ii,label in enumerate(labels) if label != replace]
    known_labels = [label for ii,label in enumerate(labels) if label != replace]
    label_names_uniques,known_labels_ids = np.unique(known_labels,return_inverse=True)
    f1 = interp1d(known_samples, known_labels_ids, kind='nearest',fill_value='extrapolate')
    unknown_labels_close_enough_ids = f1(unknown_samples)
    unknown_labels_close_enough = [label_names_uniques[int(label_id)] for label_id in unknown_labels_close_enough_ids]
    current_id = 0
    pl1 = 0
    pl2 = 0
    out_list = []
    for ii in range(len(labels)):
        if unknown_samples[pl1] == current_id:
            out_list.append(unknown_labels_close_enough[pl1])
            pl1 += 1
        else:
            out_list.append(known_labels[pl2])
            pl2 +=1
        current_id += 1
    return out_list
