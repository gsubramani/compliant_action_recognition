import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors

from scipy.signal import medfilt
import matplotlib.patches as patches
color_map = "Paired"


def splitSignal(array):
    split_indices = []
    labelorder = []
    sizes = []
    previous_element = array[0]
    sizes.append(0)
    labelorder.append(previous_element)
    for element in array:
        if(previous_element == element):
            sizes[-1]  = sizes[-1] + 1
            previous_element = element
        else:
            sizes.append(1)
            labelorder.append(element)
            previous_element = element
    return sizes,labelorder



def plotlabelpositions(array,timestamps):
    sizes,labelorder = splitSignal(array)
    label_indices = []
    start = []
    stop = []
    sum = 0
    for size in sizes:
        label_indices.append(timestamps[sum+size/2])
        start.append(timestamps[sum])
        sum = size + sum
        #print sum,len(timestamps)
        stop.append(timestamps[sum-1])

    return label_indices,labelorder,start,stop



def get_color_map(labelNames):
    color_map_rgba = matplotlib.cm.ScalarMappable(cmap=color_map).to_rgba(range(len(labelNames)))
    color_map_rgba[labelNames.index('')] = [1,1,1,1]
    return color_map_rgba


def plot_legend(label_names,ax = None):
    if ax is None:
        plt.figure()
        ax = plt.subplot(1,1,1)

    if '' not in label_names : label_names = label_names + ['']

    color_map = get_color_map(label_names)

    starts = range(len(label_names))
    stops = range(1,len(label_names) + 1)

    for color,start,stop in zip(color_map,starts,stops):
        ax.axvspan(start, stop, alpha=1, color=color)
    for start in starts:
        ax.axvline(start, linewidth=1, color='k')
    ax.axvline(len(label_names), linewidth=1, color='k')
    for id,label in enumerate(label_names):
        ax.text(id + 0.5, 0.5, label, fontsize=12,horizontalalignment='center')
    return ax


def plotResult_colorbars(testPredict,
                         timestamps,ax = None,
                         labelNames = [],
                         medfiltwidth = 101,
                         labelnumfilts= None,
                         time_ticks = False):


    if ax is None:
        ax = plt.gca()
    if len(labelNames) ==0:
        labelNamesSet = set(testPredict)
        labelNames = list(labelNamesSet)


    testPredictIds = [labelNames.index(label) for idx, label in enumerate(testPredict)]
    testPredictIds = medfilt(testPredictIds,medfiltwidth)
    testPredictIds = (testPredictIds.astype(int)).tolist()
    testPredict = [labelNames[id] for id in testPredictIds]

    if labelnumfilts !=None:
        out_testPredict = []
        sizes, labelorder  = splitSignal(testPredict)
        for size,label in zip(sizes,labelorder):
            if size < labelnumfilts[labelNames.index(label)]:
                out_testPredict = out_testPredict + ['']*size
            else: out_testPredict = out_testPredict + [label]*size
        testPredict = out_testPredict
    #print splitSignal(testPredict)

    testPredictIds = [labelNames.index(label) for idx, label in enumerate(testPredict)]


    for jj,label in enumerate(labelNames):
        testPredictIds[jj] = jj

    color_map_rgba = matplotlib.cm.ScalarMappable(cmap=color_map).to_rgba(range(len(labelNames)))

    color_map_rgba[labelNames.index('')] = [1,1,1,1]

    ax.set_ylim([-0.25, 0.25])
    ax.get_yaxis().set_visible(False)

    ax.get_xaxis().set_visible(time_ticks)

    if testPredict != None:
        label_indices ,label_order,starts,stops = plotlabelpositions(testPredict,timestamps)

        # label_order_plot = [label.split('_')[0] for label in label_order]
        # ax.xaxis.set(ticks=label_indices, ticklabels=label_order_plot)
        for start,stop,label in zip(starts,stops,label_order):
            color = color_map_rgba[labelNames.index(label)]
            ax.axvspan(start, stop, alpha=1, color=color)
            ax.axvline(start,linewidth=1, color='k')
            ax.axvline(stop, linewidth=1, color='k')

            # ax.xaxis.grid(False)
            ax.yaxis.grid(True)

    return ax

def plot_confusion_matrix(confusion_matrix,ax = None,labelNames = None):
    #actual ->rows ; predicted _> columns
    if ax is None:
        ax = plt.gca()
    for idx,row in enumerate(confusion_matrix):
        if idx == 0:
            continue
        total = np.sum(row)
        total = total + 1
        for idy,element in enumerate(row):
            if idy == 0:
                continue
            norm = matplotlib.colors.Normalize(vmin = 0, vmax=total)
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap="Blues")
            ax.scatter(idx,idy, s = 1 ,lw=0)
            ax.add_patch(patches.Rectangle((idx - 0.4, idy - 0.4 ),0.8,0.8,color = m.to_rgba(element)))
            ax.text(idx,idy,'%.1f' % ((float(element)/total)*100),horizontalalignment='center',verticalalignment='center',color = "crimson",family = 'sans-serif',weight = 'bold')
            # ax.text(idx,idy,element,horizontalalignment='center',verticalalignment='center',color = "crimson")
    if labelNames == None:
        label_names = range(len(confusion_matrix))

    ax.xaxis.set(ticks=np.arange(0, len(labelNames)), ticklabels=labelNames)
    ax.yaxis.set(ticks=np.arange(0, len(labelNames)), ticklabels=labelNames)



