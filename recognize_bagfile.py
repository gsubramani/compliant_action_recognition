
from signal_database.labelSelectedData import SignalDB,SignalBundle,LabeledData
from bagfile_io.bagfile_reader import bagfile_reader,write_to_bagfile
import argparse
import numpy as np

import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import get_signal_features,LSTM_model,SeqGen,sortify,get_wavelet_features

from keras.callbacks import TensorBoard
from keras.models import load_model
import cPickle as pickle
from std_msgs.msg import String





__doc__ = "Adds bag files to the database. Used to create training data for detecting compliant actions "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Provide the bagfile path")
    # parser.add_argument("destination", help="The location of the database")
    # parser.add_argument("-w", "--write_permission")
    # parser.add_argument("-f",action='store_true')
    args = parser.parse_args()

    bagfile = args.bagfile

    label_names_df = pickle.load(open("label_names_df.pkl", "rb"))

    bfr = bagfile_reader(bagfile)

    wrench1_, wrencht1 = bfr.get_topic_msgs("/ftmini40")
    wrench2_, wrencht2 = bfr.get_topic_msgs("/ftmini402")
    # labels_,labelst = bfr.get_topic_msgs("/labels")

    timesamples = wrencht1

    # labels = [label_messsage.data for label_messsage in labels_]

    wrench1 = np.array([[f.wrench.force.x, f.wrench.force.y, f.wrench.force.z,
                         f.wrench.torque.x, f.wrench.torque.y, f.wrench.torque.z] for f in wrench1_])

    wrench2 = np.array([[f.wrench.force.x, f.wrench.force.y, f.wrench.force.z,
                         f.wrench.torque.x, f.wrench.torque.y, f.wrench.torque.z] for f in wrench2_])

    wrench1 = np.array([np.interp(timesamples, wrencht1, wrench1[:, ii]) for ii in range(6)]).transpose()
    wrench2 = np.array([np.interp(timesamples, wrencht2, wrench2[:, ii]) for ii in range(6)]).transpose()

    data = np.transpose(np.append(wrench1, wrench2, axis=1)).tolist()

    # addings stuff to the database
    sb = SignalBundle(data, timesamples)
    ld = LabeledData(sb)
    data = [ld]

    timestamps = np.array(ld.signal_bundle.timestamps)

    ####################################################################################################################
    ##  Wavelet Recognition  ###########################################################################################
    ####################################################################################################################

    # estimation
    # training data
    features = (0, 5)
    X_signal, _ = get_signal_features(data)
    X_wavelet, _ = get_wavelet_features(data, features=features)
    X = np.append(X_signal, X_wavelet, axis=1)

    # Loading and configuring model parameters
    window = 100
    batch_stride = 1
    batch_size = 1
    num_classes = len(label_names_df)
    data_dim = np.shape(X)[1]

    model_LSTM_saved = load_model("./trained_models/LSTM_wavelet_100ep_5feat.hdf5")
    model_LSTM = LSTM_model(data_dim, window, batch_size, num_classes, stateful=False)
    model_LSTM.set_weights(model_LSTM_saved.get_weights())

    # Creating data blocks for training the model
    skip = 100
    sg = SeqGen(X[0:-1:skip], timestamps[0:-1:skip], 1, window, batch_stride=1)

    predictions = np.array([]).reshape(0, len(label_names_df))
    predictions_timestamps = np.array([])
    for ii in range(len(sg)):
        predictions = np.append(predictions, model_LSTM.predict(sg[ii][0], batch_size=batch_size), axis=0)
        predictions_timestamps = np.append(predictions_timestamps, sg[ii][1], axis=0)

    predictions_wavelets = sortify(predictions, predictions_timestamps.tolist())

    predict_df = pd.DataFrame(predictions, columns=label_names_df)
    predict_list = list(predict_df.idxmax(axis=1))

    predict_list_sorted = sortify(predict_list, predictions_timestamps.tolist())

    predict_list_sorted_wavelet = predict_list_sorted

    ####################################################################################################################
    ##  LSTM_only Recognition  #########################################################################################
    ####################################################################################################################

    X_signal, Y = get_signal_features(data)
    X = X_signal

    window = 100
    batch_stride = 1
    batch_size = 1
    num_classes = len(label_names_df)
    data_dim = np.shape(X)[1]

    model_LSTM_saved = load_model("./trained_models/LSTM_only.hdf5")
    model_LSTM = LSTM_model(data_dim, window, batch_size, num_classes, stateful=False)
    model_LSTM.set_weights(model_LSTM_saved.get_weights())

    skip = 100

    sg = SeqGen(X[0:-1:skip], timestamps[0:-1:skip], 1, window, batch_stride=1)

    predictions = np.array([]).reshape(0, len(label_names_df))
    predictions_timestamps = np.array([])

    for ii in range(len(sg)):
        predictions = np.append(predictions, model_LSTM.predict(sg[ii][0], batch_size=batch_size), axis=0)
        predictions_timestamps = np.append(predictions_timestamps, sg[ii][1], axis=0)

    predictions_LSTM_only = sortify(predictions, predictions_timestamps.tolist())

    predict_df = pd.DataFrame(predictions, columns=label_names_df)
    predict_list = list(predict_df.idxmax(axis=1))

    predict_list_sorted = sortify(predict_list, predictions_timestamps.tolist())
    predict_list_sorted_LSTM_only = predict_list_sorted

    plt.figure(1)
    ax = plt.subplot(3, 1, 1)
    plotResult_colorbars(predict_list_sorted_wavelet, range(len(predict_list_sorted_wavelet)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    ax.set_xlim([0, len(predict_list_sorted_wavelet)])

    ax = plt.subplot(3, 1, 2)
    plotResult_colorbars(predict_list_sorted_LSTM_only, range(len(predict_list_sorted_LSTM_only)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    ax.set_xlim([0, len(predict_list_sorted_wavelet)])

    predictions_combined = np.array(predictions_wavelets) * np.array(predictions_LSTM_only)

    predict_df = pd.DataFrame(predictions_combined, columns=label_names_df)
    predict_list = list(predict_df.idxmax(axis=1))

    ax = plt.subplot(3, 1, 3)
    plotResult_colorbars(predict_list, range(len(predict_list)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=11)
    ax.set_xlim([0, len(predict_list_sorted_wavelet)])

    plt.show()