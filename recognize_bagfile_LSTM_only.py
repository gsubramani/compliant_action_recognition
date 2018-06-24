
from signal_database.labelSelectedData import SignalDB,SignalBundle,LabeledData
from bagfile_io.bagfile_reader import bagfile_reader,write_to_bagfile
import argparse
import numpy as np

import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import get_signal_features,LSTM_model,SeqGen,sortify
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


    wrench1_,wrencht1 = bfr.get_topic_msgs("/ftmini40")
    wrench2_,wrencht2 = bfr.get_topic_msgs("/ftmini402")
    # labels_,labelst = bfr.get_topic_msgs("/labels")

    timesamples = wrencht1

    # labels = [label_messsage.data for label_messsage in labels_]

    wrench1 = np.array([[f.wrench.force.x,f.wrench.force.y,f.wrench.force.z,
                    f.wrench.torque.x,f.wrench.torque.y,f.wrench.torque.z] for f in wrench1_])

    wrench2 = np.array([[f.wrench.force.x,f.wrench.force.y,f.wrench.force.z,
                    f.wrench.torque.x,f.wrench.torque.y,f.wrench.torque.z] for f in wrench2_])


    wrench1 = np.array([np.interp(timesamples,  wrencht1, wrench1[:,ii]) for ii in range(6)]).transpose()
    wrench2 = np.array([np.interp(timesamples,  wrencht2, wrench2[:,ii]) for ii in range(6)]).transpose()


    data = np.transpose(np.append(wrench1,wrench2,axis = 1)).tolist()

    #addings stuff to the database
    sb = SignalBundle(data,timesamples)
    ld = LabeledData(sb)
    data = [ld]

    timestamps = np.array(ld.signal_bundle.timestamps)

    # estimation

    X_signal,Y = get_signal_features(data)
    # X_wavelet, _ = get_wavelet_features(data)
    # X = np.append(X_signal,X_wavelet,axis = 1)
    X = X_signal
    model_LSTM_saved = load_model("./trained_models/LSTM_only.hdf5")


    window = 100
    batch_stride = 1
    batch_size = 1
    num_classes = len(label_names_df)
    data_dim = np.shape(X)[1]

    model_LSTM = LSTM_model(data_dim, window, batch_size, num_classes, stateful=False)

    model_LSTM.set_weights(model_LSTM_saved.get_weights())

    skip = 100

    Y_one_hot = pd.get_dummies(Y,columns=label_names_df)
    # sg = SeqGen(X[0:-1:skip],Y_one_hot[0:-1:skip],1,window,batch_stride=1)
    sg = SeqGen(X[0:-1:skip], timestamps[0:-1:skip], 1, window, batch_stride=1)


    predictions = np.array([]).reshape(0,len(label_names_df))

    for ii in range(len(sg)):
        predictions = np.append(predictions,model_LSTM.predict(sg[ii][0], batch_size=batch_size),axis = 0)

    # predictions = model_LSTM.predict(sg[0][0], batch_size=batch_size)

    print np.shape(sg[ii][1])

    predict_df = pd.DataFrame(predictions, columns=label_names_df)
    predict_list = list(predict_df.idxmax(axis=1))



    print predict_list



    test_one_hot_predict_series = []
    test_one_hot_predict_series_timestamps = []
    test_one_hot_truth_series = []

    for ii in range(len(sg)):
        test_one_hot_predict = model_LSTM.predict(sg[ii][0], batch_size=batch_size)
        test_one_hot_predict_timestamps = sg[ii][1].tolist()
        test_one_hot_predict_df = pd.DataFrame(test_one_hot_predict, columns=label_names_df)
        test_one_hot_predict_series_batch = list(test_one_hot_predict_df.idxmax(axis=1))
        test_one_hot_predict_series = test_one_hot_predict_series + test_one_hot_predict_series_batch
        test_one_hot_predict_series_timestamps = test_one_hot_predict_series_timestamps + test_one_hot_predict_timestamps

        # test_one_hot_truth = (sg[ii][1] > 0.9).astype(int)
        # print np.shape(test_one_hot_truth)
        # test_one_hot_truth_df = pd.DataFrame(test_one_hot_truth, columns=label_names_df)
        # test_one_hot_truth_series_batch = list(test_one_hot_truth_df.idxmax(axis=1))
        # test_one_hot_truth_series = test_one_hot_truth_series + test_one_hot_truth_series_batch


    predict_list_sorted_timestamps = test_one_hot_predict_series_timestamps
    predict_list_sorted = sortify(test_one_hot_predict_series,test_one_hot_predict_series_timestamps)

    print predict_list_sorted

    plt.figure(1)
    ax = plt.subplot(2, 1, 1)

    plotResult_colorbars(predict_list_sorted, range(len(predict_list_sorted)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    # ax.set_xlim([min(predict_list_sorted_timestamps), max(predict_list_sorted_timestamps)])
    ax = plt.subplot(2, 1, 2)

    # plotResult_colorbars(test_one_hot_truth_series, range(len(test_one_hot_truth_series)),
    #                      labelNames=list(label_names_df) + [''], medfiltwidth=1)
    # ax.set_xlim([0, len(test_one_hot_truth_series)])


    plt.show()
    exit()

    labels_t = sg[0][1]

    string_message_list = []

    for label in predict_list:
        message = String()
        message.data = label
        string_message_list.append(message)

    write_to_bagfile(bagfile, '/labels',string_message_list, labels_t, 'a', createbackup=True)
