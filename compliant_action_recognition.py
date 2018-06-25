from signal_database.labelSelectedData import SignalDB,SignalBundle,LabeledData
from bagfile_io.bagfile_reader import bagfile_reader,write_to_bagfile
import argparse
import numpy as np

import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import relabel_data,get_signal_features,LSTM_model,SeqGen,load_db,get_wavelet_features

from keras.callbacks import TensorBoard
from keras.models import load_model
import cPickle as pickle
from std_msgs.msg import String
import os

label_map = {
        "freespace": "freespace",
        "grab_hex": "grab",
        "grab_cup": "grab",
        "grab_foamblock": "grab",
        "release_foamblock": "release",
        "release_cup": "release",
        "release_hex": "release",
        "hold_cup": "hold",
        "hold_hex": "hold",
        "hold_foamblock": "hold",
        "hand_on": "hand_on",
        "hand_off": "hand_off",
        "still": "still",
        "": ""
    }



class CompliantActionRecognition():

    class meta_data:
        pass

    def __init__(self,labeled_data_sets, label_map_in = None,label_names = None,meta = "",folder = None):
        self.md = CompliantActionRecognition.meta_data()
        self.data = labeled_data_sets
        self.folder = None

        if label_map_in is None:   self.label_map = label_map
        else:                   self.label_map = label_map_in

        if label_names is None: self.label_names_df = None
        else: self.label_names_df = label_names

        self.data = relabel_data(data, self.label_map)

        self.LSTM_wavelet_trained = False
        self.LSTM_trained = False

        self.wavelet_tf_train_completed = False
        self.wavelet_tf_test_completed = False
        self.wavelet_tf_validation_completed = False

        self.meta = meta
        self.LSTM_wavelet_batch_size = None
        self.LSTM_wavelet_timesteps = None
        self.LSTM_wavelet_epochs = None
        self.LSTM_wavelet_skip = None
        self.LSTM_wavelet_features = None
        self.LSTM_wavelet_batch_stride = None
        self.LSTM_wavelet_window = None


    def train_LSTM_wavelet(self,ids,valid_ids = None,
                           batch_size = 100,epochs = 30,skip = 100,features = (0,5),
                           batch_stride = 100,window = 100,timesteps = 100):

        self.LSTM_wavelet_batch_size = batch_size
        self.LSTM_wavelet_timesteps = timesteps
        self.LSTM_wavelet_epochs = epochs
        self.LSTM_wavelet_skip = skip
        self.LSTM_wavelet_features = features
        self.LSTM_wavelet_batch_stride = batch_stride
        self.LSTM_wavelet_window = window

        training_data = [data[id] for id in ids]
        trainX_signal,trainY = get_signal_features(training_data)

        if not self.wavelet_tf_train_completed:
            self.trainX_wavelet, _ = get_wavelet_features(training_data,features=features)


        trainX = np.append(trainX_signal,self.trainX_wavelet,axis = 1)

        train_sample_weights = compute_sample_weight('balanced', trainY)
        Y_train_one_hot = pd.get_dummies(trainY)

        if self.label_names_df is None: self.label_names_df = list(Y_train_one_hot.columns.values)

        data_dim = np.shape(trainX)[1]
        self.LSTM_wavelet_data_dim = data_dim


        num_classes = len(self.label_names_df)

        self.model_LSTM_wavelet = LSTM_model(data_dim, timesteps, batch_size, num_classes, stateful=False)

        trainX_sampled = trainX[0:-1:skip]
        trainY_one_hot_sampled = Y_train_one_hot[0:-1:skip]
        sw_train_sampled = train_sample_weights[0:-1:skip]

        sg_train = SeqGen(trainX_sampled, trainY_one_hot_sampled, batch_size, window,
                          batch_stride=batch_stride, sample_weights=sw_train_sampled)

        ## Training
        tb = TensorBoard(log_dir='./tensorboard_logs/LSTM_wavelets', histogram_freq=0,
                                    write_graph=True, write_images=True)

        if valid_ids is None:
            self.model_LSTM_wavelet.fit_generator(sg_train, epochs=epochs, callbacks=[tb])
        else:
            validation_data = [data[id] for id in valid_ids]
            validationX_signal, validationY = get_signal_features(validation_data)
            if not self.wavelet_tf_validation_completed:
                self.validationX_wavelet, _ = get_wavelet_features(validation_data, features=features)

            validation_sample_weights = compute_sample_weight('balanced', validationY)
            Y_validation_one_hot = pd.get_dummies(validationY)

            validationX = np.append(validationX_signal, self.validationX_wavelet, axis=1)

            validationX_sampled = validationX[0:-1:skip]
            validationY_one_hot_sampled = Y_validation_one_hot[0:-1:skip]
            sw_validation_sampled = validation_sample_weights[0:-1:skip]

            sg_validation = SeqGen(validationX_sampled, validationY_one_hot_sampled, batch_size, window,
                                   batch_stride=batch_stride, sample_weights=sw_validation_sampled)

            self.model_LSTM_wavelet.fit_generator(sg_train, validation_data=sg_validation,
                                     epochs=epochs, callbacks=[tb])

        self.LSTM_wavelet_trained = True
        print "Trained model!"


    def train_LSTM(self,ids,valid_ids = None,
                           batch_size = 100,epochs = 30,skip = 100,features = (0,5),
                           batch_stride = 100,window = 100,timesteps = 100):

        self.LSTM_batch_size = batch_size
        self.LSTM_timesteps = timesteps
        self.LSTM_epochs = epochs
        self.LSTM_skip = skip
        self.LSTM_features = features
        self.LSTM_batch_stride = batch_stride
        self.LSTM_window = window

        training_data = [data[id] for id in ids]
        trainX_signal,trainY = get_signal_features(training_data)

        trainX = trainX_signal

        train_sample_weights = compute_sample_weight('balanced', trainY)
        Y_train_one_hot = pd.get_dummies(trainY)

        if self.label_names_df is None: self.label_names_df = list(Y_train_one_hot.columns.values)

        data_dim = np.shape(trainX)[1]
        self.LSTM_data_dim = data_dim

        num_classes = len(self.label_names_df)

        self.model_LSTM = LSTM_model(data_dim, timesteps, batch_size, num_classes, stateful=False)

        trainX_sampled = trainX[0:-1:skip]
        trainY_one_hot_sampled = Y_train_one_hot[0:-1:skip]
        sw_train_sampled = train_sample_weights[0:-1:skip]

        sg_train = SeqGen(trainX_sampled, trainY_one_hot_sampled, batch_size, window,
                          batch_stride=batch_stride, sample_weights=sw_train_sampled)

        ## Training
        tb = TensorBoard(log_dir='./tensorboard_logs/LSTM_wavelets', histogram_freq=0,
                                    write_graph=True, write_images=True)

        if valid_ids is None:
            self.model_LSTM.fit_generator(sg_train, epochs=epochs, callbacks=[tb])
        else:
            validation_data = [data[id] for id in valid_ids]
            validationX_signal, validationY = get_signal_features(validation_data)

            validation_sample_weights = compute_sample_weight('balanced', validationY)
            Y_validation_one_hot = pd.get_dummies(validationY)

            validationX = validationX_signal

            validationX_sampled = validationX[0:-1:skip]
            validationY_one_hot_sampled = Y_validation_one_hot[0:-1:skip]
            sw_validation_sampled = validation_sample_weights[0:-1:skip]

            sg_validation = SeqGen(validationX_sampled, validationY_one_hot_sampled, batch_size, window,
                                   batch_stride=batch_stride, sample_weights=sw_validation_sampled)

            self.model_LSTM.fit_generator(sg_train, validation_data=sg_validation,
                                     epochs=epochs, callbacks=[tb])

        self.LSTM_trained = True
        print "Trained model!"


    def save_trained_models(self,folder = None):
    # Stuff is saved and important information about training is put into meta-data.pkl file
        if folder == None:
            folder = "./trained_models/temp/"

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.folder = folder

        self.md.folder = folder
        self.md.meta = self.meta

        ## Saving LSTM_wavelet stuff
        if self.LSTM_wavelet_trained:
            self.model_LSTM_wavelet.save(folder + "/LSTM_wavelet" + ".hdf5")

            self.md.LSTM_wavelet_batch_size = self.LSTM_wavelet_batch_size
            self.md.LSTM_wavelet_timesteps = self.LSTM_wavelet_timesteps
            self.md.LSTM_wavelet_epochs =  self.LSTM_wavelet_epochs
            self.md.LSTM_wavelet_skip = self.LSTM_wavelet_skip
            self.md.LSTM_wavelet_features = self.LSTM_wavelet_features
            self.md.LSTM_wavelet_batch_stride = self.LSTM_wavelet_batch_stride
            self.md.LSTM_wavelet_window = self.LSTM_wavelet_window
            self.md.label_names_df = self.label_names_df
            self.md.LSTM_wavelet_data_dim = self.LSTM_wavelet_data_dim

        if self.LSTM_trained:
            self.model_LSTM.save(folder + "/LSTM" + ".hdf5")

            self.md.folder = folder
            self.md.meta = self.meta
            self.md.LSTM_batch_size = self.LSTM_batch_size
            self.md.LSTM_timesteps = self.LSTM_timesteps
            self.md.LSTM_epochs = self.LSTM_epochs
            self.md.LSTM_skip = self.LSTM_skip
            self.md.LSTM_features = self.LSTM_features
            self.md.LSTM_batch_stride = self.LSTM_batch_stride
            self.md.LSTM_window = self.LSTM_window
            self.md.label_names_df = self.label_names_df
            self.md.LSTM_data_dim = self.LSTM_data_dim

        pickle.dump(self.md,open( folder + "meta_data.pkl", "wb" ))


if __name__ == "__main__":


    data = load_db('grab_objects', path='./')
    car = CompliantActionRecognition(data)
    car.train_LSTM_wavelet([0],epochs = 1)
    car.save_trained_models(folder = "./trained_models/test/")

    # data = relabel_data(data, label_map)
    # training_data = data[0:10]
    # test_data = data[10:-2]
    # validation_data = data[-2:]
    #
    # features = (0,5)
    #
    # trainX_signal,trainY = get_signal_features(training_data)
    # trainX_wavelet, _ = get_wavelet_features(training_data,features=features)
    # trainX = np.append(trainX_signal,trainX_wavelet,axis = 1)
    #
    # train_sample_weights = compute_sample_weight('balanced', trainY)
    # Y_train_one_hot = pd.get_dummies(trainY)
    #
    # testX_signal ,testY = get_signal_features(test_data)
    # testX_wavelet, _ = get_wavelet_features(test_data,features=features)
    # testX = np.append(testX_signal,testX_wavelet,axis = 1)
    #
    #
    # test_sample_weights = compute_sample_weight('balanced', testY)
    # Y_test_one_hot = pd.get_dummies(testY)
    #
    # validationX_signal, validationY = get_signal_features(validation_data)
    # validationX_wavelet, _ = get_wavelet_features(validation_data,features=features)
    # validationtX = np.append(validationX_signal,validationX_wavelet,axis = 1)
    #
    # validation_sample_weights = compute_sample_weight('balanced', validationY)
    # Y_validation_one_hot = pd.get_dummies(validationY)
    #
    #
    #
    #
    # label_names_df = list(Y_train_one_hot.columns.values)
    #
    # pickle.dump(label_names_df, open("label_names_df.pkl", "wb"))
    #
    #
    # data_dim = np.shape(trainX)[1]
    # timesteps = 100
    # batch_size = 100
    # num_classes = len(label_names_df)
    #
    # model_LSTM = LSTM_model(data_dim, timesteps, batch_size, num_classes, stateful=False)
    #
    #
    #
    # skip = 100
    # trainX_sampled = trainX[0:-1:skip]
    # trainY_one_hot_sampled = Y_train_one_hot[0:-1:skip]
    # sw_train_sampled = train_sample_weights[0:-1:skip]
    #
    # testX_sampled = testX[0:-1:skip]
    # testY_one_hot_sampled = Y_test_one_hot[0:-1:skip]
    # sw_test_sampled = test_sample_weights[0:-1:skip]
    #
    # validationX_sampled = testX[0:-1:skip]
    # validationY_one_hot_sampled = Y_validation_one_hot[0:-1:skip]
    # sw_validation_sampled = validation_sample_weights[0:-1:skip]
    #
    #
    # data_dim = np.shape(trainX)[1]
    # timesteps = 100
    # window = 100
    # batch_stride = 100
    # batch_size = 100
    #
    # sg_train = SeqGen(trainX_sampled,trainY_one_hot_sampled,batch_size,window,
    #                   batch_stride=batch_stride,sample_weights=sw_train_sampled)
    #
    # sg_test = SeqGen(testX_sampled,testY_one_hot_sampled,batch_size,window,
    #                   batch_stride=batch_stride, sample_weights=sw_test_sampled)
    #
    # sg_validation = SeqGen(validationX_sampled,validationY_one_hot_sampled,batch_size,window,
    #                   batch_stride=batch_stride, sample_weights=sw_validation_sampled)
    #
    # tb = TensorBoard(log_dir='./LSTM_wavelets', histogram_freq=0,
    #                             write_graph=True, write_images=True)
    #
    #
    #
    # model_LSTM.fit_generator(sg_train,validation_data=sg_validation,
    #                     epochs=epochs,callbacks=[tb])
    #
    # model_LSTM.save("./trained_models/LSTM_wavelet_100ep_5feat.hdf5")
    #
    # test_one_hot_predict_series = []
    # test_one_hot_truth_series = []
    #
    # for ii in range(len(sg_test)):
    #     test_one_hot_predict = model_LSTM.predict(sg_test[ii][0], batch_size=batch_size)
    #     test_one_hot_predict_df = pd.DataFrame(test_one_hot_predict, columns=label_names_df)
    #     test_one_hot_predict_series_batch = list(test_one_hot_predict_df.idxmax(axis=1))
    #     test_one_hot_predict_series = test_one_hot_predict_series + test_one_hot_predict_series_batch
    #
    #     test_one_hot_truth = (sg_test[ii][1] > 0.9).astype(int)
    #     test_one_hot_truth_df = pd.DataFrame(test_one_hot_truth, columns=label_names_df)
    #     test_one_hot_truth_series_batch = list(test_one_hot_truth_df.idxmax(axis=1))
    #     test_one_hot_truth_series = test_one_hot_truth_series + test_one_hot_truth_series_batch
    #
    # plt.figure(1)
    # ax = plt.subplot(2, 1, 1)
    #
    # plotResult_colorbars(test_one_hot_predict_series, range(len(test_one_hot_predict_series)),
    #                      labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    # ax.set_xlim([0, len(test_one_hot_predict_series)])
    # ax = plt.subplot(2, 1, 2)
    #
    # plotResult_colorbars(test_one_hot_truth_series, range(len(test_one_hot_truth_series)),
    #                      labelNames=list(label_names_df) + [''], medfiltwidth=1)
    # ax.set_xlim([0, len(test_one_hot_truth_series)])
    #
    # print set(test_one_hot_truth_series)
    # print set(test_one_hot_predict_series)
    #
    #
    # plt.show()