from signal_database.labelSelectedData import SignalDB,SignalBundle,LabeledData
from bagfile_io.bagfile_reader import bagfile_reader,write_to_bagfile
import argparse
import numpy as np

import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import relabel_data,get_signal_features,LSTM_model,SeqGen,\
    load_db,get_wavelet_features,sortify,upsample_labels

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


class meta_data:
    def __init__(self):
        self.LSTM_wavelet_trained = False
        self.LSTM_trained = False

class CompliantActionRecognition():


    def __init__(self,labeled_data_sets, label_map_in = None,label_names = None,meta = "",folder = None):
        self.md = meta_data()
        self.data = labeled_data_sets
        self.folder = None

        if label_map_in is None:   self.label_map = label_map
        else:                   self.label_map = label_map_in

        if label_names is None: self.md.label_names_df = None
        else: self.md.label_names_df = label_names

        self.data = relabel_data(data, self.label_map)

        self.md.LSTM_wavelet_trained = False
        self.md.LSTM_trained = False

        self.wavelet_tf_train_completed = False
        self.wavelet_tf_test_completed = False
        self.wavelet_tf_validation_completed = False

        self.meta = meta


    def train_LSTM_wavelet(self,ids,valid_ids = None,
                           batch_size = 1,epochs = 30,skip = 100,features = (0,5),
                           batch_stride = 1,window = 100,timesteps = 100):

        self.md.LSTM_wavelet_batch_size = batch_size
        self.md.LSTM_wavelet_timesteps = timesteps
        self.md.LSTM_wavelet_epochs = epochs
        self.md.LSTM_wavelet_skip = skip
        self.md.LSTM_wavelet_features = features
        self.md.LSTM_wavelet_batch_stride = batch_stride
        self.md.LSTM_wavelet_window = window

        training_data = [data[id] for id in ids]
        trainX_signal,trainY = get_signal_features(training_data)

        if not self.wavelet_tf_train_completed:
            self.trainX_wavelet, _ = get_wavelet_features(training_data,features=features)


        trainX = np.append(trainX_signal,self.trainX_wavelet,axis = 1)

        train_sample_weights = compute_sample_weight('balanced', trainY)
        Y_train_one_hot = pd.get_dummies(trainY)

        if self.md.label_names_df is None: self.md.label_names_df = list(Y_train_one_hot.columns.values)

        data_dim = np.shape(trainX)[1]

        self.md.LSTM_wavelet_data_dim = data_dim


        num_classes = len(self.md.label_names_df)

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

        self.md.LSTM_wavelet_trained = True
        print "Trained LSTM + wavelets model!"


    def train_LSTM(self,ids,valid_ids = None,
                           batch_size = 1,epochs = 30,skip = 100,features = (0,5),
                           batch_stride = 1,window = 100,timesteps = 100):

        self.md.LSTM_batch_size = batch_size
        self.md.LSTM_timesteps = timesteps
        self.md.LSTM_epochs = epochs
        self.md.LSTM_skip = skip
        self.md.LSTM_features = features
        self.md.LSTM_batch_stride = batch_stride
        self.md.LSTM_window = window

        training_data = [data[id] for id in ids]
        trainX_signal,trainY = get_signal_features(training_data)

        trainX = trainX_signal

        train_sample_weights = compute_sample_weight('balanced', trainY)
        Y_train_one_hot = pd.get_dummies(trainY)

        if self.md.label_names_df is None: self.md.label_names_df = list(Y_train_one_hot.columns.values)

        data_dim = np.shape(trainX)[1]

        self.md.LSTM_data_dim = data_dim

        num_classes = len(self.md.label_names_df)

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

        self.md.LSTM_trained = True
        print "Trained LSTM model!"

    # def predict_LSTM(self,ids):
    #     test_data = [data[id] for id in ids]

    def predict_LSTM_wavelet(self,id):
        test_data = [data[id]]
        X_signal,_ = get_signal_features(test_data)
        X_wavelet, _ = get_wavelet_features(test_data,features=self.md.LSTM_wavelet_features)
        X = np.append(X_signal,X_wavelet,axis = 1)
        skip = self.md.LSTM_wavelet_skip
        timestamps = data[id].signal_bundle.timestamps
        sg = SeqGen(X[0:-1:skip], timestamps[0:-1:skip], 1, self.md.LSTM_wavelet_window,
                    batch_stride=self.md.LSTM_wavelet_batch_stride)

        predictions = np.array([]).reshape(0,len(self.md.label_names_df))
        predictions_timestamps = np.array([])
        for ii in range(len(sg)):
            predictions = np.append(predictions,
                                    self.model_LSTM_wavelet.predict(sg[ii][0],
                                                                batch_size=self.md.LSTM_wavelet_batch_size),
                                    axis = 0)
            predictions_timestamps = np.append(predictions_timestamps,sg[ii][1],axis = 0)

        predictions_wavelets = sortify(predictions, predictions_timestamps.tolist())

        predict_df = pd.DataFrame(predictions, columns=self.md.label_names_df)
        predict_list = list(predict_df.idxmax(axis=1))

        predict_list_sorted = sortify(predict_list, predictions_timestamps.tolist())
        predict_list_sorted_wavelet = upsample_labels(predict_list_sorted, predictions_timestamps, timestamps)
        predictions_wavelets_upsampled = upsample_labels(predictions_wavelets, predictions_timestamps, timestamps)

        return predict_list_sorted_wavelet,predictions_wavelets_upsampled

    def predict_LSTM(self,id):
        test_data = [self.data[id]]
        X_signal, Y = get_signal_features(test_data)
        X = X_signal

        skip = self.md.LSTM_skip
        timestamps = data[id].signal_bundle.timestamps
        window = self.md.LSTM_window
        batch_stride = self.md.LSTM_batch_stride
        batch_size = self.md.LSTM_batch_size
        sg = SeqGen(X[0:-1:skip], timestamps[0:-1:skip], 1, window, batch_stride=batch_stride)

        predictions = np.array([]).reshape(0, len(self.md.label_names_df))
        predictions_timestamps = np.array([])

        for ii in range(len(sg)):
            predictions = np.append(predictions, self.model_LSTM.predict(sg[ii][0], batch_size=batch_size), axis=0)
            predictions_timestamps = np.append(predictions_timestamps, sg[ii][1], axis=0)

        predictions_LSTM_only = sortify(predictions, predictions_timestamps.tolist())

        predict_df = pd.DataFrame(predictions, columns=self.md.label_names_df)
        predict_list = list(predict_df.idxmax(axis=1))

        predict_list_sorted = sortify(predict_list, predictions_timestamps.tolist())

        predict_list_sorted_LSTM = upsample_labels(predict_list_sorted, predictions_timestamps, timestamps)
        predictions_LSTM_upsampled = upsample_labels(predictions_LSTM_only, predictions_timestamps, timestamps)

        return predict_list_sorted_LSTM,predictions_LSTM_upsampled


    def predict_combined(self,id):
        predict_list_sorted_wavelet, predictions_wavelets_upsampled = self.predict_LSTM_wavelet(id)
        predict_list_sorted_LSTM, predictions_LSTM_upsampled = self.predict_LSTM(id)

        predictions_combined = np.array(predictions_wavelets_upsampled) * np.array(predictions_LSTM_upsampled)
        predict_df = pd.DataFrame(predictions_combined, columns=self.md.label_names_df)
        predict_list_combined = list(predict_df.idxmax(axis=1))

        return predict_list_combined


    def load_from_folder(self,folder):
        self.md = pickle.load( open( folder + "/meta_data.pkl", "rb" ) )
        if self.md.LSTM_wavelet_trained:
            self.trainX_wavelet = pickle.load(open(folder + "/trainX_wavelet.pkl", "rb"))

            model_LSTM_wavelet_saved = load_model(folder + "/LSTM_wavelet" + ".hdf5")
            num_classes = len(self.md.label_names_df)
            self.model_LSTM_wavelet = LSTM_model(self.md.LSTM_wavelet_data_dim,
                                         self.md.LSTM_wavelet_window, self.md.LSTM_wavelet_batch_size, num_classes)
            self.model_LSTM_wavelet.set_weights(model_LSTM_wavelet_saved.get_weights())

        if self.md.LSTM_trained:

            model_LSTM_saved = load_model(folder + "/LSTM" + ".hdf5")

            num_classes = len(self.md.label_names_df)
            self.model_LSTM = LSTM_model(self.md.LSTM_data_dim,
                                         self.md.LSTM_window, self.md.LSTM_batch_size, num_classes)
            self.model_LSTM.set_weights(model_LSTM_saved.get_weights())



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
        if self.md.LSTM_wavelet_trained:
            self.model_LSTM_wavelet.save(folder + "/LSTM_wavelet" + ".hdf5")
            pickle.dump(self.trainX_wavelet, open(folder + "/trainX_wavelet.pkl", "wb"))

        if self.md.LSTM_trained:
            self.model_LSTM.save(folder + "/LSTM" + ".hdf5")

        pickle.dump(self.md,open( folder + "/meta_data.pkl", "wb" ))


if __name__ == "__main__":


    data = load_db('grab_objects', path='./')
    car = CompliantActionRecognition(data)
    car.train_LSTM_wavelet([0],epochs = 1)
    car.train_LSTM([0],epochs = 1)
    car.save_trained_models(folder = "./trained_models/test/")
    car.load_from_folder(folder = "./trained_models/test/")
    print car.predict_combined(0)
