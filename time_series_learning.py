
from copy import deepcopy

from label_manipulation import fill_closest
from SignalRecognition.cwt_learner.wavelet_feature_engineering import CWT_learner
import tensorflow as tf
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import Sequence
import numpy as np
from signal_database.labelSelectedData import SignalDB, SignalBundle



def load_db(db_name,path = './'):

    sdb = SignalDB(db_name, path=path)
    training_data_ = sdb.get_labeleddata()
    return training_data_

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


def relabel_data(training_data_,label_map):
    training_data_relabeled = deepcopy(training_data_)
    for example in training_data_relabeled:
        example.labels = [label_map[label] for label in example.labels]
    return training_data_relabeled



def get_wavelet_features(training_data_relabeled,signal_indices = (0, 1, 2, 3),features = (0,10)):
    cwt_learn_training = CWT_learner(signal_indices=list(signal_indices))
    training_data = training_data_relabeled
    for ld in training_data:
        labels = [label for label in ld.labels]
        signals = []
        signals.append(list(np.array(ld.signal_bundle.signals[0]) + np.array(ld.signal_bundle.signals[1])))
        signals.append(list(ld.signal_bundle.signals[2]))

        signals.append(list(np.array(ld.signal_bundle.signals[3]) + np.array(ld.signal_bundle.signals[4])))
        signals.append(list(ld.signal_bundle.signals[5]))

        labels = ld.labels
        try:
            labels = fill_closest(labels)
        except:
            print(" nothing in labels")


        cwt_learn_training.add_training_data(signals, labels)



    (trainXCWT, trainYCWT) = cwt_learn_training.get_examples_with_weights(cwt_learn_training.training_data_sets,
                                                                          with_window="Off",
                                                                          signal_indices=cwt_learn_training.signal_indices
                                                                          , wavelet=cwt_learn_training.wavelet,
                                                                          features=features)

    return trainXCWT,trainYCWT



def sortify(labels,timestamps):
    indices = np.argsort(timestamps)
    return [labels[ind] for ind in indices]

def get_signal_features(training_data_relabeled,signal_indices = (0, 1, 2, 3)):
    training_data = training_data_relabeled
    signal_train_X = np.array([[]]).reshape(0,len(signal_indices))
    signal_train_Y = np.array([])
    for ld in training_data:

        signals = []
        signals.append(list(np.array(ld.signal_bundle.signals[0]) + np.array(ld.signal_bundle.signals[1])))
        signals.append(ld.signal_bundle.signals[2])
        signals.append(list(np.array(ld.signal_bundle.signals[3]) + np.array(ld.signal_bundle.signals[4])))
        signals.append(ld.signal_bundle.signals[5])

        labels = ld.labels
        try:
            labels = fill_closest(labels)
        except:
            print(" nothing in labels")

        signals = np.transpose(signals)

        signal_train_X = np.append(signal_train_X, signals, axis=0)
        signal_train_Y = np.append(signal_train_Y, labels)

    print np.shape(signal_train_X),np.shape(signal_train_Y)

    return signal_train_X,signal_train_Y



def LSTM_model(data_dim, timesteps, batch_size, num_classes, stateful=False,
               sample_weight_mode=None):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(50, return_sequences=True,
                        input_shape=(timesteps, data_dim), batch_size=batch_size
                        , stateful=stateful))  # returns a sequence of vectors of dimension 32
    model_LSTM.add(LSTM(50, stateful=stateful))  # return a single vector of dimension 32
    model_LSTM.add(Dense(50, activation='relu'))
    # model_LSTM.add(Dense(10, activation='relu'))
    model_LSTM.add(Dense(num_classes, activation='softmax'))
    model_LSTM.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'],
                       sample_weight_mode=sample_weight_mode)
    return model_LSTM


class SeqGen(Sequence):
    def __init__(self, x_set, y_set, batch_size, window_size, batch_stride=1,
                 sample_weights=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.window_size = window_size
        self.batch_stride = batch_stride
        self.sw = sample_weights

    def __len__(self):
        return len(self.x) // self.batch_stride

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        current_x = np.roll(self.x, idx * self.batch_stride, axis=0)
        current_y = np.roll(self.y, idx * self.batch_stride, axis=0)

        for ii in range(self.batch_size):
            batch_x.append(np.roll(current_x, ii, axis=0)[:self.window_size])
            batch_y.append(np.roll(current_y, ii, axis=0)[:self.window_size][-1])

        if self.sw is None:
            return np.array(batch_x), np.array(batch_y)

        else:
            batch_sw = []
            current_sw = np.roll(self.sw, idx * self.batch_stride, axis=0)
            for ii in range(self.batch_size):
                batch_sw.append(np.roll(current_sw, ii, axis=0)[:self.window_size][-1])

            return np.array(batch_x), np.array(batch_y), np.array(batch_sw)