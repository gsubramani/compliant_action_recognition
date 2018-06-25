import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import label_map,relabel_data,get_signal_features,MLP_model,SeqGen,load_db,get_wavelet_features
from keras.callbacks import TensorBoard
from keras.models import load_model
import cPickle as pickle



if __name__ == "__main__":

    epochs = 100
    continue_training = False


    data = load_db('grab_objects', path='./')
    data = relabel_data(data, label_map)
    training_data = data[0:10]
    test_data = data[10:-2]


    features = (0,8)

    trainX_signal,trainY = get_signal_features(training_data)
    trainX_wavelet, _ = get_wavelet_features(training_data,features=features)
    trainX = np.append(trainX_signal,trainX_wavelet,axis = 1)

    train_sample_weights = compute_sample_weight('balanced', trainY)
    Y_train_one_hot = pd.get_dummies(trainY)

    testX_signal ,testY = get_signal_features(test_data)
    testX_wavelet, _ = get_wavelet_features(test_data,features=features)
    testX = np.append(testX_signal,testX_wavelet,axis = 1)


    test_sample_weights = compute_sample_weight('balanced', testY)
    Y_test_one_hot = pd.get_dummies(testY)



    label_names_df = list(Y_train_one_hot.columns.values)

    pickle.dump(label_names_df, open("label_names_df.pkl", "wb"))


    data_dim = np.shape(trainX)[1]
    num_classes = len(label_names_df)

    model = MLP_model(data_dim, num_classes)


    model.fit(trainX, Y_train_one_hot,
              sample_weight = train_sample_weights,
              epochs=epochs, batch_size=32,
              validation_data=(testX, Y_test_one_hot))

    model.save("./trained_models/wavelet_10ep_8feat.hdf5")

    test_predict_one_hot = model.predict(testX)

    predict_df = pd.DataFrame(test_predict_one_hot, columns=label_names_df)
    test_predict = list(predict_df.idxmax(axis=1))

    plt.figure(1)
    ax = plt.subplot(2, 1, 1)

    plotResult_colorbars(test_predict, range(len(test_predict)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    ax.set_xlim([0, len(test_predict)])
    ax = plt.subplot(2, 1, 2)

    plotResult_colorbars(testY, range(len(testY)),
                         labelNames=list(label_names_df) + [''], medfiltwidth=1)
    ax.set_xlim([0, len(testY)])

    # print set(test_one_hot_truth_series)
    # print set(test_one_hot_predict_series)


    plt.show()