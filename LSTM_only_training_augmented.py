import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from time_series_learning import label_map,relabel_data,get_signal_features,LSTM_model,SeqGen,load_db
from keras.callbacks import TensorBoard
from keras.models import load_model
import argparse
import cPickle as pickle



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('epochs')

    epochs = 100
    continue_training = False

    data = load_db('grab_objects', path='./')
    data = relabel_data(data, label_map)
    training_data = data[0:10]
    test_data = data[10:-2]
    validation_data = data[-2:]


    trainX,trainY = get_signal_features(training_data)
    train_sample_weights = compute_sample_weight('balanced', trainY)
    Y_train_one_hot = pd.get_dummies(trainY)


    trainX = np.roll(np.array(Y_train_one_hot),1,axis=0)

    # print np.roll(np.array(Y_train_one_hot),1,axis = 1)
    # exit()

    testX ,testY = get_signal_features(test_data)
    test_sample_weights = compute_sample_weight('balanced', testY)
    Y_test_one_hot = pd.get_dummies(testY)

    testX = np.roll(np.array(Y_test_one_hot),1,axis=0)


    validationX, validationY = get_signal_features(validation_data)
    validation_sample_weights = compute_sample_weight('balanced', validationY)
    Y_validation_one_hot = pd.get_dummies(validationY)

    validationX =  np.roll(np.array(Y_validation_one_hot),1,axis=0)

    label_names_df = list(Y_train_one_hot.columns.values)

    pickle.dump(label_names_df, open("label_names_df.pkl", "wb"))

    data_dim = np.shape(trainX)[1]
    timesteps = 100
    batch_size = 100
    num_classes = len(label_names_df)

    model_LSTM = LSTM_model(data_dim, timesteps, batch_size, num_classes, stateful=False)

    # if continue_training:
    #     try:
    #         model_LSTM  = load_model("./trained_models/LSTM_only.hdf5")
    #         print "loading model"
    #     except:
    #         "model doesn't exist!"
    #         exit()

    skip = 100
    trainX_sampled = trainX[0:-1:skip]
    trainY_one_hot_sampled = Y_train_one_hot[0:-1:skip]
    sw_train_sampled = train_sample_weights[0:-1:skip]

    testX_sampled = testX[0:-1:skip]
    testY_one_hot_sampled = Y_test_one_hot[0:-1:skip]
    sw_test_sampled = test_sample_weights[0:-1:skip]

    validationX_sampled = testX[0:-1:skip]
    validationY_one_hot_sampled = Y_validation_one_hot[0:-1:skip]
    sw_validation_sampled = validation_sample_weights[0:-1:skip]


    data_dim = np.shape(trainX)[1]
    timesteps = 100
    window = 100
    batch_stride = 100
    batch_size = 100

    sg_train = SeqGen(trainX_sampled,trainY_one_hot_sampled,batch_size,window,
                      batch_stride=batch_stride,sample_weights=sw_train_sampled)

    sg_test = SeqGen(testX_sampled,testY_one_hot_sampled,batch_size,window,
                      batch_stride=batch_stride, sample_weights=sw_test_sampled)

    sg_validation = SeqGen(validationX_sampled,validationY_one_hot_sampled,batch_size,window,
                      batch_stride=batch_stride, sample_weights=sw_validation_sampled)


    tb = TensorBoard(log_dir='./LSTM_only_augmented', histogram_freq=0,
                                write_graph=True, write_images=True)

    model_LSTM.fit_generator(sg_train,validation_data=sg_validation,
                        epochs=epochs,callbacks=[tb])


    model_LSTM.save("./trained_models/LSTM_only_augmented.hdf5")

    test_one_hot_predict_series = []
    test_one_hot_truth_series = []

    for ii in range(len(sg_test)):
        test_one_hot_predict = model_LSTM.predict(sg_test[ii][0], batch_size=batch_size)
        test_one_hot_predict_df = pd.DataFrame(test_one_hot_predict, columns=label_names_df)
        test_one_hot_predict_series_batch = list(test_one_hot_predict_df.idxmax(axis=1))
        test_one_hot_predict_series = test_one_hot_predict_series + test_one_hot_predict_series_batch

        test_one_hot_truth = (sg_test[ii][1] > 0.9).astype(int)
        test_one_hot_truth_df = pd.DataFrame(test_one_hot_truth, columns=label_names_df)
        test_one_hot_truth_series_batch = list(test_one_hot_truth_df.idxmax(axis=1))
        test_one_hot_truth_series = test_one_hot_truth_series + test_one_hot_truth_series_batch

    plt.figure(1)
    ax = plt.subplot(2, 1, 1)

    plotResult_colorbars(test_one_hot_predict_series, range(len(test_one_hot_predict_series)),
                         labelNames=list(label_names_df) + [''], ax=ax, medfiltwidth=1)
    ax.set_xlim([0, len(test_one_hot_predict_series)])
    ax = plt.subplot(2, 1, 2)

    plotResult_colorbars(test_one_hot_truth_series, range(len(test_one_hot_truth_series)),
                         labelNames=list(label_names_df) + [''], medfiltwidth=1)
    ax.set_xlim([0, len(test_one_hot_truth_series)])

    print set(test_one_hot_truth_series)
    print set(test_one_hot_predict_series)


    plt.show()

