
from compliant_action_recognition import CompliantActionRecognition, meta_data
from time_series_learning import relabel_data,get_signal_features,LSTM_model,SeqGen,\
    load_db,get_wavelet_features,sortify,upsample_labels,bag_file_to_labeled_data
import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars,plot_legend


if __name__ == "__main__":

    datum = bag_file_to_labeled_data("../data/july26/openDrawer1.bag")
    #
    car = CompliantActionRecognition([datum])
    car.load_from_folder("./trained_models/grab_objects_batch_stride100/")
    predictions,_ = car.predict_LSTM_wavelet(0)
    # predictions = car.predict_combined(0)
    plt.figure(1)
    ax = plt.subplot(4,1,1)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    predictions, _ = car.predict_LSTM(0)

    ax = plt.subplot(4,1,2)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    predictions = car.predict_combined(0)

    ax = plt.subplot(4,1,3)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    ax = plt.subplot(4,1,4)
    plot_legend(car.md.label_names_df,ax)
    plt.show()










