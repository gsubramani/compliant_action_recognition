from compliant_action_recognition import CompliantActionRecognition
from time_series_learning import bag_file_to_labeled_data
import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars,plot_legend

label_map = {
        "grab": "grab",
        "hold": "hold",
        "release": "release",
        "still": "still",
        "": ""
    }



if __name__ == "__main__":

    # datum = bag_file_to_labeled_data("../data/july26/openDrawer1.bag")
    datum = bag_file_to_labeled_data("/media/fast/openDrawer2.bag")
    #
    car = CompliantActionRecognition([datum],label_map_in = label_map)
    car.load_from_folder("./trained_models/train_grab_constraints_july27/")



    predictions,_ = car.predict_LSTM_wavelet(0)

    plt.figure(1)
    ax = plt.subplot(5,1,1)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    predictions, _ = car.predict_LSTM(0)

    ax = plt.subplot(5,1,2)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    predictions = car.predict_combined(0)
    ax = plt.subplot(5,1,3)
    plotResult_colorbars(predictions, range(len(predictions)),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    ax = plt.subplot(5, 1, 4)
    plotResult_colorbars(datum.labels , range(len(datum.labels )),
                         labelNames=list(car.md.label_names_df) + [''], ax=ax, medfiltwidth=1)

    ax = plt.subplot(5,1,5)
    plot_legend(car.md.label_names_df,ax)
    plt.show()










