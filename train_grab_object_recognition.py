from compliant_action_recognition import CompliantActionRecognition, meta_data
from time_series_learning import relabel_data,get_signal_features,LSTM_model,SeqGen,\
    load_db,get_wavelet_features,sortify,upsample_labels
import matplotlib.pyplot as plt
from plot_generator import plotResult_colorbars
if __name__ == "__main__":

    data = load_db('grab_objects', path='./')
    car = CompliantActionRecognition(data)
    car.train_LSTM_wavelet(range(0,len(data)),epochs = 100,batch_size=100,batch_stride=100)
    car.train_LSTM(range(0,len(data)),epochs = 100,batch_size=100,batch_stride=100)
    car.save_trained_models(folder = "./trained_models/grab_objects_batch_stride100/")
