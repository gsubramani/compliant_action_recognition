from compliant_action_recognition import CompliantActionRecognition
from time_series_learning import  bag_file_to_labeled_data

import os


if __name__ == "__main__":
    folder_location = "/home/guru/Dropbox/lab-hp/workspace/research/end_to_end/data"+ \
        "/july26/labeled_bagfiles"

    # Mount a RAMDISK drive at /media/fast/ with: sudo mount -t tmpfs -o size=16000M tmpfs /media/fast/
    # Need to mount this every time
    folder_location = "/media/fast/"

    file_names = os.listdir(folder_location)


    labeled_data_set_test2 = []
    for ii,file_name in enumerate(file_names):
        # if ii > 1:
        #     break
        ld = bag_file_to_labeled_data(folder_location + file_name,label_topic="/labels")
        labeled_data_set_test2.append(ld)
        print ii

    label_map = {
        "grab": "grab",
        "hold": "hold",
        "release": "release",
        "still": "still",
        "": ""
    }

    car = CompliantActionRecognition(labeled_data_set_test2, label_map_in=label_map)
    car.train_LSTM_wavelet(range(0, len(labeled_data_set_test2)), epochs=100, batch_size=100, batch_stride=100)
    car.train_LSTM(range(0, len(labeled_data_set_test2)), epochs=100, batch_size=100, batch_stride=100)
    car.save_trained_models(folder="./trained_models/train_grab_constraints_july27/")