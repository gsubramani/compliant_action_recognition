
from signal_database.labelSelectedData import SignalDB,SignalBundle,LabeledData
from bagfile_io.bagfile_reader import bagfile_reader
import argparse
import numpy as np

import os

__doc__ = "Adds bag files to the database. Used to create training data for detecting compliant actions "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Provide the bagfile path")
    parser.add_argument("destination", help="The location of the database")
    parser.add_argument("-w", "--write_permission")
    parser.add_argument("-f",action='store_true')
    args = parser.parse_args()

    if args.f == True:
        files = [args.bagfile + "/" + file for file in os.listdir(args.bagfile)]
        print files
    else:
        files = [args.bagfile]

    for bagfile in files:
        print "processing bagfile: " + bagfile
        destination_db = args.destination

        sdb = SignalDB(destination_db)
        bfr = bagfile_reader(bagfile)


        wrench1_,wrencht1 = bfr.get_topic_msgs("/ftmini40")
        wrench2_,wrencht2 = bfr.get_topic_msgs("/ftmini402")
        labels_,labelst = bfr.get_topic_msgs("/labels")

        timesamples = labelst

        labels = [label_messsage.data for label_messsage in labels_]

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

        #getting label names
        ld_out = sdb.findld(ld) #locates signal in database
        if ld_out != None:
            ld = ld_out

        label_names = list(set(list(ld.labels)))

        ld.labels = labels

        if args.write_permission == True:
            sdb.add_labeleddata(ld,overwrite=True)
        else:
            sdb.add_labeleddata(ld)

        sdb.commit()



        rec = [r for r in sdb.db_sig]
        print rec[0]['labels']
        print len(rec)

