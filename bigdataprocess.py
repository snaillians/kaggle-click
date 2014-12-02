__author__ = 'snail-apple'

import sys, os, time, pickle, shutil
import numpy as np
import pandas as pd
import click_utilities as util

training_file = "data/train.csv"
name=pd.read_csv(training_file,nrows=10,header=0)
map_column=[x for x in name.columns if x not in ["id","click","device_id","device_ip"]]
if not os.path.exists("data/single"):
    os.mkdir("data/single")

for col in map_column:
    single_file="data/single/"+col+".csv"
    single_map_file="data/single/"+col+".pickle"
    if not (os.path.exists(single_file) and os.path.exists(single_map_file)):
        single=pd.read_csv(training_file,header=0,usecols=["click",col])
        print "read in %d records of %s" %(single.shape[0],col)

        if col=="hour":
            single.loc[:,col]=single[col].astype(str).map(lambda x: x[-2:]).astype(np.int32)

        if os.path.exists(single_map_file):
            single_map=pickle.load(open(single_map_file,"rb"))
        else:
            single_map=util.generate_map(single)
            pickle.dump(single_map,open(single_map_file,"wb"))
            print "The map for %s is stored." %(col)
        if os.path.exists(single_file):
            pass
        else:
            single=util.remap_data(single,single_map)[col]
            single.to_csv(single_file,header=True,index=False,float_format="%5.4f")
            print col+" is processed and stored."

hash_file="data/single/device_id_ip.csv"
if not os.path.exists(hash_file):
    single=pd.read_csv(training_file,header=0,usecols=["device_id","device_ip"])
    print "read in %d records of %s" %(single.shape[0],"(device_id,device_ip)")
    single=single.apply(lambda t: t.map(lambda x: (abs(hash(x))%10**6)/float(10**6)))
    single.to_csv(hash_file,comments='', header=True,index=False,float_format="%5.4f")
    print "device_id,ip are processed and stored."

map_file="data/map_all.pickle"
if not os.path.exists(map_file):
    data_map={}
    for col in map_column:
        single_map_file="data/single/"+col+".pickle"
        single_map=pickle.load(open(single_map_file,"rb"))
        data_map.update(single_map)
    pickle.dump(data_map,open(map_file,"wb"))
    print "%d maps are combined and restored." %(len(data_map))


intermediate_file="data/intermediate_all.csv"
if not os.path.exists(intermediate_file):
    dataframe=pd.read_csv(training_file,header=0)
    for col in map_column:
        single_file="data/single/"+col+".csv"
        if os.path.exists(single_file):
            dataframe.loc[:,col]=pd.read_csv(single_file,header=0)
            print col+" is read in."
    dataframe.loc[:,["device_id","device_ip"]]=pd.read_csv(hash_file,header=0)
    print "device_id,device_ip are read in"
    dataframe.to_csv(intermediate_file,comments='',header=True,index=False,float_format="%5.4f")
    print "intermediate file is constructed."

shutil.rmtree("data/single")
