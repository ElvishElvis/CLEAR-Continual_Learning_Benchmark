import os
import numpy as np
import argparse
import glob
from parse_log_to_result import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--timestamp",type=int,default=10)
args = argparser.parse_args()


logpath='/data/jiashi/metric'
log_list=sorted(os.listdir(logpath))
unique_name=sorted(list(set(list(map(lambda log_name: log_name[:log_name.index('line')+4],log_list)))))

# each unique prefix
for name in unique_name:
    all_name=glob.glob(os.path.join(logpath,name+"*"))
    stat_list=[]
    # for all element in same prefix 
    for elemt in all_name:
        result_list=[]
        log_file_name=os.path.join(logpath,elemt)
        file=open(log_file_name, 'r')
        while(True):
            try:
                line=file.readline()
            except:
                break
            if('Top1_Acc_Stream/eval_phase/test_stream/Task00' in line):
                result_list.append(float(line.split()[-1]))
            if not line:
                break
        file.close()
        if(len(result_list)!=int(args.timestamp*args.timestamp)):
            if('online' in name):
                # assert np.max(result_list[:args.timestamp])<0.3
                result_list=result_list[args.timestamp:]
            stat_list.append(np.mean(result_list))
            # print("{} count of {}, with mean of {}".format(name,len(result_list), np.mean(result_list)))
        else:
            result_list=np.array(result_list)
            if('online' in name):
                # assert np.max(result_list[:args.timestamp])<0.3
                index_list=get_online_protocol_index(class_=args.timestamp)
            else:
                index_list=get_offline_protocol_index(class_=args.timestamp)
            result_list=[str(np.mean(result_list[np.array(item[1])])) for item in index_list.items()]
            key_list=[item[0] for item in index_list.items()]
            stat_list.append(result_list)
    try:
        stat_list=np.array(stat_list).astype(float)
        stat_list=np.unique(stat_list,axis=0)
        print(stat_list)
        print("{} with {} of mean of {} std of {} with {} elem".format(name,", ".join(key_list),np.mean(stat_list,axis=0).tolist(),np.std(stat_list,axis=0),len(all_name)))
    except:
        print('---------------------------------------')
        print('skip {}'.format(name))
        print('---------------------------------------')
            