import os
import numpy as np
import argparse
import glob
from parse_log_to_result import *
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_2d_matrix(matrix, label_ticks, title, save_name,save_path='../plot_avalanche',min_acc=None, max_acc=None, normalize_by_column=False, plotting_trick=False):
    if plotting_trick:
        matrix_score = matrix.copy()
        matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
        cbar_str = "Test Accuracy"
        format_score = lambda s : f"{s:.2f}"
    elif normalize_by_column:
        matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
        cbar_str = "Test Accuracy / Average Test accuracy per column"
        format_score = lambda s : f"{s:.2f}"
        matrix_score = matrix
    else:
        cbar_str = "Test Accuracy"
        format_score = lambda s : f"{s:.2%}"
        matrix_score = matrix

    plt.figure(figsize=(10,10))
    x = ["Test " + n for n in label_ticks]
    y = ['Train ' + n for n in label_ticks]
    p = plt.imshow(matrix, interpolation='none', cmap=f'Blues', vmin=min_acc, vmax=max_acc)
    # cbar = plt.colorbar()


    # cbar.ax.set_ylabel(f"{cbar_str}", rotation=-90, va="bottom")
    # cbar.ax.set_ylim(bottom=min_acc, top=max_acc)
#         cbar.ax.invert_yaxis()

    plt.xticks(range(len(x)), x, fontsize=11, rotation = -90)
    plt.yticks(range(len(y)), y, fontsize=11)
    plt.title(title, fontsize=15)
    for i in range(len(x)):
        for j in range(len(y)):
            text = plt.text(j, i, format_score(matrix_score[i, j]),
                       ha="center", va="center", color="black")
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(os.path.join(save_path,'{}.png'.format(save_name)))

argparser = argparse.ArgumentParser()
argparser.add_argument("--timestamp",type=int,default=10)
argparser.add_argument("--plot",type=int,default=0) # 1 for generating plot
argparser.add_argument("--verbose",type=int,default=0) # 1 for print out detailed metric
args = argparser.parse_args()


logpath='/data/jiashi/metric'
log_list=sorted(os.listdir(logpath))
unique_name=sorted(list(set(list(map(lambda log_name: log_name[:log_name.index('line')+4],log_list)))))

# each unique prefix
for name in unique_name:
    all_name=glob.glob(os.path.join(logpath,name+"*"))
    stat_list=[]
    plot_list=[]
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
            plot_list.append(result_list.reshape((args.timestamp,args.timestamp)))
            result_list=[str(np.mean(result_list[np.array(item[1])])) for item in index_list.items()]
            key_list=[item[0] for item in index_list.items()]
            stat_list.append(result_list)
    try:
        stat_list=np.array(stat_list).astype(float)
        stat_list=np.unique(stat_list,axis=0)
        if(args.verbose==1):
            print(stat_list)
        print("{} with {} of mean of {} std of {} with {} elem".format(name,", ".join(key_list),np.mean(stat_list,axis=0).tolist(),np.std(stat_list,axis=0),len(all_name)))
    except:
        print('---------------------------------------')
        print('skip {}'.format(name))
        print('---------------------------------------')
    if(args.plot==1 and 'online' not in name and 'Joint' not in name):
        try:
            plot_array=np.mean(np.array(plot_list),axis=0)
            plot_2d_matrix(plot_array, [str(i) for i in range(1,11)], '',name,normalize_by_column=False,
                    plotting_trick=False)
        except:
            print('---------------------------------------')
            print('Plot skipped {}'.format(name))
            print('---------------------------------------')
            




