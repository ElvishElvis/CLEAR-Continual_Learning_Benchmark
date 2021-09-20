import os
import numpy as np
import argparse

def get_offline_protocol_index(class_=10):
    eval_list={'offline':[],'online':[],'backward':[],'forward':[],'accuracy':[]}
    count=0
    for i in range(class_):
        for k in range(class_):
            if(i==k):
                eval_list['offline'].append(count)
                eval_list['accuracy'].append(count)
            if(i+1==k):
                eval_list['online'].append(count)
            if(i>k):
                eval_list['backward'].append(count)
                eval_list['accuracy'].append(count)
            if(i<k):
                eval_list['forward'].append(count)
                
            count=count+1
    assert len(eval_list['offline'])==class_
    assert len(eval_list['online'])==class_-1
    assert len(eval_list['backward'])==int(class_*(class_-1)/2)
    assert len(eval_list['forward'])==int(class_*(class_-1)/2)
    assert len(eval_list['accuracy'])==int(class_*(class_+1)/2)
    return eval_list

def get_online_protocol_index(class_=10):
    eval_list={'online':[],'forward':[]}
    # to match our script, since 0-10 is 0
    count=class_
    for i in range(class_):
        for k in range(class_):
            if(i+1==k):
                eval_list['online'].append(count)
            if(i<k):
                eval_list['forward'].append(count)
                
            count=count+1
    assert len(eval_list['online'])==class_-1
    assert len(eval_list['forward'])==int(class_*(class_-1)/2)
    return eval_list

argparser = argparse.ArgumentParser()
argparser.add_argument("--split")
argparser.add_argument("--nclass",type=int)

args = argparser.parse_args()

logpath='../{}/log/'.format(args.split)
log_list=sorted(os.listdir(logpath))

for name in log_list:
    result_list=[]
    file=open(os.path.join(logpath,name), 'r')
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
    if(len(result_list)!=int(args.nclass*args.nclass)):
        if('online' in name):
            # assert np.max(result_list[:args.nclass])<0.3
            result_list=result_list[args.nclass:]
        print("{} count of {}, with mean of {}".format(name,len(result_list), np.mean(result_list)))
    else:
        result_list=np.array(result_list)
        if('online' in name):
            # assert np.max(result_list[:args.nclass])<0.3
            index_list=get_online_protocol_index(class_=args.nclass)
        else:
            index_list=get_offline_protocol_index(class_=args.nclass)
        result_list=[str(np.mean(result_list[np.array(item[1])])) for item in index_list.items()]
        key_list=[item[0] for item in index_list.items()]
        print("{} with {} of {}".format(name,", ".join(key_list),", ".join(result_list)))
        



# import matplotlib.pyplot as plt
# def plot_2d_matrix(matrix, label_ticks, title, save_name,min_acc=None, max_acc=None, normalize_by_column=False, plotting_trick=True):
#     if plotting_trick:
#         matrix_score = matrix.copy()
#         matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
#         cbar_str = "Test Accuracy"
#         format_score = lambda s : f"{s:.2f}"
#    elif normalize_by_column:
#         matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
#         cbar_str = "Test Accuracy / Average Test accuracy per column"
#         format_score = lambda s : f"{s:.2f}"
#         matrix_score = matrix
#     else:
#         cbar_str = "Test Accuracy"
#         format_score = lambda s : f"{s:.2%}"
#         matrix_score = matrix
    
#     plt.figure(figsize=(10,10))
#     x = ["Test " + n for n in label_ticks]
#     y = ['Train ' + n for n in label_ticks]
#     p = plt.imshow(matrix, interpolation='none', cmap=f'Blues', vmin=min_acc, vmax=max_acc)
#     cbar = plt.colorbar()
    
   
#     cbar.ax.set_ylabel(f"{cbar_str}", rotation=-90, va="bottom")
#     cbar.ax.set_ylim(bottom=min_acc, top=max_acc)
# #         cbar.ax.invert_yaxis()

#     plt.xticks(range(len(x)), x, fontsize=11, rotation = -90)
#     plt.yticks(range(len(y)), y, fontsize=11)
#     plt.title(title, fontsize=15)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             text = plt.text(j, i, format_score(matrix_score[i, j]),
#                        ha="center", va="center", color="black")
#     plt.savefig('./{}.png'.format(save_name))
# plot_2d_matrix(np.random.random((10,10)), [str(i) for i in range(1,11)], '','kkk',normalize_by_column=False,
#                        plotting_trick=False)
