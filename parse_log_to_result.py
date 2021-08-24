import os
import numpy as np
logpath='./log/'
log_list=os.listdir(logpath)
result_list=[]
for name in log_list:
    with open(os.path.join(logpath,name), 'r') as file:
        while True:
            try:
                line=file.readline()
                if('Top1_Acc_Stream/eval_phase/test_stream/Task00' in line):
                    result_list.append(float(line.split()[-1]))
            except:
                break
            
    print("{} have count {}, with mean of {}".format(name,len(result_list),np.mean(result_list))



# import matplotlib.pyplot as plt
# def plot_2d_matrix(matrix, label_ticks, title, save_name,min_acc=None, max_acc=None, normalize_by_column=False, plotting_trick=True):
#     if plotting_trick:
#         matrix_score = matrix.copy()
#         matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
#         cbar_str = "Test Accuracy"
#         format_score = lambda s : f"{s:.2f}"
#     elif normalize_by_column:
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
