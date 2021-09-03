
import os
import os.path as osp
from sklearn.model_selection import train_test_split
class_list=['NEGATIVE','baseball','bus','camera','cosplay','dress','hockey','laptop','racing','soccer','sweater']


data_dir = '/data3/zhiqiul/yfcc_dynamic_10/dynamic_300/images'      
# data_dir ='/scratch/zhiqiu/yfcc_dynamic_10/dynamic_300/images'


# def list_all_files(rootdir):
#     _files = []
#     list_file = os.listdir(rootdir)
#     for i in range(0,len(list_file)):
#         path = os.path.join(rootdir,list_file[i])

#         if os.path.isdir(path):
#             _files.extend(list_all_files(path))
#         if os.path.isfile(path):
#              _files.append(path)
#     return _files

def list_all_files(rootdir):
    train_list,test_list,all_list = [],[],[]
    bucket_list = os.listdir(rootdir)
    classes_list=  os.listdir(osp.join(rootdir,bucket_list[0]))
    for bucket in bucket_list:
        for classes in classes_list:
            image_list=os.listdir(osp.join(rootdir,bucket,classes))
            image_list=list(map(lambda a: osp.join(osp.join(rootdir,bucket,classes,a)), image_list))
            train_subset,test_subset=train_test_split(image_list,test_size=0.3, random_state=43)
            train_list.extend(train_subset)
            test_list.extend(test_subset)
            all_list.extend(image_list)
    return train_list,test_list,all_list

train_list, test_list,all_list= list_all_files(data_dir)

for stage in ['train','test','all']:
    if(stage=='train'):
        image_list=train_list
    elif(stage=='test'):
        image_list=test_list
    else:
        image_list=all_list

    with open('/data/jiashi/data_{}_path.txt'.format(stage) , 'w') as file:
        file.write("file class_index timestamp")
        for item in image_list:
            file.write("\n")
            name_list=item.split('/')
            classes=name_list[-2]
            if classes not in class_list:
                continue
            class_index=class_list.index(classes)
            timestamp=name_list[-3].split('_')[-1]
            file.write(item+ " "+str(class_index)+" "+str(timestamp))
    print('{} parse path finish!'.format(stage))
