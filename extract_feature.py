from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,Resize
from torch.utils.data import Dataset,Subset,DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import numpy as np
from PIL import Image
import torch
from parse_data_path import *
from load_dataset import *
from get_config import *

import torch

MOCO_PATH = "/data3/zhiqiul/self_supervised_models/moco_r50_v2-e3b0c442.pth"
INSTANCE_PATH = "/data3/zhiqiul/self_supervised_models/lemniscate_resnet50_update.pth"
BYOL_PATH = "/data3/zhiqiul/self_supervised_models/byol_r50-e3b0c442.pth"
ROT_PATH = "/data3/zhiqiul/self_supervised_models/rotation_r50-cfab8ebb.pth"
MOCO_YFCC_GPU_8_PATH = "/data3/zhiqiul/self_supervised_models/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar"
MOCO_YFCC_GPU_4_RESNET18_PATH = "/data3/zhiqiul/self_supervised_models/yfcc_moco_models/sep_16_bucket_11_idx_0_gpu_4_resnet18/checkpoint_0199.pth.tar"

def moco_v2(model, path=MOCO_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model

def byol(model, path=BYOL_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model

def rot(model, path=ROT_PATH):
    checkpoint = torch.load(path)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model


def load_moco_ckpt(model, path):
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    return model

def moco_v2_yfcc_feb18_bucket_0_gpu_8(model, path=MOCO_YFCC_GPU_8_PATH):
    return load_moco_ckpt(model, path=path)

def moco_v2_yfcc_sep16_bucket_0_gpu_4_resnet18(model, path=MOCO_YFCC_GPU_4_RESNET18_PATH):
    return load_moco_ckpt(model, path=path)



def get_instance_time(args,idx,all_timestamp_index):
    for index,list in enumerate(all_timestamp_index):
        if(idx in list):
            return index
    assert False, "couldn't find timestamp info for data with index {}".format(idx)
def collator(input_):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image,label=input_[0]
    tensor=transform(image).unsqueeze(0)
    return tensor, torch.tensor(int(label))


def extract_feature(args):
    dataset,all_timestamp_index=get_feature_extract_loader(args)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collator)
    os.makedirs(args.feature_path,exist_ok=True)
    
    feature=args.pretrain_feature
    feature_path=os.path.join(args.feature_path,'{}'.format(feature))
    class_list=args.class_list.split()
    if(os.path.isdir(feature_path)):
        return 

    # create feature folder
    os.makedirs(feature_path,exist_ok=True)
    for ii in range(1,args.timestamp+1):
        os.makedirs(os.path.join(feature_path,str(ii)),exist_ok=True)
        for item in class_list:
            os.makedirs(os.path.join(feature_path,"bucket_"+str(ii),item),exist_ok=True)

    os.makedirs(feature_path,exist_ok=True)
    model=resnet18(pretrained=False)
    model=moco_v2_yfcc_sep16_bucket_0_gpu_4_resnet18(model)
    model.cuda()
    model.eval()
    for index,item in enumerate(loader):
        if(index%500==0):
            print('finished extract feature {}'.format(index))
        image,class_=item
        image=image.cuda()
        class_=class_.cuda()
        timestamp=get_instance_time(args,index,all_timestamp_index)
        output=model(image).detach().cpu() #torch.Size([1, 1000])
        target_path=os.path.join(feature_path,"bucket_"+str(timestamp),class_list[class_.detach().cpu().item()])
        torch.save(output,os.path.join(target_path,'{}.pth'.format(index)))




# global args
# parser=get_config()
# args = parser.parse_args()
# feature_list=['moco']
# # import pdb;pdb.set_trace()
# extract_feature(args,feature_list)


