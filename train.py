from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
from torchvision.models import resnet18,resnet50,resnet101
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torchvision
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.gdumb import GDumbPlugin
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC,JointTraining,SynapticIntelligence,CoPE
from avalanche.training.strategies.icarl import ICaRL
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.deep_slda import StreamingLDA
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training.plugins.load_best import LoadBestPlugin
from load_dataset import *
from parse_data_path import *
import argparse
from get_config import *
from extract_feature import *
from parse_log_to_result import *
import glob
import json
                       
def build_logger(name):
    # log to text file
    text_logger = TextLogger(open('../{}/log/log_{}.txt'.format(args.split,name), 'w'))
    
    # print to stdout
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger('../{}/tb_data'.format(args.split))
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, epoch_running=True,experience=True, stream=True,trained_experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        # cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=args.num_classes, save_image=False,
                                 stream=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    return text_logger ,interactive_logger,eval_plugin

def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler


def extract(args):
    print('extract feature for {}'.format(args.data_folder_path))
    args.temp_split=args.split
    args.split='temp_folder' # dummy folder for extracting feature
    args=extract_feature(args)
    print('Finished extract feature {}'.format(args.pretrain_feature))
    os.system('rm -rf ../temp_folder')
    args.split=args.temp_split
    args.data_folder_path=os.path.join(args.feature_path,args.pretrain_feature)
    return args


def move_data_trinity(input_path,remove=False):
    '''
    Move data from /data to /scratch (for trinity server)
    It only move the data in args.data_folder_path, not the current script
    '''
    target_path=os.path.join('/scratch/jiashi/',"/".join(input_path[1:].split('/')[:-1]))
    print('Moving data {} to local server'.format(input_path))
    # '/scratch/jiashi/data/jiashi/moco_resnet50_clear_10_feature'
    path_on_scratch=os.path.join(target_path,input_path.split('/')[-1])
    print(path_on_scratch)
    if(os.path.isdir(path_on_scratch)==False):
        if(remove==True):
            print('rm previous data path')
            assert False
        os.makedirs(target_path,exist_ok=True)
        os.system('cp -rf {} {}'.format(input_path,target_path))
    return path_on_scratch
global args
args=get_config()





try:
    restart=int(args.restart)
except:
    print('restart flag must be 0/1')
    assert False
if(restart==1):
    print('???!!!!!!!!!!!!!!!!!!!!!!!!!You sure to remove the old checkpoint ???!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('enter Y/y to continue')
    value=input()
    if(value=="y" or value=='Y'):
        assert False
        print('remove old split folder')
os.makedirs("../{}".format(args.split),exist_ok=True)
os.makedirs("../{}/log/".format(args.split),exist_ok=True)
os.makedirs("../{}/model/".format(args.split),exist_ok=True)
os.makedirs("../{}/metric/".format(args.split),exist_ok=True)
method_query=args.method.split() # list of CL method to run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.get_device_name(0)
# torch.cuda.device_count() 
'''
Remember to delete the old feature path before generating new feature 
'''
if(args.pretrain_feature!='None'):
    if(args.data_test_path !='' and args.data_train_path!=''):
        data_test_path=args.data_test_path
        data_train_path=args.data_train_path
        train_num_instance_each_class=args.num_instance_each_class
        for stage in ['train','test']:
            if(stage=='train'):
                args.data_folder_path=data_train_path
                args.data_train_path=''
                args.pretrain_feature='train_'+args.pretrain_feature
                args=extract(args)
                args.data_train_path=args.data_folder_path
            else:
                args.num_instance_each_class=args.num_instance_each_class_test
                args.data_folder_path=data_test_path
                args.data_test_path=''
                args.pretrain_feature=args.pretrain_feature.replace('train','test')
                args=extract(args)
                args.data_test_path=args.data_folder_path
                args.num_instance_each_class=train_num_instance_each_class
    else:    
        args=extract(args)

if(args.data_test_path !='' and args.data_train_path!=''):
    for stage in ['train','test']:
        if(stage=='train'):
            args.data_train_path=move_data_trinity(args.data_train_path,False)
        else:
            args.data_test_path=move_data_trinity(args.data_test_path,False)

else:
    args.data_folder_path=move_data_trinity(args.data_folder_path,False)

with open('../{}/args.txt'.format(args.split), 'w') as f:
    print('args', args, file=f) # keep a copy of the args
os.system('cp -rf ../avalanche ../{}/'.format(args.split)) # keep a copy of the scripts
for strate in method_query:
    for current_mode in ['offline']:
        # skip previous train model if necessary
        import glob
        model_path=sorted(glob.glob('../{}/model/model_{}_{}*'.format(args.split,strate,current_mode)))
        if(len(model_path)==0 and args.eval==True):
            checkpoint_path='../{}/model/model_{}_{}*'.format(args.split,strate,current_mode)
            print('Checkpoint for model {} is not found at path {}'.format(strate,checkpoint_path))
            continue
        if(len(model_path)!=0):
            model_path=model_path[-1]
            state_dict=torch.load(model_path)
        else:
            state_dict=None
        if(current_mode=='offline'):
            scenario = get_data_set_offline(args)
        else:
            scenario = get_data_set_online(args)
        print('========================================================')
        print('========================================================')
        print('current strate is {} {}'.format(strate,current_mode))
        print('========================================================')
        print('========================================================')
        if args.pretrain_feature=='None':
            pretrain=args.image_train_pretrain
            model=torchvision.models.__dict__[args.image_train_model_arch](pretrained=pretrain)
            if(args.image_train_model_arch=='resnet50' and args.image_train_attribute=='moco'):
                model=moco_v2_yfcc_feb18_bucket_0_gpu_8(model)

        else:
            model=nn.Linear(args.pretrain_feature_shape,args.num_classes)
        data_count=int(args.num_classes*args.num_instance_each_class) if current_mode=='online' else int(args.num_classes*args.num_instance_each_class*(1-args.test_split))
        print('data_count is {}'.format(data_count))
        data_count=min(args.max_memory_size,data_count) # buffer_size cannot be greater than 3000
        if(strate.split("_")[-1].isnumeric()==False):
            buffer_size=data_count
        else:
            buffer_size=int(strate.split("_")[-1])

        if torch.cuda.device_count() > 1:
            print("Let's use all GPUs!")
            model = nn.DataParallel(model)
        else:
            print("only use one GPU")
        if(args.load_prev==True and state_dict is not None):
            model.load_state_dict(state_dict)
            print()
            print('loaded previous model {}'.format(model_path))
            print()
        if(torch.cuda.is_available()):
            model=model.cuda()
        optimizer=SGD(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.start_lr, weight_decay=float(args.weight_decay),momentum=args.momentum)
        scheduler= make_scheduler(optimizer,args.step_schedular_decay,args.schedular_step)
        # patience=5 # Number of epochs to wait without generalization improvements before stopping the training .
        # EarlyStoppingPlugin(patience, 'train_stream')
        # 
        plugin_list=[LRSchedulerPlugin(scheduler),LoadBestPlugin('train_stream')]
        text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
        if strate=='CWRStar':
            cl_strategy = CWRStar(
                model, optimizer,
                CrossEntropyLoss(),cwr_layer_name=None, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'Replay' in strate: 
            cl_strategy = Replay(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif (strate=='JointTraining' and current_mode=='offline'):
            cl_strategy = JointTraining(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch*args.timestamp//3, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'GDumbFinetune' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='class_balance')
        # stanard gdumb= reset model+ class_balance buffer'
        elif 'GDumb' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=True,buffer='class_balance')
        elif 'BiasReservoir' in strate:
            if('reset' in strate):
                resett=True
            else:
                resett=False
            alpha_mode ='Dynamic' if 'Dynamic' in strate else 'Fixed'
            alpha_value=float(strate.split("_")[-1])
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=resett,buffer='bias_reservoir_sampling',
                alpha_mode=alpha_mode,alpha_value=alpha_value)
        # this is basically the 'reservoir sampling in the paper(no reset+ reservoir sampling'
        elif 'Reservoir' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='reservoir_sampling')
        elif 'Cumulative' in strate:
            if('reset' in strate):
                resett=True
            else:
                resett=False
            cl_strategy = Cumulative(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,reset=resett)
        elif strate=='LwF':
            cl_strategy = LwF(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='GEM':
            cl_strategy = GEM(
                model, optimizer,
                CrossEntropyLoss(), patterns_per_exp=data_count,memory_strength=0.5, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'AGEMFixed' in strate:
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(),patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,reservoir=True)
        elif 'AGEM' in strate:
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(),patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,reservoir=False)
        elif strate=='EWC':
            cl_strategy = EWC(
                model, optimizer,
                CrossEntropyLoss(), ewc_lambda=0.4, mode='online',decay_factor=0.1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='Naive':
            cl_strategy = Naive(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        # elif strate=='ICaRL':
        #     cl_strategy = ICaRL(
        #         model, optimizer,
        #         CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
        #         evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='SynapticIntelligence':
            cl_strategy = SynapticIntelligence(
                model, optimizer,
                CrossEntropyLoss(), si_lambda=0.0001,train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'CoPE' in strate:
            cl_strategy = CoPE(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        # elif strate=='AR1':
        #     cl_strategy = AR1(
        #         model, optimizer,
        #         CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
        #         evaluator=eval_plugin,device=device,plugins=plugin_list)
        # elif strate=='StreamingLDA':
        #     cl_strategy = StreamingLDA(
        #         slda_model=model, 
        #         criterion=CrossEntropyLoss(), input_size= 224,num_classes=args.num_classes,train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
        #         evaluator=eval_plugin,device=device,plugins=plugin_list)
        
        else:
            continue
        print('Starting experiment...')
        train_metric={}
        test_metric = {}
        if(strate=='JointTraining' and current_mode=='offline'):
            model_save_path='../{}/model/model_{}_{}_time{}.pth'.format(args.split,strate,current_mode,0)
            if(args.eval==False):
                train_metric[0]=cl_strategy.train(scenario.train_stream)
            test_metric[0]=cl_strategy.eval(scenario.test_stream)
            print('current strate is {} {}'.format(strate,current_mode))
            torch.save(model.state_dict(), model_save_path)
            with open("../{}/metric/train_metric_{}.json".format(args.split,strate), "w") as out_file:
                    json.dump(train_metric, out_file, indent = 6)
            with open("../{}/metric/test_metric_{}.json".format(args.split,strate), "w") as out_file:
                json.dump(test_metric, out_file, indent = 6)
            
        else:
            train_list=scenario.train_stream
            cur_timestep=0
            if(len(model_path)!=0 and args.load_prev==True):
                try:
                    with open("../{}/metric/train_metric_{}.json".format(args.split,strate), "r") as file:
                        prev_train_metric=json.load(file)
                    with open("../{}/metric/test_metric_{}.json".format(args.split,strate), "r") as file:
                        prev_test_metric=json.load(file)
                    #extract ../clear100_imgnet_res50/model/model_BiasReservoir_Dynamic_1.0_offline_time05.pth as 5
                    load_prev_time_index=int(model_path.split('_')[-1].split('.')[0][4:])
                    train_list=train_list[load_prev_time_index+1:]
                    cur_timestep=load_prev_time_index+1
                    test_metric=prev_test_metric
                    train_metric=prev_train_metric
                    print('start runing from bucket {}'.format(cur_timestep))
                except:
                    pass

            for experience in train_list:
                model_save_path='../{}/model/model_{}_{}_time{}.pth'.format(args.split,strate,current_mode,str(cur_timestep).zfill(2))
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)
                print('current strate is {} {}'.format(strate,current_mode))
                # offline
                if(current_mode=='offline'):
                    # train returns a dictionary which contains all the metric values
                    print('current strate is {} {}'.format(strate,current_mode))
                    print('Training completed')
                    print('Computing accuracy on the whole test set')
                    # test also returns a dictionary which contains all the metric values
                    if(args.eval==False):
                        train_metric[cur_timestep]=cl_strategy.train(experience)
                    test_metric[cur_timestep]=cl_strategy.eval(scenario.test_stream)
                    print('current strate is {} {}'.format(strate,current_mode))
                # online
                else:
                    print('current strate is {} {}'.format(strate,current_mode))
                    print('Computing accuracy on the future timestamp')
                    test_metric[cur_timestep]=cl_strategy.eval(scenario.test_stream)
                    if(args.eval==False):
                        train_metric[cur_timestep]=cl_strategy.train(experience)
                    # train returns a dictionary which contains all the metric values
                    print('Training completed')
                    print('current strate is {} {}'.format(strate,current_mode))
                torch.save(model.state_dict(), model_save_path)
                log_path='../{}/log/'.format(args.split)
                log_name='log_{}.txt'.format("{}_{}".format(strate,current_mode))
                with open("../{}/metric/train_metric_{}.json".format(args.split,strate), "w") as out_file:
                    json.dump(train_metric, out_file, indent = 6)
                with open("../{}/metric/test_metric_{}.json".format(args.split,strate), "w") as out_file:
                    # convert tensor to string for json dump
                    test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream']=\
                    test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'].numpy().tolist()
                    json.dump(test_metric, out_file, indent = 6)
                out_file.close()
                cur_timestep+=1
                if(args.eval==True):
                    break
                # move_metric_to_main_node(log_path,log_name,main_server_path='/data/jiashi/metric')