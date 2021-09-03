from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
import torch.nn as nn
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
from load_dataset import *
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--method",default='Naive GDumb')
                       


def build_logger(name):
    # log to text file
    text_logger = TextLogger(open('./log/log_{}.txt'.format(name), 'w'))

    # print to stdout
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        # cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                                 stream=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    return text_logger ,interactive_logger,eval_plugin

def make_scheduler(optimizer, step_size=30, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler

nepoch=70
step=30
batch_size=64
start_lr=0.01
weight_decay=1e-5
momentum=0.9
timestamp=10
num_classes=11
os.makedirs("./log/",exist_ok=True)
os.makedirs("./model/",exist_ok=True)
args = argparser.parse_args()
method_query=args.method.split()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.get_device_name(0)
# torch.cuda.device_count() 


# for strate in ['EWC','CWRStar','Replay','GDumb','Cumulative','Naive','GEM','AGEM','LwF']:
# ['GDumb','Naive','JointTraining','Cumulative']
for strate in method_query:
    for current_mode in ['offline','online']:
        if(current_mode=='offline'):
            scenario = get_data_set_offline()
        else:
            scenario = get_data_set_online()
        print('========================================================')
        print('========================================================')
        print('current strate is {} {}'.format(strate,current_mode))
        print('========================================================')
        print('========================================================')
        model=resnet18(pretrained=False)
        data_count=3300 if current_mode=='online' else 2310
        if torch.cuda.device_count() > 1:
            print("Let's use all GPUs!")
            model = nn.DataParallel(model)
        else:
            print("nly use one GPU")
        if(torch.cuda.is_available()):
            model=model.cuda()
        optimizer=SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum)
        scheduler= make_scheduler(optimizer,30,0.1)
        if strate=='CWRStar':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = CWRStar(
                model, optimizer,
                CrossEntropyLoss(),cwr_layer_name=None, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='Replay':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = Replay(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,mem_size=data_count,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif (strate=='JointTraining' and current_mode=='offline'):
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = JointTraining(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='GDumb':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)],mem_size=data_count)
        elif strate=='Cumulative':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = Cumulative(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='LwF':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = LwF(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=timestamp).tolist(),temperature=1,
                train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='GEM':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = GEM(
                model, optimizer,
                CrossEntropyLoss(), patterns_per_exp=256,memory_strength=0.5, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='AGEM':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(),patterns_per_exp=256,sample_size=256, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='EWC':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = EWC(
                model, optimizer,
                CrossEntropyLoss(), ewc_lambda=0.4, mode='separate',decay_factor=0.1,
                train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='Naive':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = Naive(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        # elif strate=='ICaRL':
        #     text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
        #     cl_strategy = ICaRL(
        #         model, optimizer,
        #         CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
        #         evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='SynapticIntelligence':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = SynapticIntelligence(
                model, optimizer,
                CrossEntropyLoss(), si_lambda=0.0001,train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        elif strate=='CoPE':
            text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
            cl_strategy = CoPE(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,mem_size=data_count,
                evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        # elif strate=='AR1':
        #     text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
        #     cl_strategy = AR1(
        #         model, optimizer,
        #         CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
        #         evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        # elif strate=='StreamingLDA':
        #     text_logger ,interactive_logger,eval_plugin=build_logger("{}_{}".format(strate,current_mode))
        #     cl_strategy = StreamingLDA(
        #         slda_model=model, 
        #         criterion=CrossEntropyLoss(), input_size= 224,num_classes=num_classes,train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
        #         evaluator=eval_plugin,device=device,plugins=[LRSchedulerPlugin(scheduler)])
        
        else:
            continue
        # except:
        #     print('###########################################')
        #     print('###########################################')
        #     print('skipping {}'.format(strate))
        #     continue
        # TRAINING LOOP
        
        print('Starting experiment...')
        results = []
        if(strate=='JointTraining' and current_mode=='offline'):
            cl_strategy.train(scenario.train_stream)
            results.append(cl_strategy.eval(scenario.test_stream))
            torch.save(model.state_dict(), './model/model_{}__{}.pth'.format(strate,current_mode))
        else:
            for experience in scenario.train_stream:
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)
                # offline
                if(current_mode=='offline'):
                    # train returns a dictionary which contains all the metric values
                    res = cl_strategy.train(experience)
                    print('Training completed')
                    print('Computing accuracy on the whole test set')
                    # test also returns a dictionary which contains all the metric values
                    results.append(cl_strategy.eval(scenario.test_stream))
                # online
                else:
                    print('Computing accuracy on the future timestamp')
                    results.append(cl_strategy.eval(scenario.test_stream))
                    # train returns a dictionary which contains all the metric values
                    res = cl_strategy.train(experience)
                    print('Training completed')
                torch.save(model.state_dict(), './model/model_{}__{}.pth'.format(strate,current_mode))
                
            
            

