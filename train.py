from torch.optim import SGD
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC,JointTraining
from load_dataset import *

def build_logger(name):
    # log to text file
    text_logger = TextLogger(open('log_{}.txt'.format(name), 'w'))

    # print to stdout
    interactive_logger = InteractiveLogger()

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

nepoch=100
step=30
batch_size=64
start_lr=0.01
weight_decay=1e-5
momentum=0.9


# scenario = SplitMNIST(n_experiences=5)
scenario = get_data_set()
# MODEL CREATION
# model = SimpleMLP(num_classes=scenario.n_classes)
model=resnet18(pretrained=False)
# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# for strate in ['EWC','CWRStar','Replay','GDumb','Cumulative','Naive','GEM','AGEM','LwF']:
for strate in ['GDumb','GEM','Naive','JointTraining','Cumulative','GEM','LwF']:
    print('current strate is {}'.format(strate))

    if strate=='CWRStar':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = CWRStar(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(),cwr_layer_name=None, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='Replay':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = Replay(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='JointTraining':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = JointTraining(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='GDumb':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = GDumb(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='Cumulative':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = Cumulative(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='LwF':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = LwF(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(),
            alpha=[0, 0.5, 1.333, 2.25, 3.2],temperature=1,
             train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='GEM':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = GEM(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), 256,0.5, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='AGEM':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = AGEM(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(),256,256, train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='EWC':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = EWC(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), 0.4, 'online',decay_factor=0.1,
            train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    elif strate=='Naive':
        text_logger ,interactive_logger,eval_plugin=build_logger(strate)
        cl_strategy = Naive(
            model, SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay,momentum=momentum),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=nepoch, eval_mb_size=batch_size,
            evaluator=eval_plugin)
    # except:
    #     print('###########################################')
    #     print('###########################################')
    #     print('skipping {}'.format(strate))
    #     continue
    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))

