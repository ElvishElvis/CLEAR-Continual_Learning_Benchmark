import operator
from copy import deepcopy

from avalanche.training.plugins import StrategyPlugin


class LoadBestPlugin(StrategyPlugin):
    def __init__(self, val_stream_name: str,
                 metric_name: str = 'Top1_Acc_Epoch', mode: str = 'max'):
        """
        Load the best model after the training epochs finishs

        :param val_stream_name: Name of the validation stream to search in the
        metrics. The corresponding stream will be used to keep track of the
        evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
        reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
        given metric should me maximized (resp. minimized).
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.metric_name = metric_name
        self.metric_key = f'{self.metric_name}/train_phase/' \
                          f'{self.val_stream_name}'
        if mode not in ('max', 'min'):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == 'max' else operator.lt
        # Top1_Acc_Exp/eval_phase/test_stream/
        # Top1_Acc_Epoch/train_phase/train_stream/
        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_epoch = None
        # self.reset=reset # reset the best in each experience training 

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_epoch = None
        strategy.evaluator.reset_last_metrics()

    def after_training_epoch(self, strategy, **kwargs):
        self._update_best(strategy)

    def before_eval(self,strategy,**kwargs):
        if(self.best_state==None):
            print('Not using best model since it is None')
        else:
            strategy.model.load_state_dict(self.best_state)
            print('###########################################################')
            print('Loading best model from epoch {} with acc {}'.format(self.best_epoch,self.best_val))
            print('###########################################################')

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        val_acc = res.get(self.metric_key)
        # dict have thing like Top1_Acc_Epoch/train_phase/train_stream/Task000, but self.metric_key don't have the suffix of Task000
        if(val_acc==None):
            key_match=self.metric_key
            key_list=list(res.keys())
            for key in key_list:
                if(self.metric_key in key):
                    key_match=key
                    break
            val_acc=res.get(key_match)
        if self.best_val is None or self.operator(val_acc, self.best_val):
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_epoch = strategy.clock.train_exp_epochs
