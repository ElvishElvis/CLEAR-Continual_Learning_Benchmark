import copy
from typing import TYPE_CHECKING

from avalanche.models import DynamicModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer,ReservoirSamplingBuffer,BiasedReservoirSamplingBuffer

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class GDumbPlugin(StrategyPlugin):
    """ GDumb plugin.

    At each experience the model is trained  from scratch using a buffer of
    samples collected from all the previous learning experiences.
    The buffer is updated at the start of each experience to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.
    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size: int = 200,reset=True,buffer='class_balance',alpha_mode=None,alpha_value=None):
        super().__init__()
        self.mem_size = mem_size
        self.reset=reset
        # model initialization
        self.buffer = {}
        if(buffer=='class_balance'):
            self.storage_policy = ClassBalancedBuffer(
                max_size=self.mem_size,
                adaptive_size=True
            )
        elif(buffer=='reservoir_sampling'):
            self.storage_policy = ReservoirSamplingBuffer(
                max_size=self.mem_size
            )
        elif(buffer=='bias_reservoir_sampling'):
            assert type(alpha_mode)==str
            assert type(alpha_value)==float or type(alpha_value)==int
            self.storage_policy = BiasedReservoirSamplingBuffer(
                max_size=self.mem_size,
                alpha_mode=alpha_mode,
                alpha_value=float(alpha_value))
        else:
            assert False, 'Need to select buffer from class_balance/reservoir_sampling/bias_reservoir_sampling'
        self.init_model = None

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        print('reset is {}'.format(self.reset==True))
        if self.reset==True:
            if self.init_model is None:
                self.init_model = copy.deepcopy(strategy.model)
            else:
                strategy.model = copy.deepcopy(self.init_model)
            strategy.model_adaptation(self.init_model)
    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        if self.reset==True:
            strategy.model_adaptation(self.init_model)
    def after_train_dataset_adaptation(self, strategy: "BaseStrategy",
                                       **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        strategy.adapted_dataset = self.storage_policy.buffer
