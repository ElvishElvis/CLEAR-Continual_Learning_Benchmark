import torch
from torch.utils.data import random_split

import math
from torch.utils.data import random_split, DataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset,AvalancheDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader, \
    GroupBalancedInfiniteDataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer


class AGEMPlugin(StrategyPlugin):
    """ Average Gradient Episodic Memory Plugin.
    
    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int,reservoir:bool=False):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.sample_size = int(sample_size)

        self.buffers = []  # one AvalancheDataset for each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None
        if(reservoir==True):
            print('using AGEM-fixed with reservoir sampling')
            self.reservoir_buffer=ReservoirSamplingBuffer(self.sample_size)
            self.reservoir=True
        else:
            print('not using reservoir sampling')
            self.reservoir=False


    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        # print('buffer')
        # print(len(self.buffers))
        # if(len(self.buffers)>0):
            # print(len(self.buffers[0]))
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            batch_size=64
            iteration=math.ceil(tid.shape[0]/batch_size)
            # split (out = avalanche_forward(strategy.model, xref, tid)) into different iteration
            for itera in range(iteration):
                xref_sub= xref[itera*batch_size:(itera+1)*batch_size]
                tid_sub=tid[itera*batch_size:(itera+1)*batch_size]
                yref_sub=yref[itera*batch_size:(itera+1)*batch_size]
                xref_sub=xref_sub.to(strategy.device)
                out_sub = avalanche_forward(strategy.model, xref_sub,tid_sub)
                yref_sub=yref_sub.to(strategy.device)
                loss = strategy._criterion(out_sub, yref_sub)
                loss.backward()
                del xref_sub, out_sub,yref_sub
            self.reference_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            current_gradients = torch.cat(current_gradients)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - \
                    self.reference_gradients * alpha2
                
                count = 0 
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count:count+n_param].view_as(p))
                    count += n_param

    def after_training_exp(self, strategy, **kwargs):
        """ Update replay memory with patterns from current experience. """
        self.update_memory(strategy.experience.dataset)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffer_dliter)

    @torch.no_grad()
    def update_memory(self, dataset):
        """
        Update replay memory with patterns from current experience.
        """
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            dataset, _ = random_split(dataset,
                                      [self.patterns_per_experience,
                                       removed_els])
        # use reservoir sampling to keep the buffer size= self.patterns_per_experience
        if(self.reservoir==True and len(self.buffers)!=0):
            self.reservoir_buffer.update_from_dataset(dataset)
            self.buffers[0]=self.reservoir_buffer.buffer
        else:
            self.buffers.append(dataset)
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=self.sample_size // len(self.buffers),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True)
        self.buffer_dliter = iter(self.buffer_dataloader)
