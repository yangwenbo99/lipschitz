"""lipschitz.py Enforcing Lipschitz continuity on PyTorch CNNs

Currently, we only consider l-inf norm

With reference to the following paper:
    H. Gouk, E. Frank, B. Pfahringer and M. Cree, Regularisation of Neural Networks by Enforcing Lipschitz
Continuity
"""

import torch
from torch import nn
# from exceptions import NotImplemented

from sys import argv

from typing import List, Tuple

DEBUG = 2

class LipschitzEnforcer:
    def correct(*args):
        raise NotImplemented

    def __call__(self):
        self.correct()


class FullConnectedLipschitzEnforcer(LipschitzEnforcer):
    def __init__(self, model: nn.Linear, max_L):
        """
        @param model: the full conntected layer
        @param max_L: the maximum allowed value of the Lipschitz constant
        """
        self._model = model
        self._max_L = max_L

    def correct(self, max_L=None):
        """
        The method of calculation is the same but the corrcection is  not exactly
        """
        if max_L is None:
            max_L = self._max_L
        with torch.no_grad():
            row_sum = torch.abs(self._model.weight).sum(axis=1)
            L = max(row_sum)
            if L <= max_L:
                # Constrain already satisfied
                return
            trimed_row_sum = row_sum.clamp(max=max_L)
            scale = trimed_row_sum / row_sum
            if DEBUG > 4:
                print('L:', L, row_sum.shape, self._model.weight.shape)
            if DEBUG > 2:
                print('Min scale:', torch.min(scale))
                print('Max scale:', torch.max(scale))
            weight = self._model.weight.data
            self._model.weight.data = (weight.t() * scale).t()

    @staticmethod
    def is_applicable_to(model) -> bool:
        """
        Check whether the enforcer is automatically applicable to the model
        """
        return isinstance(model, nn.Linear)

    def get_object(self):
        """
        Get the object that the instance of enforcer controls
        """
        return self._model

class ConvolutionalLipschitzEnforcer(LipschitzEnforcer):
    def __init__(self, model: nn.Linear, max_L):
        """
        @param model: the full conntected layer
        @param max_L: the maximum allowed value of the Lipschitz constant
        """
        self._model = model
        self._max_L = max_L

    def correct(self, max_L=None):
        """
        The method of calculation is the same but the corrcection is  not exactly

        See the paper for the definition of variables in this method (member
        function)
        """
        if max_L is None:
            max_L = self._max_L
        with torch.no_grad():
            weight = self._model.weight.data
            F = torch.abs(weight).sum(axis=(2, 3))
            row_sum = F.sum(axis=1)
            L = max(row_sum)
            if L <= max_L:
                # Constrain already satisfied
                return

            trimed_row_sum = row_sum.clamp(max=max_L)
            scale = trimed_row_sum / row_sum
            mscale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*weight.shape)
            if DEBUG > 4:
                print('Conv', 'L:', L, row_sum.shape, self._model.weight.shape)
            if DEBUG > 2:
                print('Conv Min scale:', torch.min(scale))
                print('Conv Max scale:', torch.max(scale))
            self._model.weight.data *= mscale

    @staticmethod
    def is_applicable_to(model) -> bool:
        return isinstance(model, nn.Conv2d)

    def get_object(self):
        return self._model

class TensorParameterTrimer(LipschitzEnforcer):

    def __init__(self, param: nn.Parameter, min=-1.0, max=1.0):
        """
        """
        self._param = param
        self._min = min
        self._max = max

    def correct(self):
        with torch.no_grad():
            self._param.data = self._param.data.clamp(
                    min=self._min,
                    max=self._max)

    @staticmethod
    def is_applicable_to(model) -> bool:
        return False

    def get_object(self):
        return self._model


class NetworkLipschitzEnforcer():
    def __init__(self, model: nn.Module):
        self._correctors = []
        self._model = model

    def correct(self):
        for c in self._correctors:
            c.correct()

    def import_from_list(
            self,
            corrector_list: List[Tuple[type, tuple]],
            excepts: List[nn.Module] = []
            ):
        """
        Automatically add enforcers into this class.

        @param corrector_list: a list (or other iterable) with each element
                               to be a pair, whose first element is a class
                               and whose second element is the default
                               parameter(s) in  tuple
        @param excepts: the modules that should be skipped
        """
        for m in self._model.modules():
            if m in excepts:
                continue
            for correrctor_cls, params in corrector_list:
                if correrctor_cls.is_applicable_to(m):
                    if DEBUG >= 2:
                        print('Applying', correrctor_cls, 'to', m)
                    self._correctors.append(correrctor_cls(m, *params))

    def add_enforcer(self, enforcer: LipschitzEnforcer):
        self._correctors.append(enforcer)

    @classmethod
    def default_enforcer(cls, model, excepts: List[nn.Module] = []):
        """
        This method construct a default enforcer for a network.

        The default one in construct in this way:
            attemp to add all (automatically) applicable enforcer to each
            module, with all parameters set to default and max_L set to 1
        """
        corrector_list = [
                (ConvolutionalLipschitzEnforcer, (1,)),
                (FullConnectedLipschitzEnforcer, (1,))
                ]
        enf = cls(model)
        enf.import_from_list(corrector_list, excepts)
        return enf


def test(ttype):
    import test_model
    device = torch.device('cuda:0')
    net = test_model.Net().to(device)
    ec1 = ConvolutionalLipschitzEnforcer(net.conv1, 1)
    ec2 = ConvolutionalLipschitzEnforcer(net.conv2, 1)
    e1 = FullConnectedLipschitzEnforcer(net.fc1, 1)
    e2 = FullConnectedLipschitzEnforcer(net.fc2, 1.5)
    e3 = FullConnectedLipschitzEnforcer(net.fc3, 3)
    if 'adv' in ttype:
        def callback(i, *args):
            if i % 20 == 0:
                ec1.correct()
                ec2.correct()
                e1.correct()
                e2.correct()
                e3.correct()
    else:
        def callback(i, *args):
            return

    trainer = test_model.Trainer(net, device)
    if 'loadpure' in ttype:
        print('loading pure')
        trainer.load('pure.pth')
    if 'loadadv' in ttype:
        print('loading adv')
        trainer.load('adv.pth')
    if 'skip' not in ttype:
        callback(0)
        for i in range(5):
            trainer.train_single_epoch(callback=callback)
            callback(0)
            print('Accuracy:', trainer.test())

    if 'goforever' in ttype:
        while True:
            trainer.train_single_epoch(callback=callback)
            callback(0)
            print('Accuracy:', trainer.test())
            if 'savepure' in ttype:
                print('saving pure')
                trainer.save('pure.pth')
            if 'saveadv' in ttype:
                print('saving adv')
                trainer.save('adv.pth')

    if 'savepure' in ttype:
        print('saving pure')
        trainer.save('pure.pth')
    if 'saveadv' in ttype:
        print('saving adv')
        trainer.save('adv.pth')
    print('Accuracy:', trainer.test())

if __name__ == '__main__':
    if len(argv) >= 2 and argv[1] == 'test':
        if len(argv) >= 3:
            ttype = argv[2].split()
        else:
            ttype = []
        test(ttype)
