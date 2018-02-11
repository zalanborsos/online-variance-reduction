# MIT License
#
# Copyright (c) 2018 Zalan Borsos
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from abc import ABCMeta, abstractmethod
import segment_tree


class SamplerBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n, random_state):
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        self.n = n
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def update(self, loss):
        pass

    @abstractmethod
    def reset(self):
        pass


class VarianceReducerBandit(SamplerBase):
    """
    Variance Reducer Bandit with segment tree sampler

    Args:
            n: the number of points
            random_state: np.random.RandomState instance or int used for seeding
            reg: regularizer; can be 1d np.ndarray of shape n or a scalar. If array is passed,
                the regularization happens per point, otherwise globally.
            theta: coefficient of mixing with uniform distribution
    """

    def __init__(self, n, random_state, reg, theta):
        super(VarianceReducerBandit, self).__init__(n, random_state)
        self.reg = reg
        self.theta = theta
        self.st = segment_tree.SegmentTreeSampler(self.n, np.ones(self.n) * reg, self.random_state)

    def sample(self, batch_size):
        """
        Samples a batch of size batch_size.

        Args:
            batch_size: size of the batch

        Returns:
            a tuple consisting of the sampled indices and their corresponding probabilities
        """
        last_sampled, p = self.st.batch_sample(batch_size, self.theta)
        self.last_sampled = np.asarray(last_sampled)
        self.p = np.asarray(p)
        return self.last_sampled.reshape(-1), self.p

    def update(self, loss):
        """
        Updates the sampling distribution based on the received loss.

        Args:
            loss: np.ndarray representing the losses incurred for the points returned by the last call to sample()

        """
        self.st.batch_update(self.last_sampled, loss ** 2 / self.p)

    def reset(self):
        """
        Resets the sampling distribution to its initial state.

        """
        self.st = segment_tree.SegmentTreeSampler(self.n, np.ones(self.n) * self.reg, self.random_state)
