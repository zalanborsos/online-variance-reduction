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
import segment_tree
import vrb


def test_sampling():
    n = 100000
    nr_step = 2000
    batch_size = 100
    theta = 0.1
    init = 100 * np.ones(n)
    st = segment_tree.SegmentTreeSampler(n, init, np.random.RandomState(0))
    cumul = init

    for i in range(nr_step):
        ind, batch = np.random.choice(n, batch_size), np.random.randint(0, 100, batch_size)

        # update naive sampling
        cumul[ind] += batch
        t = np.sqrt(cumul)
        t /= np.sum(t)
        t = (1 - theta) * t + theta / n

        # update segment tree
        st.batch_update(ind, batch.astype(float))
        ind, p = st.batch_sample(batch_size, theta)

        # naive sampling and segment tree sampling should give close match
        assert np.sum(np.abs(t[ind] - p)) < 1e-5
        assert np.sum(p) > batch_size * .9 / n


def test_bandit():
    n, d = 10, 100
    X = np.random.rand(n, d)

    # vrb with regularization L = 1, 0 mixing
    sampler = vrb.VarianceReducerBandit(X, np.random.RandomState(0), 1, 0)

    # sampler a point, check if its sampling probability is 1 / n
    ind, p = sampler.sample(1)
    assert np.allclose([p], [1. / n])

    # provide 1 as loss, find it again and assert its sampling probability is np.sqrt(11) / (np.sqrt(11) + 9)
    sampler.update(1)
    ind2, p = sampler.sample(1)
    while ind != ind2:
        sampler.update(0)
        ind2, p = sampler.sample(1)
    assert np.allclose([p], [np.sqrt(11) / (np.sqrt(11) + 9)])

    # test if resetting works
    sampler.reset()
    ind, p = sampler.sample(1)
    assert np.allclose([p], [1. / n])
