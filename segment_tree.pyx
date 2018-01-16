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
import cython
cimport numpy as np

cdef class SegmentTreeSampler:

    cdef object random_state
    cdef double[:] tree
    cdef long[:] indx
    cdef long[:] b_indx
    cdef int n

    def __init__(self, int n, np.ndarray[double, ndim=1, mode="c"] init, random_state):
        self.random_state = random_state
        self.tree = np.zeros(4 * n)
        self.indx = np.zeros(4 * n).astype(int)
        self.b_indx = np.zeros(n + 1).astype(int)
        self.n = n
        self.batch_update(np.arange(n), init)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(self, int pos, float value):
        cdef int left_child
        cdef int right_child
        cdef int div
        cdef int node = 1
        cdef int left = 1
        cdef int right = self.n

        while left != right:
            div = (left + right) / 2
            left_child = 2 * node
            right_child = 2 * node + 1
            if pos <= div:
                right = div
                node = left_child
            else:
                left = div + 1
                node = right_child

        self.tree[node] = np.sqrt(self.tree[node] ** 2 + value)
        self.indx[node] = pos
        self.b_indx[pos] = node
        node /= 2
        while node > 0:
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            node = node / 2

    def batch_update(self, np.ndarray[long, ndim=1, mode="c"] indx, np.ndarray[double, ndim=1, mode="c"] values):
        cdef int i = 0

        while i < indx.size:
            self.update(indx[i] + 1, values[i])
            i += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long sample(self, float mass):
        cdef int left_child
        cdef int right_child
        cdef int div
        cdef int node = 1
        cdef int left = 1
        cdef int right = self.n

        while left != right:
            div = (left + right) / 2
            left_child = 2 * node
            right_child = 2 * node + 1
            if self.tree[left_child] >= mass:
                right = div
                node = left_child
            else:
                left = div + 1
                node = right_child
                mass = mass - self.tree[left_child]

        return self.indx[node] - 1


    def batch_sample(self, int batch_size, float theta):
        cdef int i = 0
        cdef int cnt = 0
        cdef long[:] sampled_indx
        cdef double[:] sampled_ps
        cdef double[:] rands
        cdef double[:] ps

        rands = self.random_state.rand(2 * batch_size)
        sampled_indx = np.zeros(batch_size).astype(int)
        ps = np.zeros(batch_size)

        while i < batch_size:
            if rands[cnt] <= theta:
                sampled_indx[i] = self.random_state.choice(self.n)
            else:
                sampled_indx[i] = self.sample(rands[cnt + 1] * self.tree[1])
                cnt += 1
            i += 1
            cnt += 1
        i = 0
        while i < batch_size:
            ps[i] = (1 - theta) * self.tree[self.b_indx[sampled_indx[i] + 1]] / self.tree[1] + theta / self.n
            i += 1
        return sampled_indx, ps
