Online Variance Reduction
===

Introduction
---

The package is the implementation of the bandit sampling algorithm for online variance reduction presented in the paper:

> [**Online Variance Reduction for Stochastic Optimization**](https://arxiv.org/pdf/1802.04715.pdf)
> *Zalán Borsos, Andreas Krause, Kfir Y. Levy*.
> Conference On Learning Theory (COLT), 2018.

The implementation is compatible with Python 2 and 3

Installation
---
First, install the dependencies with 
```
pip install numpy nose Cython
```

You can install the package `vrb` locally by running:
```
python setup.py build_ext --inplace
```

Usage
---

The main entry point of the sampler is `vrb.VarianceReducerBandit`. The sampler should be used with alternatingly calling its `sample()` and `update()` as the following snippet shows:
```python
n = 100 # number of data points
sampler = vrb.VarianceReducerBandit(n=n, random_state=0, reg=1, theta=0.1)
for t in range(100): # proceed in 100 rounds
    i, p = sampler.sample(1) # sample 1 points
    loss = adversary.get_loss(i, p) # loss provided by the adversary, e.g. norm of the gradient in SGD
    sampler.update(loss) # feed the loss back to the sampler 
```  
For a detailed example, see the ipython notebook in examples.

Tests
---
Use `nose` in the package directory to run the unit tests:
```
nosetests
```

Feedback
---
Please send any feedback to [Zalán Borsos](https://las.inf.ethz.ch/people/zalan-borsos).

License
---
The code is licenced under the MIT license and free to use by anyone without any restrictions.
