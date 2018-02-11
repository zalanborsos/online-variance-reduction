Online Variance Reduction
===

Introduction
---

The package is a Cython implementation of the bandit sampling algorithm for online variance reduction presented in the paper:

Installation
---
First, install `numpy` with 
```
pip install numpy
```

You can install the package `vrb` locally by running:
```
pip install cython
pip install .
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

Tests
---
Use `nose` in the package directory to run the unit tests:
```
nosetests
```

Feedback
---
Please send any feedback to [Zalan Borsos](https://las.inf.ethz.ch/people/zalan-borsos).

License
---
The code is licenced under the MIT license and free to use by anyone without any restrictions.
