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

The main entry point of the sampler is `vrb.VarianceReducerBandit`. The sampler should be trained with alternatingly calling its `sample()` and `update()` methods.  

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
