[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/multidynet/blob/master/LICENSE)

## Fast Variational Inference of Dynamic LSMs using Bayesian P-Splines

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for the model described in
"Fast Variational Inference of Latent Space Models for Dynamic Networks using Bayesian P-Splines." Inference is performed using
stochastic variational inference. For more details, see [Loyal (2023)]().

Dependencies
------------
``splinetlsm`` requires:

- Python (>= 3.10)

and the requirements highlighted in [requirements.txt](requirements.txt). To install the requirements, run

```python
pip install -r requirements.txt
```

Installation
------------
You need a working installation of numpy, scipy, and Cython to install ``splinetlsm``. Install these required dependencies before proceeding.  Use the following commands to get the copy from GitHub and install all the dependencies:

```
>>> git clone https://github.com/joshloyal/splinetlsm.git
>>> cd splinetlsm
>>> pip install -r requirements.txt
>>> python setup.py install
```

Example
-------
```python
print("Hello, world!")
```

