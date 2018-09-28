"""
feature_scaling.py - a custom library for scaling numerical features for statistical learning.

Numerical feautres can differ in magnitude, units, and range. This means algorithms that rely
on a distance metric (such as L2 norm) will be sensitive to these differences, negatively
impacting model performance.

Scale for:
  - kNN (esp. when using Lp distance for example)
  - PCA (prevent skewing results towards high magnitude features)
  - gradient descent-based training (MLPs, generalized linear models trained with GD)

Other models can handle differing feature scales. Scaling may not be needed for:
  - Tree-based models
  - Naive Bayes
  - Linear Discriminant Analysis

Rule of thumb: if the algorithm computes a distance metric or assumes normality, scale features

"""
__author__ = "Chris Livingston"

import logging
import numpy as np

logging.basicConfig()
log = logging.getLogger(__file__)


def standardize(array, in_place=False):
  """
  Transform an array real numbers to their Z scores for the array.
  This transforms values to have a mean of 0 and a std dev of 1.

  Args:
    array (array-like): a numpy array or a Python List to transform
    in_place (boolean): whether to transform elements in place
  Returns:
    (numpy.ndarray): a numpy array containing standardized elements of the input array
  """
  if type(array) == type(list()):
    if in_place:
      log.warn('Modifiying the input array in place is disabled for Python List objects')
    array = np.asarray(array)
  mu = array.mean()
  sigma = np.std(array)
  func = lambda x: x - mu / sigma
  if in_place:
    array[:] = func(array)
  else:
    array = func(array)
  return array


def mean_normalize(array, in_place=False):
  """
  Transform an array real numbers to mean-normalized values.
  This distribution will have values between -1 and 1 with a mean of 0.

  Args:
    array (array-like): a numpy array or a Python List to transform
    in_place (boolean): whether to transform elements in place
  Returns:
    (numpy.ndarray): a numpy array containing mean-normalized elements of the input array
  """
  if type(array) == type(list()):
    if in_place:
      log.warn('Modifiying the input array in place is disabled for Python List objects.')
    array = np.asarray(array)
  mu = array.mean()
  max_val = array.max()
  min_val = array.min()
  func = lambda x: (x - mu) / (max_val - min_val)
  if in_place:
    array[:] = func(array)
  else:
    array = func(array)
  return array


def min_max_scale(array, in_place=False):
  """
  Transform an array real numbers to minmax-normalized values.
  This distribution will have values between 0 and 1.

  Args:
    array (array-like): a numpy array or a Python List to transform
    in_place (boolean): whether to transform elements in place
  Returns:
    (numpy.ndarray): a numpy array containing min-max scaled elements of the input array
  """
  if type(array) == type(list()):
    if in_place:
      log.warn('Modifiying the input array in place is disabled for Python List objects.')
    array = np.asarray(array)
  max_val = array.max()
  min_val = array.min()
  func = lambda x: (x - min_val) / (max_val - min_val)
  if in_place:
    array[:] = func(array)
  else:
    array = func(array)
  return array


if __name__ == '__main__':
  import doctest
  doctest.testmod()
