# -*- coding: utf-8 -*-
"""
Created on Monday November 28th 2016
@author: Emmanuel-Lin TOULEMONDE and Alexis BONDU
"""

import distutils.core as _dc
import Cython.Build as _cb

_dc.setup(
    ext_modules = _cb.cythonize("timehorizonlearner/cythonGenerator.pyx")
)


from setuptools import setup, find_packages
setup(name = 'timehorizonlearner',
      version = '0.1',
      description = 'Time Horizon Learner: a classifier at multi horrizon',
      long_description = 'Modelizing class evolution in time using decision trees and markov chains.',
      author = 'Alexis Bondu, Emmanuel-Lin Toulemonde',
      author_email = 'el.toulemonde@protonmail.com, alexis.bondu@gmail.com',
      license = 'GPL-3 | file LICENSE',
      packages=find_packages())