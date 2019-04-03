import os
from setuptools import setup

def read(fname):
      return open(os.path.join(os.path.dirname(__file__), fname)).read()

requirements = read('requirements.txt').split()

setup(name='bnlearn',
      version = '0.0.x',
      description='A Framework for learning Equilibria in Bayesian Games',
      url='https://gitlab.lrz.de/heidekrueger/bnelearn',
      author='Stefan Heidekrüger',
      author_email='stefan.heidekrueger@in.tum.de',
      license='proprietary, all rights reserved.',
      packages=['bnelearn'],
      install_requires=requirements,
      zip_safe=False)