import os
from setuptools import setup

def read(fname):
      return open(os.path.join(os.path.dirname(__file__), fname)).read()

requirements = read('requirements.txt').split()

setup(name='bnlearn',
      version = '0.1',
      description='A Framework for Equilibrium Learning in Auctions',
      url='https://github.com/heidekrueger/bnelearn',
      author='Stefan Heidekrüger, Nils Kohring, Paul Sutterer, Martin Bichler',
      author_email='stefan.heidekrueger@in.tum.de',
      license='GNU-GPLv3',
      packages=['bnelearn'],
      install_requires=requirements,
      zip_safe=False)