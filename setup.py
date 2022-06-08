from setuptools import setup, find_packages

setup(name='bnelearn',
      version = '1.0.0',
      description='A Framework for learning Equilibria in Bayesian Games',
      url='https://gitlab.lrz.de/heidekrueger/bnelearn',
      author='Stefan Heidekrüger',
      author_email='stefan.heidekrueger@in.tum.de',
      license='GNU-GPLv3',
      #package_dir={'bnelearn'},
      packages=find_packages(where='.'),
      python_requires = '>=3.9, <3.10',
      install_requires=['torch>=1.10', 'tensorboard', 'matplotlib', 'pandas',
            'numpy', 'future', 'jupyterlab', 'tabulate', 'tqdm', 'sympy'],
      extras_require={
            'external_solvers': ['qpth', 'gurobipy', 'cvxpy'],
            'dev': ['pylint', 'pylint-gitlab', 'pynvml'],
            'test': ['pytest', 'pytest-cov', 'pytest-xdist'],
            'docs': ['Sphinx', 'sphinx-rtd-theme']
            },
      zip_safe=False)