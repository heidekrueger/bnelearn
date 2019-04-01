import warnings
import pytest
import torch
from bnelearn.mechanism import PrisonersDilemma

"""Setup shared objects"""

pd = PrisonersDilemma(cuda=True)
device = pd.device

# TODO: write tests
def test_pd_correctness():
    pass


