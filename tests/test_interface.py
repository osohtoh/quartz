import os

import pytest
from quartz.xtbinterface import get_props


def test_runner():
    input = "tests/sample_data/1F0R.pdb"
    get_props(input)
