"""This module defines pytest fixtures and sets constants to be used in test files."""

import pytest

@pytest.fixture()
def check_gurobi():
    gurobipy = pytest.importorskip('gurobipy', reason="This test requires gurobi, but it is not installed.")
    try:
        gurobipy.Model()
    except Exception as e: #this would be a GurobiError in practice (but we can't reference it without gurobi.)
        raise AssertionError(
            "This test requires gurobi. Gurobi is installed, but no valid license is available." + 
            "If you do not have a gurobi license, you can still use bnelearn without gurobi, to do so," +
            "please uninstall gurobipy."
        ) from e