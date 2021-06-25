import pytest

try:
    import gurobipy 
    gurobipy_installed = True
except ModuleNotFoundError:
    gurobipy_installed = False

if gurobipy_installed: 
    try:
        gurobipy.Model()
        gurobi_licence_valid = True
    except GurobiError:
        gurobi_licence_valid = False

#globally accessable variables
pytest.gurobi_installed = gurobipy_installed
pytest.gurobi_licence_valid = gurobi_licence_valid
