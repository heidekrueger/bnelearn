import pytest

try:
    import gurobipy
    gurobipy.Model() 
    pytest.gurobi_licence_valid = True  
except Exception:
    pytest.gurobi_licence_valid = False


