from casadi import *
import os
from SrdPy import SymbolicEngine

def generateDynamicsGeneralizedCoordinatesModel(symbolicEngine:SymbolicEngine,
                                                H,c,T,
                                                functionName_H,
                                                functionName_c,
                                                functionName_T,
                                                casadi_cCodeFilename,
                                                path):

    print('Starting writing function for the '+functionName_H)
    g_dynamics_H = Function(functionName_H, [symbolicEngine.q], [H], ['q'], ['H'])

    print('Starting writing function for the '+functionName_c)
    g_dynamics_c = Function(functionName_c, [symbolicEngine.q,symbolicEngine.v],[c], ['q', 'v'], ['c'])

    print('Starting writing function for the '+functionName_T)
    g_dynamics_T = Function(functionName_T, [symbolicEngine.q], [T], ['q'], ['T'])

    c_function_name = casadi_cCodeFilename+'.c'
    so_function_name = casadi_cCodeFilename+'.so'

    current_cwd = os.getcwd()
    if os.path.isdir(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    CG = CodeGenerator(c_function_name)
    CG.add(g_dynamics_H)
    CG.add(g_dynamics_c)
    CG.add(g_dynamics_T)
    CG.generate()

    command = "gcc -fPIC -shared "+c_function_name +" -o "+so_function_name
    print("Running "+command)

    os.system(command)

    os.chdir(current_cwd)

    print("Generated C code!")

    return     {"functionName_H":functionName_H,
                "functionName_c":functionName_c,
                "functionName_T":functionName_T,
                "casadi_cCodeFilename":casadi_cCodeFilename,
                "path":path,
                "dofConfigurationSpaceRobot":symbolicEngine.dof,
                "dofControl":symbolicEngine.u.shape[0]}
