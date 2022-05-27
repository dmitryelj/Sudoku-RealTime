# C-wrapper for solving the soduku board
# Input data: List of 9x9=81 cells

from typing import List
import ctypes


lib = None


def solve_c(bo: List) -> bool:
    global lib
    if lib is None:
        lib = ctypes.CDLL('solver_lib.so')
        lib.solve.argtypes = [ctypes.POINTER(ctypes.c_int)]

    board_data = (ctypes.c_int * len(bo))(*bo)
    res = lib.solve(board_data)
    if res:
        bo[:] = list(board_data)
    return res
