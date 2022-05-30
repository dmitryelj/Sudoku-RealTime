/*
 Sudoku C-solver
 Original idea: https://github.com/techwithtim/Sudoku-GUI-Solver

 Compile:
 Linux: gcc -shared -Wl,-soname,solver_lib -o solver_lib.so -fPIC solver_lib.c
 OSX: gcc -shared -Wl,-install_name,solver_lib.so -o solver_lib.so -fPIC solver_lib.c
 Windows: not tested yet, to be done

 Run:
 import ctypes
 lib = ctypes.CDLL('solver_lib.so')
 lib.solve.argtypes = [ctypes.POINTER(ctypes.c_int)]

 bo = [1, 6, 0, ... 7]  # 9x9=81 digits array
 board_data = (ctypes.c_int * len(bo))(*bo)
 res = lib.solve(board_data)
*/


#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

int find_duplicates(int *bo) {
    // If any row or column has duplicated digits, then no solution is available
    for(int line=0; line<9; line++) {
        for(int i=0; i<8; i++) {
            for(int j=i+1; j<9; j++) {
                // Check horizontal line
                if (bo[9*line + i] != 0 && bo[9*line + i] == bo[9*line + j])
                    return 1;
                // Check vertical line
                if (bo[9*i + line] != 0 && bo[9*i + line] == bo[9*j + line])
                    return 1;
            }
        }
    }
    return 0;
}

int find_empty_cells(int *bo) {
    // If any 3x3 cell of the board is empty, the board is not valid
    for(int cx=0; cx<9; cx+=3) {
        for(int cy=0; cy<9; cy+=3) {
            if (bo[9*cy + cx] == bo[9*cy + cx + 1] && bo[9*cy + cx + 1] == bo[9*cy + cx + 2] && bo[9*cy + cx + 2] == bo[9*(cy + 1) + cx] &&
                bo[9*(cy + 1) + cx] == bo[9*(cy + 1) + cx + 1] && bo[9*(cy + 1) + cx + 1] == bo[9*(cy + 1) + cx + 2] &&
                bo[9*(cy + 1) + cx + 2] == bo[9*(cy + 2) + cx] && bo[9*(cy + 2) + cx] == bo[9*(cy + 2) + cx + 1] &&
                bo[9*(cy + 2) + cx + 1] == bo[9*(cy + 2) + cx + 2] && bo[9*(cy + 2) + cx + 2] == 0)
                 return 1;
        }
    }
    return 0;
}

int find_empty(int *bo, int *pos_i, int *pos_j) {
    // Finds an empty space in the board
    for(int i=0; i<9; i++) {
        for(int j=0; j<9; j++) {
            if (bo[9*i + j] == 0) {
                *pos_i = i;
                *pos_j = j;
                return 1;
            }
        }
    }
    return 0;
}

int valid(int *bo, int pos0, int pos1, int num) {
    // Returns if the attempted move is valid
    // Check row
    int i0 = pos0, i1 = pos1;
    for(int i=0; i<9; i++) {
        if (bo[9*i0 + i] == num && i1 != i)
            return 0;
    }
    // Check Col
    for(int i=0; i<9; i++) {
        if (bo[9*i + i1] == num && i1 != i)
            return 0;
    }
    // Check box
    int box_x = (int)(i1/3), box_y = (int)(i0/3);
    for(int i=box_y*3; i<box_y*3 + 3; i++) {
        for(int j=box_x*3; j<box_x*3 + 3; j++) {
            if (bo[9*i + j] == num && (i != pos0 || j != pos1))
                return 0;
        }
    }
    return 1;
}

int solve_(int *bo) {
    // Solves a sudoku board using backtracking
    int row, col;
    if (!find_empty(bo, &row, &col))
        return 1;

    for(int digit=1; digit<=9; digit++) {
        if (valid(bo, row, col, digit)) {
            bo[9*row + col] = digit;
            if (solve_(bo))
                return 1;

            bo[9*row + col] = 0;
        }
    }
    return 0;
}

void print_board(int *bo) {
	for(int i = 0; i < 9; i++) {
		for(int j = 0; j <= 9; j++)
			printf(" %d", bo[9*i + j]);
		printf("\n");
	}
}

DLL_EXPORT int solve(int *bo) {
    if (find_duplicates(bo) || find_empty_cells(bo))
       return 0;

    return solve_(bo);
}


/*
// For testing only:
int main(int argc, char *argv[])
{
    // Code test:
    int board1[9*9] = {1, 6, 0, 0, 7, 0, 0, 0, 0,
                       4, 0, 0, 0, 0, 0, 0, 0, 0,
                       5, 0, 0, 9, 0, 0, 0, 3, 1,
                       0, 0, 9, 0, 4, 0, 6, 0, 0,
                       0, 2, 0, 0, 0, 0, 0, 5, 7,
                       0, 0, 0, 0, 0, 0, 1, 0, 9,
                       0, 0, 0, 0, 0, 2, 0, 0, 5,
                       0, 1, 0, 0, 9, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 7, 9, 0, 2};
	int board2[9*9] = {3, 2, 0, 0, 0, 0, 0, 6, 0,
                       0, 0, 0, 7, 0, 1, 0, 0, 0,
                       9, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 4, 0, 0, 8, 0, 7,
                       0, 0, 0, 0, 2, 0, 5, 0, 0,
                       6, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 8, 7, 5, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 3, 0, 0, 9, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0};
    int board3[9*9] = {1, 6, 0, 0, 7, 0, 0, 0, 0,
                       4, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 0, 0, 9, 0, 0, 0, 1, 1,
                       0, 0, 9, 0, 4, 0, 6, 0, 0,
                       0, 2, 0, 0, 0, 0, 0, 1, 7,
                       0, 0, 0, 0, 0, 0, 1, 0, 9,
                       0, 0, 0, 0, 0, 2, 0, 0, 1,
                       0, 1, 0, 0, 9, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 7, 9, 0, 2};
    int board4[9*9] = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 8, 0, 0, 0, 0,
                       0, 0, 0, 8, 0, 1, 0, 0, 0,
                       0, 4, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0};

    int ret;
    clock_t begin = clock(), end;

    int *bo = board2;
    ret = solve(bo);
    end = clock();
    printf("Res: %d\n", ret);
    printf("dT: %f\n", (double)(end - begin) / CLOCKS_PER_SEC);
    if (ret)
        print_board(bo);

    return 0;
}*/
