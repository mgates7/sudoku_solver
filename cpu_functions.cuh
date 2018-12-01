#ifndef _CPU_FUNCTIONS
#define _CPU_FUNCTIONS

// CPU Directives
#define GIG 1000000000
//#define CPG 2.5 // Cycles per GHz -- Adjust to your computer

int cpu_solveSudoku(int h_puzzle[][9]);
int isValid(int number, int h_puzzle[][9], int row, int column);
int sudokuHelper(int h_puzzle[][9], int row, int column);
void printSudoku(int h_puzzle[][9]);

/*
 * Timing structs/functions
 */

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

/*
  * A helper function to call sudokuHelper recursively
  */
int cpu_solveSudoku(int h_puzzle[][9])
{
    return sudokuHelper(h_puzzle, 0, 0);
}

/*
  * A recursive function that does all the gruntwork in solving
  * the h_puzzle.
  */
int sudokuHelper(int h_puzzle[][9], int row, int column)
{
    int nextNumber;
    /*
      * Have we advanced past the h_puzzle?  If so, hooray, all
      * previous cells have valid contents!  We're done!
      */
    if (row == 9)
    {
        return 1;
    }

    /*
      * Is this element already set?  If so, we don't want to 
      * change it.
      */
    if (h_puzzle[row][column])
    {
        if (column == 8)
        {
            if (sudokuHelper(h_puzzle, row + 1, 0))
                return 1;
        }
        else
        {
            if (sudokuHelper(h_puzzle, row, column + 1))
                return 1;
        }
        return 0;
    }

    /*
      * Iterate through the possible numbers for this empty cell
      * and recurse for every valid one, to test if it's part
      * of the valid solution.
      */
    for (nextNumber = 1; nextNumber < 10; nextNumber++)
    {
        if (isValid(nextNumber, h_puzzle, row, column))
        {
            h_puzzle[row][column] = nextNumber;
            if (column == 8)
            {
                if (sudokuHelper(h_puzzle, row + 1, 0))
                    return 1;
            }
            else
            {
                if (sudokuHelper(h_puzzle, row, column + 1))
                    return 1;
            }
            h_puzzle[row][column] = 0;
        }
    }
    return 0;
}

/*
  * Checks to see if a particular value is presently valid in a
  * given position.
  */
int isValid(int number, int h_puzzle[][9], int row, int column)
{
    int i;
    int modRow = 3 * (row / 3);
    int modCol = 3 * (column / 3);
    int row1 = (row + 2) % 3;
    int row2 = (row + 4) % 3;
    int col1 = (column + 2) % 3;
    int col2 = (column + 4) % 3;

    /* Check for the value in the given row and column */
    for (i = 0; i < 9; i++)
    {
        if (h_puzzle[i][column] == number)
            return 0;
        if (h_puzzle[row][i] == number)
            return 0;
    }

    /* Check the remaining four spaces in this sector */
    if (h_puzzle[row1 + modRow][col1 + modCol] == number)
        return 0;
    if (h_puzzle[row2 + modRow][col1 + modCol] == number)
        return 0;
    if (h_puzzle[row1 + modRow][col2 + modCol] == number)
        return 0;
    if (h_puzzle[row2 + modRow][col2 + modCol] == number)
        return 0;
    return 1;
}

/*
  * Convenience function to print out the h_puzzle.
  */
void printSudoku(int h_puzzle[][9])
{
    int i = 0, j = 0;
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            if (2 == j || 5 == j)
            {
                printf("%d | ", h_puzzle[i][j]);
            }
            else if (8 == j)
            {
                printf("%d\n", h_puzzle[i][j]);
            }
            else
            {
                printf("%d ", h_puzzle[i][j]);
            }
        }
        if (2 == i || 5 == i)
        {
            puts("------+-------+------");
        }
    }
}

#endif