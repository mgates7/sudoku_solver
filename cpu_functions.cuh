#ifndef _CPU_FUNCTIONS
#define _CPU_FUNCTIONS

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <math.h>

// Forward declarations of CPU functions
int cpu_solveSudoku(int *elems);
int isValid(int number, int *puzzleElem, int row, int column);
int sudokuHelper(int *puzzleElem, int row, int column);
void initializePuzzle(int puzzleNum, char *fileName, int *elems);
void printPuzzle(Puzzle p);

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
int cpu_solveSudoku(int *elems)
{
    return sudokuHelper(elems, 0, 0);
}

/*
  * A recursive function that does all the gruntwork in solving
  * the h_puzzle.
  */
int sudokuHelper(int *puzzleElem, int row, int column)
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
    if (puzzleElem[row * 9 + column])
    {
        if (column == 8)
        {
            if (sudokuHelper(puzzleElem, row + 1, 0))
                return 1;
        }
        else
        {
            if (sudokuHelper(puzzleElem, row, column + 1))
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
        if (isValid(nextNumber, puzzleElem, row, column))
        {
            puzzleElem[row * 9 + column] = nextNumber;
            if (column == 8)
            {
                if (sudokuHelper(puzzleElem, row + 1, 0))
                    return 1;
            }
            else
            {
                if (sudokuHelper(puzzleElem, row, column + 1))
                    return 1;
            }
            puzzleElem[row * 9 + column] = 0;
        }
    }
    return 0;
}

/*
  * Checks to see if a particular value is presently valid in a
  * given position.
  */
int isValid(int number, int *puzzleElem, int row, int column)
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
        if (puzzleElem[i * 9 + column] == number)
            return 0;
        if (puzzleElem[row * 9 + i] == number)
            return 0;
    }

    /* Check the remaining four spaces in this sector */
    if (puzzleElem[(row1 + modRow) * 9 + (col1 + modCol)] == number)
        return 0;
    if (puzzleElem[(row2 + modRow) * 9 + (col1 + modCol)] == number)
        return 0;
    if (puzzleElem[(row1 + modRow) * 9 + (col2 + modCol)] == number)
        return 0;
    if (puzzleElem[(row2 + modRow) * 9 + (col2 + modCol)] == number)
        return 0;
    return 1;
}

void initializePuzzle(int puzzleNum, char *fileName, int *elems)
{
    char temp_ch;
    int i, j;
    long int offset = (puzzleNum * 81) + puzzleNum; // Accounts for "\n" at end of each line in file.
    FILE *sudokuFile;
    sudokuFile = fopen(fileName, "r");

    fseek(sudokuFile, offset, SEEK_SET);

    if (sudokuFile == NULL)
    {
        printf("Couldn't open test file for reading!");
        return;
    }

    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            fscanf(sudokuFile, "%c", &temp_ch);
            *elems = atoi(&temp_ch);
            elems++;
        }
    }
}

void printPuzzle(Puzzle p)
{
    int i, j;
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            printf("%d ", p.elements[i * 9 + j]);
        }
        printf("\n");
    }
}

#endif