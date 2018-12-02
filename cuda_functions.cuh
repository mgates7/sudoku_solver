#ifndef _CUDA_FUNCTIONS
#define _CUDA_FUNCTIONS

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// GPU Directives
#define IMUL(a, b) __mul24(a, b)

// Function definitions
__device__ int d_isValid(const int blockId, const int threadIdX, const int threadIdY ,bool bitmap[][9][9]);

/* 
This function's main purpose is to determine how many blocks and threads
we will need in total.
 */
__global__ void bitmapSet(Puzzle d_puzzle)
{
    const int tid = IMUL(blockDim.x, blockIdx.x) + IMUL(threadIdx.y, 9) + threadIdx.x; // Total thread
    const int b_tid = IMUL(threadIdx.y, 9) + threadIdx.x; // Thread within the block
    //const int threadN = IMUL(blockDim.x, gridDim.x);

   
    /////////////
    // PHASE I //
    /////////////

    // Phase description: convert the puzzle to its associated bitmap so we
    // can store more puzzles in shared memory without losing information.

    __shared__ bool bitmap[9][9][9];  // bitmap[blockIdx.x][threadIdx.y][threadIdx.x]
    __shared__ bool empties[9][9][9]; // Location of empties (i.e. when d_puzzle == 0) , 1 = Empty, 0 = Not Empty

    if (d_puzzle[b_tid] == (blockIdx.x + 1))
    {
        bitmap[blockIdx.x][threadIdx.y][threadIdx.x] = 1;
        empties[blockIdx.x][threadIdx.y][threadIdx.x] = 0;
    }
    else if (d_puzzle[b_tid] == 0)
    {
        bitmap[blockIdx.x][threadIdx.y][threadIdx.x] = 0;
        empties[blockIdx.x][threadIdx.y][threadIdx.x] = 1;
    }
    else
    {
        bitmap[blockIdx.x][threadIdx.y][threadIdx.x] = 0;
        empties[blockIdx.x][threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads(); // Wait for bitmaps to all be completed before starting Phase II

    //////////////
    // PHASE II //
    //////////////

    // Phase description: Find all potential solutions through parallel backtracking.

    // if (empties[blockIdx.x][threadIdx.y][threadIdx.x])
    // {
    //     if(d_isValid(blockIdx.x, threadIdx.x, threadIdx.y, bitmap)){ 
    //         // printf("Valid\n");
    //         bitmap[blockIdx.x][threadIdx.y][threadIdx.x] = 1;
    //         empties[blockIdx.x][threadIdx.y][threadIdx.x] = 0;
    //     }
        

    // }

    // printf("blockID = %d, ThreadID = %d, bitmap = %d, empty = %d\n",
    //     blockIdx.x, b_tid, bitmap[blockIdx.x][threadIdx.y][threadIdx.x], empties[blockIdx.x][threadIdx.y][threadIdx.x]);
}

__device__ int d_isValid(const int blockId, const int threadIdX, const int threadIdY ,bool bitmap[][9][9]){

    // Compute row & column of each thread
    int row = threadIdY;
    int column = threadIdX;

    if ((blockId == 0) && (threadIdX == 0) && (threadIdY == 8)) {
        for (int i=0;i<9;i++){
            for (int j=0;j<9;j++){
                printf("%d ", bitmap[blockId][i][j]);
            }
            printf("\n");
        } 
    }

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
        if (bitmap[blockId][i][column])
            return 0;
        if (bitmap[blockId][row][i])
            return 0;
    }

    /* Check the remaining four spaces in this sector */
    if (bitmap[blockId][row1 + modRow][col1 + modCol])
        return 0;
    if (bitmap[blockId][row2 + modRow][col1 + modCol])
        return 0;
    if (bitmap[blockId][row1 + modRow][col2 + modCol])
        return 0;
    if (bitmap[blockId][row2 + modRow][col2 + modCol])
        return 0;
    return 1;

}

// __device__ int sudokuHelper(bool bitmap[], int row, int column)
// {
//     int nextNumber;
//     /*
//      * Have we advanced past the h_puzzle?  If so, hooray, all
//      * previous cells have valid contents!  We're done!
//      */
//     if (row == 9)
//     {
//         return 1;
//     }

//     /*
//      * Is this element already set?  If so, we don't want to
//      * change it.
//      */
//     if (h_puzzle[row][column])
//     {
//         if (column == 8)
//         {
//             if (sudokuHelper(h_puzzle, row + 1, 0))
//                 return 1;
//         }
//         else
//         {
//             if (sudokuHelper(h_puzzle, row, column + 1))
//                 return 1;
//         }
//         return 0;
//     }

//     /*
//      * Iterate through the possible numbers for this empty cell
//      * and recurse for every valid one, to test if it's part
//      * of the valid solution.
//      */
//     for (nextNumber = 1; nextNumber < 10; nextNumber++)
//     {
//         if (isValid(nextNumber, h_puzzle, row, column))
//         {
//             h_puzzle[row][column] = nextNumber;
//             if (column == 8)
//             {
//                 if (sudokuHelper(h_puzzle, row + 1, 0))
//                     return 1;
//             }
//             else
//             {
//                 if (sudokuHelper(h_puzzle, row, column + 1))
//                     return 1;
//             }
//             h_puzzle[row][column] = 0;
//         }
//     }
//     return 0;
// }

#endif