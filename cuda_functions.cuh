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
__global__ void bitmapSet(Puzzle d_puzzle);
__global__ void findSolutionBoards(Puzzle d_puzzle, bool *solutionBoard, bool *isEmpty, int num);
__global__ void solutionFind(bool *inputBoards);

__device__ int d_isValid(bool *bitmap, const int threadIdX, const int threadIdY);

__device__ bool *out_solutionBoard[9]; // Where solution boards will be put. Order does not matter.
__device__ bool *out_isEmpty[9];
//__device__ int *totalSolutions;

/////////////
// PHASE I //
/////////////

// Phase description: convert the puzzle to its associated bitmap so we
// can maximize parallelism losing information.

__global__ void bitmapSet(Puzzle d_puzzle)
{
    const int tid = IMUL(blockIdx.x, 81) + IMUL(threadIdx.y, 9) + threadIdx.x; // Total thread
    const int b_tid = IMUL(threadIdx.y, 9) + threadIdx.x;                      // Thread within the block
    //const int threadN = IMUL(blockDim.x, gridDim.x);

    // __shared__ bool bitmap[9][9][9];  // bitmap[blockIdx.x][threadIdx.y][threadIdx.x]
    // __shared__ bool empties[9][9][9]; // Location of empties (i.e. when d_puzzle == 0) , 1 = Empty, 0 = Not Empty

    if (d_puzzle.elements[b_tid] == (blockIdx.x + 1))
    {
        d_puzzle.bitmap[tid] = 1;
        d_puzzle.isEmpty[tid] = 0;
    }
    else if (d_puzzle.elements[b_tid] == 0)
    {
        d_puzzle.bitmap[tid] = 0;
        d_puzzle.isEmpty[tid] = 1;
    }
    else
    {
        d_puzzle.bitmap[tid] = 0;
        d_puzzle.isEmpty[tid] = 0;
    }
}

//////////////
// PHASE II //
//////////////

// Phase description: Find all potential solutions through parallel backtracking. Takes advantage of Dynamic Parallelism.
// (Assuming compute capability 3.5 or higher)

// Generates potential solution boards from ONE set. That way we can keep thread/block hierarchy constant.
__global__ void findSolutionBoards(Puzzle d_puzzle, bool *solutionBoard, bool *isEmpty, int num)
{
    const int tid = IMUL(blockIdx.x, 81) + IMUL(threadIdx.y, 9) + threadIdx.x; // Total thread
    const int b_tid = IMUL(threadIdx.y, 9) + threadIdx.x;                      // Thread within the block

    dim3 threadsPerBlock(9, 9); // NxN Puzzle
    dim3 blocksPerGrid(1);

    int blockId = blockIdx.x + num; // Takes care of the fact that each recursive call starts the blockId at 0.

    if (isEmpty[b_tid])
    {
        if (d_isValid(solutionBoard, threadIdx.x, threadIdx.y))
        {
            atomicAdd(d_puzzle.num_solutionBoards, 1); // Low number of threads writing, so this is OK.
            //printf("blockID: %d, threadX: %d, threadY: %d\n", blockIdx.x, threadIdx.x,threadIdx.y);

            size_t size = 9 * 9 * sizeof(bool);
            out_solutionBoard[blockId] = (bool *)malloc(size);
            bool *ptr = out_solutionBoard[blockId];
            out_isEmpty[blockId] = (bool *)malloc(size);
            bool *ptr2 = out_isEmpty[blockId];
            //d_puzzle.solutionBoards = (bool *)malloc(size);

            // Assign pointers
            //memset(out_solutionBoard[blockIdx.x], 0, size);
            //memset(out_isEmpty[blockIdx.x], 0, size);
            //memset(d_puzzle.solutionBoards, 0, size);

            //printf("%p\n", ptr);

            // Allocate solution board and original puzzle with new values updated

            for (int i = 0; i < 81; i++)
            {
                //out_solutionBoard[blockIdx.x][i] = solutionBoard[blockIdx.x * 81 + i];
                ptr[i] = solutionBoard[(blockId) * 81 + i];
                ptr2[i] = isEmpty[(blockId) * 81 + i];
                // out_isEmpty[num][i] = isEmpty[i];
                // d_puzzle.solutionBoards[i] = solutionBoard[i];
            }

            // New values
            //out_solutionBoard[blockIdx.x][b_tid] = 1;
            ptr[b_tid] = 1;
            ptr2[b_tid] = 0;
            // out_isEmpty[num][b_tid] = 0;
            // d_puzzle.solutionBoards[b_tid] = 1;

            // Recurse through each solution board.
            findSolutionBoards<<<blocksPerGrid, threadsPerBlock>>>(d_puzzle, ptr, ptr2, blockId);

            __syncthreads; // May not be needed since I'm calling a global function right after?

            // int total_partials = d_puzzle.num_solutionBoards;
            // int block_total = total_partials / 1024;
            // dim3 threadsPerBlock_3(1024); // NxN Puzzle
            // dim3 blocksPerGrid_3(block_total);

            // if ((blockIdx.x == 3) && (b_tid == 36))
            // {
            //     int sizeOfPartials[9];

            //     for (int i = 0; i < 9; i++)
            //     {
            //         sizeOfPartials[i] = sizeof(out_solutionBoard[i])/sizeof(bool);
            //         printf("Size of %d: %d\n", i, sizeOfPartials[i]);
            //     }
            // }

            //solutionFind<<<1, 1>>>(out_solutionBoard); // Each thread only calls once. Make it global since we need a global barrier.

            // if ((blockIdx.x == 3) && (b_tid == 36))
            // {
            //     for (int j = 0; j < 9; j++)
            //     {
            //         for (int i = 0; i < 9; i++)
            //         {
            //             printf("%d ", solutionBoard[blockIdx.x*81+j * 9 + i]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            // if ((blockIdx.x == 3) && (b_tid == 36))
            // {
            //     for (int j = 0; j < 9; j++)
            //     {
            //         for (int i = 0; i < 9; i++)
            //         {
            //             printf("%d ", out_solutionBoard[blockIdx.x][j * 9 + i]);
            //         }
            //         printf("\n");
            //     }
            // }

            // Sync threads then free pointers.
            free(out_solutionBoard);
            free(out_isEmpty);
            // free(d_puzzle.solutionBoards);
        }
    }
}

//__device__ int findNumSolutions()

// Checks if a given spot is valid. Passes in bitmap for given thread and checks row, column and sub-matrix for isValid information.
__device__ int d_isValid(bool *bitmap, const int threadIdX, const int threadIdY)
{
    int i;

    // Compute row & column of each thread
    int row = threadIdY;
    int column = threadIdX;

    int modRow = 3 * (row / 3);
    int modCol = 3 * (column / 3);
    int row1 = (row + 2) % 3;
    int row2 = (row + 4) % 3;
    int col1 = (column + 2) % 3;
    int col2 = (column + 4) % 3;

    /* Check for the value in the given row and column */
    for (i = 0; i < 9; i++)
    {
        if (bitmap[i * 9 + column])
            return 0;
        if (bitmap[row * 9 + i])
            return 0;
    }

    /* Check the remaining four spaces in this sector */
    if (bitmap[(row1 + modRow) * 9 + (col1 + modCol)])
        return 0;
    if (bitmap[(row2 + modRow) * 9 + (col1 + modCol)])
        return 0;
    if (bitmap[(row1 + modRow) * 9 + (col2 + modCol)])
        return 0;
    if (bitmap[(row2 + modRow) * 9 + (col2 + modCol)])
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