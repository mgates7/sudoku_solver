/*
 * sudokuSolver.cu
 * 
 * Command used for accessing compute node:
 * qrsh -l gpus=1 -l gpu_c=3.5
 *
 * Command for compiling:
 * nvcc -arch compute_35 -rdc=true -lcudadevrt sudokuSolver.cu -o sudokuSolver
 *
 *
 * Author: Maxwell Gates
 * 
 * A parallelized backtracking sudoku solver. Solves sodoku puzzles 
 * using the backtracking method for the CPU and parallelization 
 * techniques using GPU then compares the two. Accepts input with each
 * new puzzle separated by newline. 
 * 
 * Puzzles:
 *      0-199      EASY (45 Given Numbers)
 *      200-499    MEDIUM (30 Given Numbers)
 *      500-999    HARD (25 Given Numbers)
 *
 */

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <math.h>

// CPU & GPU Directives
#define NUM_PUZZLES 1 // Number of h_puzzles to solve
#define N 9
#define OPTIONS 2 // CPU & GPU

// Sudoku specific
typedef struct
{
    int *elements;           // CPU & GPU
    bool *bitmap;            // GPU
    bool *isEmpty;           // GPU
    int *num_solutionBoards; // GPU
    bool *solutionBoards; // GPU
} Puzzle;

#include "cpu_functions.cuh"
#include "cuda_functions.cuh"

int main(int argc, char **argv)
{
    // CPU Timing Variables
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp[OPTIONS][NUM_PUZZLES + 1];
    int clock_gettime(clockid_t clk_id, struct timespec * tp);

    size_t allocSizeElem = N * N * sizeof(int);        // NUM_PUZZLES * PUZZLE_SIZE * sizeof(int)
    size_t allocSizeBool = N * (N * N) * sizeof(bool); // N sets of NxN bool matrices
    size_t allocSizeInt = sizeof(int);

    // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;

    int OPTION;

    // Initialize puzzles for host and device
    Puzzle h_puzzle;
    Puzzle d_puzzle;

    char *fileName;
    int i, j, puzzleNum;
    int cpu_success[NUM_PUZZLES];

    fileName = "test.txt"; // Where the test h_puzzles will be.

    OPTION = 0;
    for (puzzleNum = 0; puzzleNum < NUM_PUZZLES; puzzleNum++)
    {

        // Allocate arrays on host memory
        h_puzzle.elements = (int *)malloc(allocSizeElem);
        h_puzzle.bitmap = (bool *)malloc(allocSizeBool);
        h_puzzle.isEmpty = (bool *)malloc(allocSizeBool);
        h_puzzle.num_solutionBoards = (int *)malloc(allocSizeInt);

        // Initialize host puzzle
        initializePuzzle(puzzleNum, fileName, h_puzzle.elements);
        *h_puzzle.num_solutionBoards = 0;

        // Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));

        // Allocate GPU memory
        CUDA_SAFE_CALL(cudaMalloc(&d_puzzle.elements, allocSizeElem));
        CUDA_SAFE_CALL(cudaMalloc(&d_puzzle.bitmap, allocSizeBool));
        CUDA_SAFE_CALL(cudaMalloc(&d_puzzle.isEmpty, allocSizeBool));
        CUDA_SAFE_CALL(cudaMalloc(&d_puzzle.num_solutionBoards, allocSizeInt));

        // Transfer the unsolved puzzle to GPU memory
        CUDA_SAFE_CALL(cudaMemcpy(d_puzzle.elements, h_puzzle.elements, allocSizeElem, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_puzzle.num_solutionBoards, h_puzzle.num_solutionBoards, allocSizeInt, cudaMemcpyHostToDevice));

        printf("\nSolving Puzzle #%d (CPU)\n", puzzleNum);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        cpu_success[puzzleNum] = cpu_solveSudoku(h_puzzle.elements);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        time_stamp[OPTION][puzzleNum] = diff(time1, time2);

        //printPuzzle(d_puzzle,1);

        OPTION++;
        // Call CUDA Kernel
        printf("Solving Puzzle #%d (GPU)\n\n", puzzleNum);

        // Thread/block hierarchy for each phase
        dim3 threadsPerBlock_1(N, N); // NxN Puzzle
        dim3 blocksPerGrid_1(N);

        dim3 threadsPerBlock_2(N, N); // NxN Puzzle
        dim3 blocksPerGrid_2(N);

        // Create the cuda events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record event and time on the default stream
        cudaEventRecord(start, 0);

        // Set bitmaps and isEmpty arrays
        bitmapSet<<<blocksPerGrid_1, threadsPerBlock_1>>>(d_puzzle);    // PHASE I
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024); // Set aside 512 MB heap for solution boards (PHASE II).
        //for (i = 0; i < 9; i++) // Find solution boards for each 'number'
        findSolutionBoards<<<blocksPerGrid_2, threadsPerBlock_2>>>(d_puzzle, d_puzzle.bitmap, d_puzzle.isEmpty, 0); // PHASE II

        cudaEventRecord(stop, 0);

        cudaDeviceSynchronize();

        // Check for errors during launch
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        // Transfer the results back to the host
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle.elements, d_puzzle.elements, allocSizeElem, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle.bitmap, d_puzzle.bitmap, allocSizeBool, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle.isEmpty, d_puzzle.isEmpty, allocSizeBool, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle.num_solutionBoards, d_puzzle.num_solutionBoards, allocSizeInt, cudaMemcpyDeviceToHost));

        printf("%d\n", *h_puzzle.num_solutionBoards);

        // printPuzzle(h_puzzle,1);

        cudaFree(d_puzzle.elements);
        cudaFree(d_puzzle.bitmap);
        cudaFree(d_puzzle.isEmpty);
        cudaFree(d_puzzle.num_solutionBoards);

        free(h_puzzle.elements);
        free(h_puzzle.bitmap);
        free(h_puzzle.isEmpty);
        free(h_puzzle.num_solutionBoards);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_gpu, start, stop);
        printf("\nGPU time: %f (nsec)\n", 1000000 * elapsed_gpu);
        //time_stamp[OPTION][puzzleNum] = (struct timespec)(1000000*elapsed_gpu);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // // Close file and open again

    // /* Output times */
    // printf("\n\nPuzzle #, CPU, GPU\n");
    // for (i = 0; i < NUM_PUZZLES; i++)
    // {
    //     printf("\nPuzzle #%d, ", i);
    //     for (j = 0; j < OPTIONS; j++)
    //     {
    //         if (j != 0)
    //             printf(", ");
    //         printf("%ld", (long int)((double)(GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    //     }
    // }

    // // Checks to make sure the h_puzzles were solved correctly
    // for (i = 0; i < NUM_PUZZLES; i++)
    // {
    //     if (cpu_success[i] != 1)
    //     {
    //         printf("\nError in solving h_puzzle (CPU): %d", i);
    //     }

    //     // GPU Case
    // }

    // printf("\n");

    return 0;
}
