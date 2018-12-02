/*
 * sudokuSolver.cu
 * 
 * Command used for accessing compute node:
 * qrsh -l gpus=1 -l gpu_type=M2070
 *
 * Command for compiling:
 * nvcc sudokuSolver.cu -o sudokuSolver
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
    int *elements;
    bool *bitmap;
    bool *isempty;
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
        // Initialize host puzzle
        initializePuzzle(puzzleNum, fileName, h_puzzle.elements);

        //printPuzzle(h_puzzle);

        // Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));

        // Allocate GPU memory
        size_t allocSize = N * N * sizeof(int); // NUM_PUZZLES * PUZZLE_SIZE * sizeof(int)
        CUDA_SAFE_CALL(cudaMalloc(&d_puzzle.elements, allocSize));

        // Transfer the unsolved puzzle to GPU memory
        CUDA_SAFE_CALL(cudaMemcpy(d_puzzle.elements, h_puzzle.elements, allocSize, cudaMemcpyHostToDevice));

        printf("\nSolving Puzzle #%d (CPU)\n", puzzleNum);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        cpu_success[puzzleNum] = cpu_solveSudoku(h_puzzle.elements);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        time_stamp[OPTION][puzzleNum] = diff(time1, time2);

        //printPuzzle(d_puzzle);

        OPTION++;
        // Call CUDA Kernel
        printf("Solving Puzzle #%d (GPU)\n\n", puzzleNum);

        // PHASE I thread/block hierarchy
        dim3 threadsPerBlock(9, 9); // 9x9 Puzzle
        dim3 blocksPerGrid(9);

        // Create the cuda events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record event and time on the default stream
        cudaEventRecord(start, 0);

        // Set bitmaps and isEmpty arrays
        bitmapSet<<<blocksPerGrid, threadsPerBlock>>>(d_puzzle);

        cudaEventRecord(stop, 0);

        // Check for errors during launch
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        // Transfer the results back to the host
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle, d_puzzle, allocSize, cudaMemcpyDeviceToHost));

        cudaFree(d_puzzle);

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

