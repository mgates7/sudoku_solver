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
#define PUZZLE_SIZE 81
#define OPTIONS 2 // CPU & GPU

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

    // Host variables...I will be opening file and reading on host, then transfering to global memory on GPU
    int h_puzzle[9][9];
    char temp_ch;
    char *fileName;
    FILE *sudokuFile;
    int i, j, num;
    int cpu_success[NUM_PUZZLES];

    // bool *h_bitmap;

    // GPU Variables
    int *d_puzzle;
    bool *bitmap;
    bool *empties;

    fileName = "test.txt"; // Where the test h_puzzles will be.
    sudokuFile = fopen(fileName, "r");

    if (sudokuFile == NULL)
    {
        printf("Couldn't open test file for reading!");
        return 1;
    }

    OPTION = 0;
    for (num = 0; num < NUM_PUZZLES; num++)
    {
        // Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));

        // Allocate GPU memory
        size_t allocSize_int = PUZZLE_SIZE * sizeof(int); // NUM_PUZZLES * PUZZLE_SIZE * sizeof(int)
        size_t allocSize_bool = 9 * PUZZLE_SIZE * sizeof(bool);
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_puzzle, allocSize_int));
        CUDA_SAFE_CALL(cudaMalloc((void **)&bitmap, allocSize_bool));
        CUDA_SAFE_CALL(cudaMalloc((void **)&empties, allocSize_bool));

        for (i = 0; i < 9; i++)
        {
            for (j = 0; j < 9; j++)
            {
                if ((j == 8) && (i == 8))
                    fscanf(sudokuFile, "%c\n", &temp_ch);
                else
                    fscanf(sudokuFile, "%c", &temp_ch);

                h_puzzle[i][j] = atoi(&temp_ch);
            }
        }

        // Transfer the puzzle to the GPU memory
        CUDA_SAFE_CALL(cudaMemcpy(d_puzzle, h_puzzle, allocSize, cudaMemcpyHostToDevice));

        printf("\nSolving Puzzle #%d (CPU)\n", num);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        cpu_success[num] = cpu_solveSudoku(h_puzzle);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        time_stamp[OPTION][num] = diff(time1, time2);

        OPTION++;
        // Call CUDA Kernel
        printf("Solving Puzzle #%d (GPU)\n\n", num);

        // Set up thread/block hierarchy
        dim3 threadsPerBlock(9, 9); // 9x9 Puzzle
        int numBlocks = 9;

        // Create the cuda events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // Record event on the default stream
        cudaEventRecord(start, 0);

        gpu_solveSudoku<<<numBlocks, threadsPerBlock>>>(d_puzzle, bitmap, empties);

        cudaEventRecord(stop, 0);

        // Check for errors during launch
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        // Transfer the results back to the host
        CUDA_SAFE_CALL(cudaMemcpy(h_puzzle, d_puzzle, allocSize, cudaMemcpyDeviceToHost));

        cudaFree(d_puzzle);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_gpu, start, stop);
        printf("\nGPU time: %f (nsec)\n", 1000000 * elapsed_gpu);
        //time_stamp[OPTION][num] = (struct timespec)(1000000*elapsed_gpu);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Close file and open again

    /* Output times */
    printf("\n\nPuzzle #, CPU, GPU\n");
    for (i = 0; i < NUM_PUZZLES; i++)
    {
        printf("\nPuzzle #%d, ", i);
        for (j = 0; j < OPTIONS; j++)
        {
            if (j != 0)
                printf(", ");
            printf("%ld", (long int)((double)(GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
        }
    }

    // Checks to make sure the h_puzzles were solved correctly
    for (i = 0; i < NUM_PUZZLES; i++)
    {
        if (cpu_success[i] != 1)
        {
            printf("\nError in solving h_puzzle (CPU): %d", i);
        }

        // GPU Case
    }

    printf("\n");

    return 0;
}
