/**
 * \file row.cu
 *
 * \brief Row module.
 *
 * This module provides the program's logic.
 *
 * \author Guilherme Antunes - 103600
 * \author Pedro Rasinhas - 103541
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include "common.h"
#include "cuda_runtime.h"

__global__ static void bitonicSort(int *arr, int direction, int N, int K, int rowsize);
__global__ static void validateArray(int *array, int sortType, int N, int K);
__device__ static void sort(int *arr, int start, int sortType, int N, int K);
__device__ static void merge(int *arr, int start, int sortType, int N, int K);

static dim3 getBestGridSize(int iteration);
static dim3 getBestBlockSize(int iteration);
static double get_delta_time(void);

/**
 * \brief Main function.
 *
 * \param argc number of arguments
 * \param argv arguments
 *
 * \return exit status
 */
int main(int argc, char **argv)
{
    /* set up the device */
    int dev = 0;

    cudaDeviceProp deviceProp; /* to store device properties */
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); /* get device properties */
    printf("Using Device %d: %s\n", dev, deviceProp.name); /* print device name */
    CHECK(cudaSetDevice(dev)); /* set the device */

    /* parse command line */
    if (argc < 5)
    {
        printf("Usage: %s -f <file> -k <number_subsequences>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fileName = NULL; /* file name */
    int opt, sortType = 0; /* opt, sort type */
    int K; /* number of subsequences */
    int rowsize = 1024; /* row size */
    int colsize = 1024; /* column size */
    int matrixSize = rowsize * colsize; /* matrix size */

    while ((opt = getopt(argc, argv, "k:s:f:h")) != -1)
    {
        switch (opt)
        {
        case 'k': /* number of subsequences */
            K = atoi(optarg);
            break;
        case 's': /* sort type */
            sortType = atoi(optarg);
            break;
        case 'f': /* file name */
            fileName = optarg;
            break;
        case 'h': /* help */
            printf("Usage: %s -s <sort_type> -f <file> -k <number_subsequences>\n", argv[0]);
            exit(EXIT_SUCCESS);
        default:
            printf("Usage: %s -s <sort_type> -f <file> -k <number_subsequences>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    /* read file */
    int fileSize; /* file size */
    FILE *file = fopen(fileName, "rb"); /* open file */

    if (file == NULL) /* check if file is null */
    {
        printf("Error: Could not open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    if (fread(&fileSize, sizeof(int), 1, file) != 1) /* read file size */
    {
        printf("Error: Could not read file size\n");
        exit(EXIT_FAILURE);
    }
    int *data = (int *)malloc(matrixSize * sizeof(int)); /* allocate memory for data */
    if (data == NULL) /* check if data is null */
    {
        printf("Error: Could not allocate memory\n");
        exit(EXIT_FAILURE);
    }
    if (fread(data, sizeof(int), fileSize, file) != fileSize) /* read file data */
    {
        printf("Error: Could not read file data\n");
        exit(EXIT_FAILURE);
    }
    fclose(file); /* close file */

    /* reserve memory for the gpu */
    int *d_data; /* device data */
    CHECK(cudaMalloc((void **)&d_data, matrixSize * sizeof(int))); /* allocate memory for d_data */
    CHECK(cudaMemcpy(d_data, data, matrixSize * sizeof(int), cudaMemcpyHostToDevice)); /* copy data to d_data */

    int numMerges = log2(K); /* number of merges */

    dim3 gridSize = getBestGridSize(numMerges); /* get best grid size */
    dim3 blockSize = getBestBlockSize(numMerges); /* get best block size */

    (void)get_delta_time();

    bitonicSort<<<gridSize, blockSize>>>(d_data, sortType, matrixSize, K, rowsize); /* bitonic sort */
    CHECK(cudaDeviceSynchronize()); /* synchronize device */
    CHECK(cudaGetLastError());  /* check for errors */

    double dt = get_delta_time(); /* get delta time */
    printf("GPU time: %f s\n", dt); /* print time */

    validateArray<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(d_data, sortType, matrixSize, rowsize); /* validate array */
    CHECK(cudaDeviceSynchronize()); /* synchronize device */
    CHECK(cudaGetLastError()); /* check for errors */

    free(data); /* free data */
    CHECK(cudaFree(d_data)); /* free d_data */
    CHECK(cudaDeviceReset()); /* reset device */

    return EXIT_SUCCESS;
}

/**
 * \brief Function bitonicSort.
 *
 * Its role is to sort an integer array using the bitonic sort algorithm.
 *
 * \param arr array
 * \param sortType sort type
 * \param N number of elements
 * \param K number of subsequences
 * \param rowsize row size
 */
__global__ static void bitonicSort(int *arr, int sortType, int N, int K, int rowsize)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    int size = N / K;

    for (int i = 0; (1 << i) < K + 1; i++)
    {
        if (idx * (1 << i) * N / K >= N)
        {
            return;
        }
        if (i == 0)
        {
            if (idx % 2 == 0)
            {
                sort(arr, idx * (1 << i) * N / K, sortType, size, rowsize);
            }
            else
            {
                sort(arr, idx * (1 << i) * N / K, !sortType, size, rowsize);
            }
        }
        else
        {
            if (idx % 2 == 0)
            {
                merge(arr, idx * (1 << i) * N / K, sortType, size, rowsize);
            }
            else
            {
                merge(arr, idx * (1 << i) * N / K, !sortType, size, rowsize);
            }
        }
        size <<= 1;
        __syncthreads();
    }
}

/**
 * \brief Function sort.
 *
 * Its role is to sort an integer array.
 *
 * \param arr array
 * \param start start index
 * \param sortType sort type
 * \param N number of elements
 * \param K row size
 */
__device__ static void sort(int *arr, int start, int sortType, int N, int K)
{   
    for (int i = 2; i <= N; i <<= 1)
    {
        for (int j = 0; j < N; j += i) {
            merge(arr, start + j, ((j / i % 2 != 0) ^ !sortType), i, K);
        }
    }
}

/**
 * \brief Function merge.
 *
 * Its role is to merge two subsequences.
 *
 * \param arr array
 * \param start start index
 * \param sortType sort type
 * \param N number of elements
 * \param K row size
 */
__device__ static void merge(int *arr, int start, int sortType, int N, int K)
{
    for (int j = N >> 1; j > 0; j >>= 1)
    {
        for (int k = 0; k < N; k += (j << 1))
        {
            for (int l = k; l < j + k; l++)
            { 
                if (arr[start + l] > arr[start + l + j] ^ sortType)
                {
                    int temp = arr[start + l];
                    arr[start + l] = arr[start + l + j];
                    arr[start + l + j] = temp;
                }
            }
        }
    }
}

/**
 *  \brief Function validateArray.
 *
 *  Its role is to validate an integer array.
 *
 * \param array array to validate
 * \param sortType sort type
 * \param N number of elements
 * \param K row size
 */
__global__ static void validateArray(int *array, int sortType, int N, int K)
{
    int actual_element, last_element = array[0];
    int i;
    for (i = 0; i < N; i++)
    {
        actual_element = array[i];
        if ((last_element > actual_element ^ sortType) && last_element != actual_element)
        {
            printf("Error in position %d between element %d and %d\n", i, last_element, actual_element);
            break;
        }
        last_element = actual_element;
    }
    if (i == N)
    {
        printf("Everything is OK!\n");
    }
    else
    {
        printf("Something went wrong!\n");
    }
};

/**
 * \brief Grid options.
 */
static dim3 gridOptions[11] = {
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 0, 1 << 0, 1 << 0),
};

/**
 * \brief Block options.
 */
static dim3 blockOptions[11] = {
    dim3(1 << 0, 1 << 0, 1 << 0),
    dim3(1 << 1, 1 << 0, 1 << 0),
    dim3(1 << 2, 1 << 0, 1 << 0),
    dim3(1 << 3, 1 << 0, 1 << 0),
    dim3(1 << 4, 1 << 0, 1 << 0),
    dim3(1 << 5, 1 << 0, 1 << 0),
    dim3(1 << 6, 1 << 0, 1 << 0),
    dim3(1 << 7, 1 << 0, 1 << 0),
    dim3(1 << 8, 1 << 0, 1 << 0),
    dim3(1 << 9, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
};

/**
 * \brief Get the best grid size for the given iteration.
 *
 * \param iteration iteration
 *
 * \return best grid size
 */
static dim3 getBestGridSize(int iteration)
{
    return gridOptions[iteration];
};

/**
 * \brief Get the best block size for the given iteration.
 *
 * \param iteration iteration
 *
 * \return best block size
 */
static dim3 getBestBlockSize(int iteration)
{
    return blockOptions[iteration];
};

/**
 * \brief Get the process time that has elapsed since last call of this time.
 *
 * \return process elapsed time
 */
static double get_delta_time(void)
{
    static struct timespec t0, t1;

    t0 = t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        perror("clock_gettime");
        exit(1);
    }
    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}