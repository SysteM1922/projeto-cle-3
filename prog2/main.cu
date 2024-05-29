/**
 * \file main.cu
 *
 * \brief Main module.
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

__global__ static void global_sort(int *arr, int direction, int N, int K, int iteration);
__device__ static void device_sort(int *arr, int idx, int N, int direction);
__device__ static void device_merge(int *arr, int idx, int N, int direction);
static dim3 getBestGridSize(int iteration);
static dim3 getBestBlockSize(int iteration);
static double get_delta_time(void);
__device__ static void swap(int *a, int *b, int sortType);
__device__ static void merge(int *array, int size, int sortType);
__device__ static void sort(int *array, int size, int sortType);
__device__ static void validateArray(int *array, int size, int sortType);

int main(int argc, char **argv)
{
    /* set up the device */
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    /* parse command line */
    if (argc < 5)
    {
        printf("Usage: %s -f <file> -k <number_of_subsequences>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fileName = NULL;
    int opt, sortType = 0, k;
    int matrixSize = 1024 * 1024;

    while ((opt = getopt(argc, argv, "k:s:f:h")) != -1)
    {
        switch (opt)
        {
        case 'k': /* k value */
            k = atoi(optarg);
            break;
        case 's': /* sort type */
            sortType = atoi(optarg);
            break;
        case 'f': /* file name */
            fileName = optarg;
            break;
        case 'h': /* help */
            printf("Usage: %s -s <sort_type> -f <file> -k <number_of_subsequences>\n", argv[0]);
            exit(EXIT_SUCCESS);
        default:
            printf("Usage: %s -s <sort_type> -f <file> -k <number_of_subsequences>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    /* read file */
    int fileSize;
    FILE *file = fopen(fileName, "rb");

    if (file == NULL)
    {
        printf("Error: Could not open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    if (fread(&fileSize, sizeof(int), 1, file) != 1)
    {
        printf("Error: Could not read file size\n");
        exit(EXIT_FAILURE);
    }
    printf("File size: %d\n", fileSize);
    int *data = (int *)malloc(matrixSize * sizeof(int));
    if (data == NULL)
    {
        printf("Error: Could not allocate memory\n");
        exit(EXIT_FAILURE);
    }
    if (fread(data, sizeof(int), fileSize, file) != fileSize)
    {
        printf("Error: Could not read file data\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);

    /* reserve memory for the gpu */
    int *d_data;
    CHECK(cudaMalloc((void **)&d_data, matrixSize * sizeof(int)));
    CHECK(cudaMemcpy(d_data, data, matrixSize * sizeof(int), cudaMemcpyHostToDevice));

    int numMerges = log2(k);
    
    dim3 gridSize = getBestGridSize(numMerges);
    dim3 blockSize = getBestBlockSize(numMerges);

    (void)get_delta_time();

    int i;
    for (i = 0; i < numMerges + 1; i++)
    {
        printf("Iteration: %d\n", i);
        global_sort<<<gridSize, blockSize>>>(d_data, sortType, fileSize, k, i);
        CHECK(cudaDeviceSynchronize());
    }
    printf("Last iteration\n");
    global_sort<<<gridSize, blockSize>>>(d_data, sortType, fileSize, k, -1);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    double dt = get_delta_time();
    printf("GPU time: %f s\n", dt);

    free(data);
    CHECK(cudaFree(d_data));
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

__global__ static void global_sort(int *arr, int direction, int N, int K, int iteration)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    if (idx * (1 << iteration) * N/K >= N)
    {
        return;
    }

    if (iteration == 0)
    {
        device_sort(arr, idx * N/K, N/K, idx % 2 == direction);
    }
    else if (iteration == -1)
    {
        if (idx == 0)
        {
            validateArray(arr, N, direction);
        }
        return;
    }
    else
    {
        device_merge(arr, idx * (1 << iteration) * N/K, (1 << iteration) * N/K, idx % 2 == direction);
    }
    printf("From %d to %d\n", idx * (1 << iteration) * N/K, idx * (1 << iteration) * N/K + (1 << iteration) * N/K);
}

__device__ static void device_sort(int *arr, int idx, int N, int direction)
{
    sort(arr + idx, N, direction);
}

__device__ static void device_merge(int *arr, int idx, int N, int direction)
{
    merge(arr + idx, N, direction);
}

dim3 gridOptions[21] = {
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

dim3 blockOptions[21] = {
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
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
    dim3(1 << 10, 1 << 0, 1 << 0),
};

dim3 getBestGridSize(int iteration)
{
    return gridOptions[iteration];
}

dim3 getBestBlockSize(int iteration)
{
    return blockOptions[iteration];
};

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

/**
 *  \brief Function swap.
 *
 *  Its role is to swap two elements of an integer array.
 *
 *  \param a pointer to the first element
 *  \param b pointer to the second element
 *  \param sortType sort type
 */

__device__ static void swap(int *a, int *b, int sortType)
{
    if (sortType == (*a > *b))
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

/**
 *  \brief Function merge.
 *
 *  Its role is to merge two integer arrays.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */

__device__ static void merge(int *array, int size, int sortType)
{
    if (size > 1)
    {
        int i;
        int half = size / 2;
        for (i = 0; i < half; i++)
        {
            swap(&array[i], &array[i + half], sortType);
        }
        merge(array, half, sortType);
        merge(array + half, size - half, sortType);
    }
}

/**
 *  \brief Function sort.
 *
 *  Its role is to sort an integer array.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */

__device__ static void sort(int *array, int size, int sortType)
{
    if (size > 1)
    {
        int half = size / 2;
        sort(array, half, 1);
        sort(array + half, size - half, 0);
        merge(array, size, sortType);
    }
}

/**
 *  \brief Function validateArray.
 *
 *  Its role is to validate an integer array.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */
__device__ static void validateArray(int *array, int size, int sortType)
{
    int j;
    for (j = 0; j < size - 1; j++)
    {
        if (sortType == (array[j] < array[j + 1]) && array[j] != array[j + 1])
        {
            printf("Error in position %d between element %d and %d\n", j, array[j], array[j + 1]);
            break;
        }
    }
    if (j == (size - 1))
    {
        printf("Everything is OK!\n");
    }
    else
    {
        printf("Something went wrong!\n");
    }
}