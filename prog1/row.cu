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
        printf("Usage: %s -f <file> -k <number_subsequences>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fileName = NULL;
    int opt, sortType = 0;
    int K;
    int rowsize = 1024;
    int colsize = 1024;
    int matrixSize = rowsize * colsize;

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

    int numMerges = log2(K);

    dim3 gridSize = getBestGridSize(numMerges);
    dim3 blockSize = getBestBlockSize(numMerges);

    (void)get_delta_time();

    bitonicSort<<<gridSize, blockSize>>>(d_data, sortType, matrixSize, K, rowsize);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    double dt = get_delta_time();
    printf("GPU time: %f s\n", dt);

    validateArray<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(d_data, sortType, matrixSize, rowsize);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    free(data);
    CHECK(cudaFree(d_data));
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

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

__device__ static void sort(int *arr, int start, int sortType, int N, int K)
{   
    for (int i = 2; i <= N; i <<= 1)
    {
        for (int j = 0; j < N; j += i) {
            merge(arr, start + j, ((j / i % 2 != 0) ^ !sortType), i, K);
        }
    }
}

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
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
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

static dim3 getBestGridSize(int iteration)
{
    return gridOptions[iteration];
};

static dim3 getBestBlockSize(int iteration)
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