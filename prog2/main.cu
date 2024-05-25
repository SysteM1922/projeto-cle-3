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

#include "common.h"
#include "cuda_runtime.h"

#include "bitonicSort.h"

#define SQUARE_MATRIX_DEGREE 1024

static double get_delta_time(void);

int main(int argc, char **argv)
{
    /* set up the device */
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev));

    /* parse command line */
    if (argc < 3)
    {
        printf("Usage: %s -f <file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fileName = NULL;
    int opt, sortType = 0;

    while ((opt = getopt(argc, argv, "s:f:h")) != -1)
    {
        switch (opt)
        {
        case 's':   /* sort type */
            sortType = atoi(optarg);
            break;
        case 'f':   /* file name */
            fileName = optarg;
            break;
        case 'h':   /* help */
            printf("Usage: %s -s <sort_type> -f <file>\n", argv[0]);
            exit(EXIT_SUCCESS);
        default:
            printf("Usage: %s -s <sort_type> -f <file>\n", argv[0]);
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

    int *data = (int *)malloc(fileSize * sizeof(int));
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
    int matrixSize = SQUARE_MATRIX_DEGREE * SQUARE_MATRIX_DEGREE;
    int *d_data;
    CHECK(cudaMalloc((void **)&d_data, matrixSize * sizeof(int)));
    CHECK(cudaMemcpy(d_data, data, matrixSize * sizeof(int), cudaMemcpyHostToDevice));

    /* sort */
    (void) get_delta_time();
    
    



    return EXIT_SUCCESS;
}

__global__ static void sort(int *arr, int direction, int N, int iteration) {
}

__global__ static void merge(int *arr, int direction, int N, int iteration) {
}

static double get_delta_time(void)
{
  static struct timespec t0,t1;

  t0 = t1;
  if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
  {
    perror("clock_gettime");
    exit(1);
  }
  return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}