#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <errno.h>
#include <math.h>

#include "common.h"

#define MAX_FILE_SIZE 1 << 24

struct UserInputArguments;

typedef struct UserInputArguments {

    char* filename;
    int direction;

} UserInputArguments;

void printUsage(char *cmdName);
bool validateListOrder(const int *numbersList, int numbersListSize, bool ascending);

void readBinaryFileToMemory(char* filename, int* numbersReadArray, int* numbersReadCount);
UserInputArguments readInputArguments(int argc, char *argv[]);

__global__ static void firstStep(int *arr, int direction, int N, int iteration);
__global__ static void secondStep(int *arr, int colSize, int direction, int N, int iteration);

__device__ void writeItemInSubArray(int* originalArray, int index, int value, int N);
__device__ int* obtainItemInSubArray(int* originalArray, int index, int N);

__device__ void merge(int arr[], int leftIndex, int middleOfArray, int rightIndex, int direction, int N);
__device__ int obtainMin(int x, int y);

static double get_delta_time(void);

dim3 getOptimalGridSize(int iteration);
dim3 getOptimalBlockSize(int iteration);


int main(int argc, char** argv) {

    int deviceId = 0;

    // Device properties
    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, deviceId));

    printf("Using Device %d: %s\n", deviceId, deviceProp.name);
    CHECK (cudaSetDevice (deviceId));

    // Read input arguments
    UserInputArguments userInputArguments = readInputArguments(argc, argv);

    // Obtain filename
    char* filename = userInputArguments.filename;

    // Initialize array to store numbers read from file
    int* numbersArrayHost = (int*) calloc(MAX_FILE_SIZE, sizeof(int));

    // Initialize number of numbers read from file
    int readNumbersCount = 0;

    // Read numbers from file to memory
    readBinaryFileToMemory(filename, numbersArrayHost, &readNumbersCount);

    int* numbersArrayDevice;

    // Allocate memory in device
    CHECK (cudaMalloc((void**) &numbersArrayDevice, readNumbersCount*sizeof(int)));

    // Copy numbers from host to device
    CHECK (cudaMemcpy(numbersArrayDevice, numbersArrayHost, readNumbersCount*sizeof(int), cudaMemcpyHostToDevice));

    // Obtain N
    int N = (int) sqrt(readNumbersCount);

    // Obtain the number of iterations
    int nIterations = (int) log2(N) + 1;

    // Obtain the number of threads
    int nCurrentThreads = N;

    // Obtain column size
    int colSize = N;

    // Start timer
    get_delta_time();

    for (int iter = 0; iter < nIterations; iter++) {

        dim3 optimalGridSize = getOptimalGridSize(iter);
        dim3 optimalBlockSize = getOptimalBlockSize(iter);

        if (iter == 0) {
            firstStep<<<optimalGridSize, optimalBlockSize>>>
                    (numbersArrayDevice, userInputArguments.direction, N, iter);
        } else {
            secondStep<<<optimalGridSize, optimalBlockSize>>>
                    (numbersArrayDevice, colSize, userInputArguments.direction, N, iter);
        }
        CHECK (cudaDeviceSynchronize());

        // Update number of threads
        nCurrentThreads >>= 1;

        // Update column size
        colSize <<= 1;
    }

    // Stop timer
    printf("Time elapsed: %f\n", get_delta_time());

    int* sortedNumbersArrayHost = (int*) calloc(readNumbersCount, sizeof(int));

    // Copy numbers from device to host
    CHECK (cudaMemcpy(sortedNumbersArrayHost, numbersArrayDevice, readNumbersCount*sizeof(int), cudaMemcpyDeviceToHost));

    // Free device global memory
    CHECK (cudaFree(numbersArrayDevice));

    // Validate list order
    bool hasInvalidElements = validateListOrder(sortedNumbersArrayHost, readNumbersCount, userInputArguments.direction == 0);
    if (hasInvalidElements)
        printf("\nInvalid elements found\n");
    else
        printf("\nEverything okay\n");

    free(sortedNumbersArrayHost);
    free(numbersArrayHost);

    return 0;
}

__global__ static void firstStep(int *arr, int direction, int N, int iteration) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    // Sort the subarray (Merge Sort)
    int leftIndex, rightIndex;

    // Current size varies from 1 to n/2 (sub-arrays)
    // First merge sub-arrays of size 1 to create sorted sub-arrays of size 2,
    // Then merge sub-arrays of size 2 to create sorted sub-arrays of size 4, and so on.

    for (int currentSize = 1; currentSize <= N - 1; currentSize = 2 * currentSize) {

        for (leftIndex = 0; leftIndex < N - 1; leftIndex += 2 * currentSize) {

            int middleOfArray = min(leftIndex + currentSize - 1, N - 1);
            rightIndex = min(leftIndex + 2 * currentSize - 1, N - 1);

            merge(arr +  (1 << iteration) * idx, leftIndex, middleOfArray, rightIndex, direction, N);
        }
    }
}

__global__ static void secondStep(int *arr, int colSize, int direction, int N, int iteration) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;

    // Merge sub-sequences of integers (Merge Sort)
    merge(arr + (1 << iteration) * idx, 0, (colSize >> 1) - 1, colSize - 1, direction, N);
}


__device__ void merge(int arr[], int leftIndex, int middleOfArray, int rightIndex, int direction, int N) {

    int i, j, k;
    int startOfLeftArray = middleOfArray - leftIndex + 1;
    int startOfRightArray = rightIndex - middleOfArray;

    // Create temporary arrays - one for each half of the array.
    // Use dynamic allocation because static allocation would exceed the stack size (stack overflow).
    // With dynamic allocation, we are using the heap, not the stack!
    int *leftHalfTmp = (int*) malloc(startOfLeftArray*sizeof(int));
    int *rightHalfTmp = (int*) malloc(startOfRightArray*sizeof(int));

    // Copy data from the original array to temporary arrays leftHalfTmp[] and rightHalfTmp[]
    for (i = 0; i < startOfLeftArray; i++)
        leftHalfTmp[i] = *obtainItemInSubArray(arr, leftIndex + i, N);
    for (j = 0; j < startOfRightArray; j++)
        rightHalfTmp[j] = *obtainItemInSubArray(arr, middleOfArray + 1 + j, N);

    // Merge the temporary arrays back into one array arr[leftIndex...rightIndex]

    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = leftIndex; // Initial index of merged subarray

    // If direction is 0, then merge left and right sub-arrays in ascending order
    if (direction == 0) {
        while (i < startOfLeftArray && j < startOfRightArray) {
            if (leftHalfTmp[i] <= rightHalfTmp[j]) {
                writeItemInSubArray(arr, k, leftHalfTmp[i], N);
                i++;
            } else {
                writeItemInSubArray(arr, k, rightHalfTmp[j], N);
                j++;
            }
            k++;
        }
    }

    // If direction is 0, then merge left and right sub-arrays in descending order
    else {
        while (i < startOfLeftArray && j < startOfRightArray) {
            if (leftHalfTmp[i] >= rightHalfTmp[j]) {
                writeItemInSubArray(arr, k, leftHalfTmp[i], N);
                i++;
            } else {
                writeItemInSubArray(arr, k, rightHalfTmp[j], N);
                j++;
            }
            k++;
        }
    }

    // Copy the remaining elements of leftHalfTmp[]
    while (i < startOfLeftArray) {
        writeItemInSubArray(arr, k, leftHalfTmp[i], N);
        i++;
        k++;
    }

    // Copy the remaining elements of rightHalfTmp[]
    while (j < startOfRightArray) {
        writeItemInSubArray(arr, k, rightHalfTmp[j], N);
        j++;
        k++;
    }

    free(leftHalfTmp);
    free(rightHalfTmp);

}

__device__ int obtainMin(int x, int y) {
    return (x < y) ? x : y;
}

__device__ int* obtainItemInSubArray(int* originalArray, int index, int N) {
    // Indexing by Columns
    return &originalArray[N * (index % N) + (index / N)];
}

__device__ void writeItemInSubArray(int* originalArray, int index, int value, int N) {
    // Indexing by Columns
    originalArray[N * (index % N) + (index / N)] = value;
}

/**
 *  \brief readInputArguments function.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return userInputArguments (struct with the user input arguments)
 */
UserInputArguments readInputArguments(int argc, char *argv[]) {

    UserInputArguments userInputArguments;
    int opt;

    // Set Default value
    userInputArguments.direction = 0; // Ascending

    do {

        switch ((opt = getopt(argc, argv, ":d:h"))) {

            case 'd': /* sort direction */

                if (atoi(optarg) != 0 && atoi(optarg) != 1) {
                    fprintf(stderr, "%s: invalid sort direction\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    exit(EXIT_FAILURE);
                }
                userInputArguments.direction = (int) atoi(optarg);
                break;

            case 'h': /* help mode */
                printUsage(basename(argv[0]));
                exit(EXIT_SUCCESS);

            case '?': /* invalid option */
                fprintf(stderr, "%s: invalid option\n", basename(argv[0]));
                printUsage(basename(argv[0]));
                exit(EXIT_FAILURE);

            case -1:
                break;
        }

    } while (opt != -1);

    if (optind >= argc) {
        fprintf(stderr, "%s: no file name provided\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        exit(EXIT_FAILURE);
    }

    // strcpy(userInputArguments.filename, argv[optind]);
    userInputArguments.filename = argv[optind];

    printf("Input arguments:\n");
    printf(userInputArguments.direction == 0 ? "Ascending\n" : "Descending\n");
    printf("Column Processing\n");
    printf("\n");

    return userInputArguments;
}

/**
* \brief Reads a binary file and stores the values in an array.
*
* \param filename Name of the file to be read
* \param numbersReadArray Array to store the values read from the file
* \param numbersReadCount Number of values read from the file
*/
void readBinaryFileToMemory(char* filename, int* numbersReadArray, int* numbersReadCount) {

    FILE* numbersFile = fopen(filename, "rb");

    if (numbersFile == NULL) {
        printf("Error reading binary file '%s': %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    int currentNumber;
    int buffer; // int -> 4 bytes

    // Read number of values
    if (fread(&buffer, sizeof(int), 1, numbersFile) != 0) {

        long numberOfElements = buffer;

        for (int i = 0; i < numberOfElements; i++) {

            if (fread(&buffer, sizeof(int), 1, numbersFile) != 0) {
                currentNumber = buffer;

                numbersReadArray[*numbersReadCount] = currentNumber;
                *numbersReadCount += 1;
            }
        }
    }

    fclose(numbersFile);
}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */
void printUsage(char *cmdName) {

    fprintf(stderr, "\nSynopsis: %s [OPTIONS] <FILES>\n"
                    "  OPTIONS:\n"
                    "  -d sortDir    --- set the sort direction (default: 0 = Ascending)\n"
                    "  -h help       --- print this help\n", cmdName);

}

/**
 * \brief Validate if the list is in ascending or descending order.
 *
 * \param numbersList List of numbers to validate order
 * \param numbersListSize Size of the list
 * \param ascending True if the list should be ascending, false if descending
 *
 * \return True if the list has invalid elements, false if it is valid
 */
bool validateListOrder(const int *numbersList, int numbersListSize, bool ascending) {

    bool foundInvalidElement = false;

    // Column Processing
    int N = sqrt(numbersListSize);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N - 1; j++) {
            if (ascending) {
                if (numbersList[j * N + i] > numbersList[(j + 1) * N + i]) {
                    foundInvalidElement = true;
                    printf("Invalid element in %i. %i was before %i.\n", j, numbersList[j * N + i],
                           numbersList[(j + 1) * N + i]);
                    break;
                }
            } else {
                if (numbersList[j * N + i] < numbersList[(j + 1) * N + i]) {
                    foundInvalidElement = true;
                    printf("Invalid element in %i. %i was before %i.\n", j, numbersList[j * N + i],
                           numbersList[(j + 1) * N + i]);
                    break;
                }
            }
        }

        if (i == N-1 || foundInvalidElement) {
            break;
        }

        if (ascending) {
            if (numbersList[(N - 1) * N + i] > numbersList[i + 1]) {
                foundInvalidElement = true;
                printf("Invalid element in %i. %i was before %i.\n", (N - 1) * N + i, numbersList[(N - 1) * N + i],
                       numbersList[i + 1]);
                break;
            }
        } else {
            if (numbersList[(N - 1) * N + i] < numbersList[i + 1]) {
                foundInvalidElement = true;
                printf("Invalid element in %i. %i was before %i.\n", (N - 1) * N + i, numbersList[(N - 1) * N + i],
                       numbersList[i + 1]);
                break;
            }
        }
    }

    return foundInvalidElement;
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

dim3 getOptimalGridSize(int iteration) {

    dim3 optimalGridSize[11] = {
            dim3(1 << 9, 1 << 0, 1),
            dim3(1 << 0, 1 << 5, 1),
            dim3(1 << 1, 1 << 4, 1),
            dim3(1 << 1, 1 << 4, 1),
            dim3(1 << 1, 1 << 5, 1),
            dim3(1 << 4, 1 << 0, 1),
            dim3(1 << 3, 1 << 1, 1),
            dim3(1 << 0, 1 << 3, 1),
            dim3(1 << 0, 1 << 2, 1),
            dim3(1 << 0, 1 << 0, 1),
            dim3(1 << 0, 1 << 0, 1)
    };

    return optimalGridSize[iteration];

}

dim3 getOptimalBlockSize(int iteration) {

    dim3 optimalBlockSize[11] = {
            dim3(1 << 1, 1 << 0, 1),
            dim3(1 << 4, 1 << 0, 1),
            dim3(1 << 1, 1 << 2, 1),
            dim3(1 << 0, 1 << 2, 1),
            dim3(1 << 0, 1 << 0, 1),
            dim3(1 << 1, 1 << 0, 1),
            dim3(1 << 0, 1 << 0, 1),
            dim3(1 << 0, 1 << 0, 1),
            dim3(1 << 0, 1 << 0, 1),
            dim3(1 << 1, 1 << 0, 1),
            dim3(1 << 0, 1 << 0, 1)
    };

    return optimalBlockSize[iteration];

}