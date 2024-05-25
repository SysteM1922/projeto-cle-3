/**
 * \file bitonicSort.c
 * 
 * \brief Bitonic Sort module.
 * 
 * This module provides the necessary implementations to sort an array using the Bitonic Sort algorithm.
 * 
 * \author Guilherme Antunes - 103600
 * \author Pedro Rasinhas - 103541
*/

#include <stdio.h>
#include <stdlib.h>

#include "bitonicSort.h"

/**
 *  \brief Function swap.
 *
 *  Its role is to swap two elements of an integer array.
 *
 *  \param a pointer to the first element
 *  \param b pointer to the second element
 *  \param sortType sort type
 */

void swap(int *a, int *b, int sortType)
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

void merge(int *array, int size, int sortType)
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

void sort(int *array, int size, int sortType)
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
void validateArray(int *array, int size, int sortType)
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