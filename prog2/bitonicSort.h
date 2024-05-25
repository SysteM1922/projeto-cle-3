/**
 * \file bitonicSort.h
 * 
 * \brief Bitonic Sort module.
 * 
 * This module provides the necessary functions to sort an array using the Bitonic Sort algorithm.
 * 
 * \author Guilherme Antunes - 103600
 * \author Pedro Rasinhas - 103541
*/

#ifndef BITONICSORT_H
#define BITONICSORT_H

/**
 *  \brief Function merge.
 *
 *  Its role is to merge two arrays.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */
void merge(int *array, int size, int sortType);

/**
 *  \brief Function sort.
 *
 *  Its role is to sort an array.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */
void sort(int *array, int size, int sortType);

/**
 *  \brief Function validateArray.
 *
 *  Its role is to validate an array.
 *
 *  \param array pointer to the array
 *  \param size array size
 *  \param sortType sort type
 */
void validateArray(int *array, int size, int sortType);

#endif