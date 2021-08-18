#ifndef __algorithms_h__
#define __algorithms_h__

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

template <typename Key, typename T>
static void counting_sort(
	typename std::vector<T>::iterator begin,
	typename std::vector<T>::iterator end,
	typename std::vector<T>::iterator output,
	typename std::vector<T>::iterator output_end,
	unsigned b,
	unsigned k)
{
	unsigned counts[1<<b];
	unsigned mask = (1<<b)-1;
	
	memset(counts, 0, sizeof(unsigned)*(1<<b));
	
	for (auto it = begin; it < end; it++) {
		unsigned idx = ((Key)(*it) >> (k*b)) & mask;
		counts[idx]++;
	}
	
	for (unsigned i = 1; i < 1<<b; i++) {
		counts[i] += counts[i-1];
	}
	
	for (auto it = end-1; it >= begin; it--) {
		unsigned idx = ((Key)(*it) >> (k*b)) & mask;
		unsigned pos = --counts[idx];
		*(output + pos) = *it;
	}
}

template <typename Key, typename T>
static void parallel_counting_sort(
	typename std::vector<T>::iterator begin,
	typename std::vector<T>::iterator end,
	typename std::vector<T>::iterator output,
	typename std::vector<T>::iterator output_end,
	unsigned b,
	unsigned k)
{
	#define NUM_SORT_THREADS 4
	unsigned counts[NUM_SORT_THREADS][1<<b];
	unsigned starts[NUM_SORT_THREADS][1<<b];
	unsigned mask = (1<<b)-1;
	
	memset(counts, 0, sizeof(unsigned)*(1<<b)*NUM_SORT_THREADS);
	
	#pragma omp parallel num_threads(NUM_SORT_THREADS)
	{
		#pragma omp for
		for (auto it = begin; it < end; it++) {
			unsigned idx = ((Key)(*it) >> (k*b)) & mask;
			unsigned tid = omp_get_thread_num();
			counts[tid][idx]++;
		}
		
		#pragma omp single
		{
			unsigned previous_starts = 0;
			unsigned previous_counts = 0;
			for (unsigned i = 0; i < (unsigned)1<<b; i++) {
				for (unsigned tid = 0; tid < NUM_SORT_THREADS; tid++) {
					starts[tid][i] = previous_starts + previous_counts;
					previous_starts = starts[tid][i];
					previous_counts = counts[tid][i];
				}
			}
		}
		
		#pragma omp for
		for (auto it = begin; it < end; it++) {
			unsigned idx = ((Key)(*it) >> (k*b)) & mask;
			unsigned tid = omp_get_thread_num();
			unsigned pos = starts[tid][idx]++;
			*(output + pos) = *it;
		}
	}
}

template <typename Key, typename T>
static void radix_sort(
	typename std::vector<T>::iterator begin,
	typename std::vector<T>::iterator end,
	typename std::vector<T>::iterator scratch,
	typename std::vector<T>::iterator scratch_end,
	unsigned b)
{
	if (sizeof(Key) == sizeof(uint32_t)) {
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 0);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 1);
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 2);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 3);
	}
	else if (sizeof(Key) == sizeof(uint64_t)) {
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 0);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 1);
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 2);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 3);			
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 4);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 5);
		parallel_counting_sort<Key, T>(begin, end, scratch, scratch_end, b, 6);
		parallel_counting_sort<Key, T>(scratch, scratch_end, begin, end, b, 7);		
	}
}
	
uint32_t bit_interleave_32(const uint32_t &x, const uint32_t &y, const uint32_t &z);
uint64_t bit_interleave_64(const uint64_t &x, const uint64_t &y, const uint64_t &z);
	
#endif
