
#include "algorithms.h"

#include <cstring>
#include <omp.h>
#include <immintrin.h>

unsigned bit_interleave_32(const unsigned &x, const unsigned &y, const unsigned &z)
{
	const unsigned a = _pdep_u32(x, 0x49249249 /* 0b01001001001001001001001001001001 */);
	const unsigned b = _pdep_u32(y, 0x92492492 /* 0b10010010010010010010010010010010 */);
	const unsigned c = _pdep_u32(z, 0x24924924 /* 0b00100100100100100100100100100100 */);
	return a | b | c;
}

uint64_t bit_interleave_64(const uint64_t &x, const uint64_t &y, const uint64_t &z)
{
	const uint64_t a = _pdep_u64(x, 0x9249249249249249);
	const uint64_t b = _pdep_u64(y, 0x2492492492492492);
	const uint64_t c = _pdep_u64(z, 0x4924924924924924);
	return a | b | c;
}
