#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <ostream>
#include <iostream>
#include <cstring>
#include <type_traits>

#ifndef _MSC_VER
#ifndef __forceinline
	#define __forceinline __attribute__((always_inline)) inline
#endif
#endif

namespace smd {

template <size_t K, typename T>
struct simd {
	
	typedef T   _simd  __attribute__((vector_size (K*sizeof(T)), aligned(K*sizeof(T))));
	typedef T   _usimd __attribute__((vector_size (K*sizeof(T)), aligned(1)));
	
	enum  { size = K };
	const union { _simd v; T elem[size]; };
	
	// Constructors and assignment operators
	__forceinline simd() {}
    __forceinline simd(const simd<size, T>& other) { v = other.v; }
    __forceinline simd<size, T>& operator =(const simd<size, T>& other) { v = other.v; return *this; }

    __forceinline simd(_simd  a) : v(a) {}
    __forceinline operator const _simd&() const { return v; }
    __forceinline operator       _simd&()       { return v; }
	
	// Reduce to a primitive type
	__forceinline operator const T() const { return v[0]; }
    __forceinline operator       T()       { return v[0]; }
	
	// Cast to a vector of different type
	template<typename T2> __forceinline operator const simd<K, T2>() const {
		typedef T2 _simdt2  __attribute__((vector_size (K*sizeof(T))));
		return simd<K, T2>((_simdt2)v);
	}
	template<typename T2> __forceinline operator simd<K, T2>() {
		typedef T2 _simdt2  __attribute__((vector_size (K*sizeof(T))));
		return simd<K, T2>((_simdt2)v);
	}
	
	template<typename T2>
	static __forceinline simd<K, T> bitcast(const simd<K, T2>& x) {
		return simd<K, T>((_simd)x.v);
	}
	
	template<typename T2>
	static __forceinline simd<K, T> cast(const simd<K, T2>& x) {
		simd<K, T> y;
		for(unsigned i=0; i<K; i++) y[i] = x[i];
		return y;
	}

	static __forceinline simd<K, T> broadcast(const T a)          { return a      - (_simd){}; };
	static __forceinline simd<K, T> broadcast(const simd<1, T> a) { return a.v[0] - (_simd){}; };
	
	// Attempt to cast from size 1 vector to K2 with broadcast
	// Can't seem to get this to compile, looks like other casts when K==8 try to use this
	// Seems to compile now? needs coverage testing
	//template<size_t K2, typename _ = typename std::enable_if< K == 1 >::type >
	//__forceinline operator simd<K2, T>() { return  simd<K2, T>::broadcast(*this); }

	// Array access
	__forceinline const T& operator [](size_t index) const { /*assert(index < size);*/ return v[index]; }
    __forceinline       T& operator [](size_t index)       { /*assert(index < size);*/ return v[index]; }
	
	// Arithmetic operators
	__forceinline const simd<size, T> operator +(const simd<size, T>& other) const { return v + other.v; }
	__forceinline const simd<size, T> operator -(const simd<size, T>& other) const { return v - other.v; }
	__forceinline const simd<size, T> operator *(const simd<size, T>& other) const { return v * other.v; }
	__forceinline const simd<size, T> operator /(const simd<size, T>& other) const { return v / other.v; }
	
	__forceinline simd<size, T>& operator +=(const simd<size, T>& other) { v = other.v + v; return *this; }
	__forceinline simd<size, T>& operator -=(const simd<size, T>& other) { v = other.v - v; return *this; }
	__forceinline simd<size, T>& operator *=(const simd<size, T>& other) { v = other.v * v; return *this; }
	__forceinline simd<size, T>& operator /=(const simd<size, T>& other) { v = other.v / v; return *this; }
	
	__forceinline simd<size, T>& operator +=(const T& other) { v = other + v; return *this; }
	__forceinline simd<size, T>& operator -=(const T& other) { v = other - v; return *this; }
	__forceinline simd<size, T>& operator *=(const T& other) { v = other * v; return *this; }
	__forceinline simd<size, T>& operator /=(const T& other) { v = other / v; return *this; }
	
	__forceinline simd<size, T>& operator &=(const simd<size, T>& other) { v = other.v & v; return *this; }
	__forceinline simd<size, T>& operator |=(const simd<size, T>& other) { v = other.v | v; return *this; }
	__forceinline simd<size, T>& operator ^=(const simd<size, T>& other) { v = other.v ^ v; return *this; }
	
	__forceinline simd<size, T>& operator &=(const T& other) { v = other & v; return *this; }
	__forceinline simd<size, T>& operator |=(const T& other) { v = other | v; return *this; }
	__forceinline simd<size, T>& operator ^=(const T& other) { v = other ^ v; return *this; }

	// Load and store
	static __forceinline simd<size, T> load(const void* ptr) { return simd<size, T>(*(_simd*)ptr); };
	static __forceinline simd<size, T> loadu(const void* ptr) { simd<size, T> r; r.v = *(_usimd*)ptr; return r; };
	static __forceinline void store(void* ptr, simd<size, T>& v) { *(_simd*)ptr = v; };
	static __forceinline void storeu(void* ptr, simd<size, T>& v) { *(_usimd*)ptr = v; };
};

// Comparators
template<size_t K, typename T> __forceinline simd<K, int> operator ==(const simd<K, T>& a, const simd<K, T>& b) { return a.v == b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator !=(const simd<K, T>& a, const simd<K, T>& b) { return a.v != b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator < (const simd<K, T>& a, const simd<K, T>& b) { return a.v <  b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator >=(const simd<K, T>& a, const simd<K, T>& b) { return a.v >= b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator > (const simd<K, T>& a, const simd<K, T>& b) { return a.v >  b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator <=(const simd<K, T>& a, const simd<K, T>& b) { return a.v <= b.v; }

template<size_t K, typename T> __forceinline simd<K, int> operator ==(const          T& a, const simd<K, T>& b) { return a   == b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator !=(const          T& a, const simd<K, T>& b) { return a   != b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator < (const          T& a, const simd<K, T>& b) { return a   <  b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator >=(const          T& a, const simd<K, T>& b) { return a   >= b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator > (const          T& a, const simd<K, T>& b) { return a   >  b.v; }
template<size_t K, typename T> __forceinline simd<K, int> operator <=(const          T& a, const simd<K, T>& b) { return a   <= b.v; }

template<size_t K, typename T> __forceinline simd<K, int> operator ==(const simd<K, T>& a, const          T& b) { return a.v == b  ; }
template<size_t K, typename T> __forceinline simd<K, int> operator !=(const simd<K, T>& a, const          T& b) { return a.v != b  ; }
template<size_t K, typename T> __forceinline simd<K, int> operator < (const simd<K, T>& a, const          T& b) { return a.v <  b  ; }
template<size_t K, typename T> __forceinline simd<K, int> operator >=(const simd<K, T>& a, const          T& b) { return a.v >= b  ; }
template<size_t K, typename T> __forceinline simd<K, int> operator > (const simd<K, T>& a, const          T& b) { return a.v >  b  ; }
template<size_t K, typename T> __forceinline simd<K, int> operator <=(const simd<K, T>& a, const          T& b) { return a.v <= b  ; }

// Unarary operators
template<size_t K, typename T> __forceinline simd<K, T> operator +(const simd<K, T>& a) { return  a.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const simd<K, T>& a) { return -a.v; }

// Arithmetic operators
template<size_t K, typename T> __forceinline simd<K, T> operator +(const simd<K, T>& a, const simd<K, T>& b) { return a.v + b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const simd<K, T>& a, const simd<K, T>& b) { return a.v - b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator *(const simd<K, T>& a, const simd<K, T>& b) { return a.v * b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator /(const simd<K, T>& a, const simd<K, T>& b) { return a.v / b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator +(const simd<1, T>& a, const simd<K, T>& b) { return a.v[0] + b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const simd<1, T>& a, const simd<K, T>& b) { return a.v[0] - b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator *(const simd<1, T>& a, const simd<K, T>& b) { return a.v[0] * b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator /(const simd<1, T>& a, const simd<K, T>& b) { return a.v[0] / b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator +(const simd<K, T>& a, const simd<1, T>& b) { return a.v + b.v[0]; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const simd<K, T>& a, const simd<1, T>& b) { return a.v - b.v[0]; }
template<size_t K, typename T> __forceinline simd<K, T> operator *(const simd<K, T>& a, const simd<1, T>& b) { return a.v * b.v[0]; }
template<size_t K, typename T> __forceinline simd<K, T> operator /(const simd<K, T>& a, const simd<1, T>& b) { return a.v / b.v[0]; }
template<size_t K, typename T> __forceinline simd<K, T> operator +(const          T& a, const simd<K, T>& b) { return a + b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const          T& a, const simd<K, T>& b) { return a - b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator *(const          T& a, const simd<K, T>& b) { return a * b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator /(const          T& a, const simd<K, T>& b) { return a / b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator +(const simd<K, T>& a, const          T& b) { return a.v + b; }
template<size_t K, typename T> __forceinline simd<K, T> operator -(const simd<K, T>& a, const          T& b) { return a.v - b; }
template<size_t K, typename T> __forceinline simd<K, T> operator *(const simd<K, T>& a, const          T& b) { return a.v * b; }
template<size_t K, typename T> __forceinline simd<K, T> operator /(const simd<K, T>& a, const          T& b) { return a.v / b; }

template<size_t K, typename T> __forceinline simd<K, T> operator ^(const simd<K, T>& a, const simd<K, T>& b) { return a.v ^ b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator |(const simd<K, T>& a, const simd<K, T>& b) { return a.v | b.v; }
template<size_t K, typename T> __forceinline simd<K, T> operator &(const simd<K, T>& a, const simd<K, T>& b) { return a.v & b.v; }

template<size_t K, typename T> __forceinline simd<K, T> operator !(const simd<K, T>& a) { return !a.v; }

// Min and max operations
template<size_t K, typename T> __forceinline simd<K, T> min(const simd<K, T> &a, const simd<K, T> &b) {
	simd<K, T> r;
	for (unsigned i = 0; i < K; i++) r[i] = std::min(a[i], b[i]);
	return r;
}

template<size_t K, typename T> __forceinline simd<K, T> max(const simd<K, T> &a, const simd<K, T> &b) {
	simd<K, T> r;
	for (unsigned i = 0; i < K; i++) r[i] = a[i] > b[i] ? a[i] : b[i];
	return r;
}

template<size_t K, typename T, size_t N> __forceinline simd<K, T> hadd_n(const simd<K, T> &a) {
	simd<K, int> mask;
	for (unsigned i = 0; i < K; i++) mask[i] = i^N;
	return a + __builtin_shuffle(a.v, mask.v);
}

template<size_t K, typename T> __forceinline simd<K, T> hadd(const simd<K, T> &a) {
	simd<K, T> r = a;
	if constexpr (K >=  2) r = hadd_n<K, T, 1>(r);
	if constexpr (K >=  4) r = hadd_n<K, T, 2>(r);
	if constexpr (K >=  8) r = hadd_n<K, T, 4>(r);
	if constexpr (K >= 16) r = hadd_n<K, T, 8>(r);
	return r;
}

// Unary operations
template<size_t K, typename T> __forceinline simd<K, T> rcp (const simd<K, T> &a) { return T(1) / a; }
template<size_t K, typename T> __forceinline simd<K, T> sqrt(const simd<K, T> &a) {
	simd<K, T> r;
	for (unsigned i = 0; i < K; i++) r[i] = sqrt(a.v[i]);
	return r;
}
template<size_t K, typename T> __forceinline simd<K, T> abs(const simd<K, T> &a) {
	simd<K, T> r;
	for (unsigned i = 0; i < K; i++) r[i] = std::abs(a.v[i]);
	return r;
}
template<size_t K, typename T> __forceinline bool any(const simd<K, T> &a) {
	simd<K, T> zero = simd<K, T>::broadcast(0);
	return memcmp(&a.v, &zero.v, sizeof(a)) != 0;
}

// Can't get the compiler to generate these without builtins so use constexpr
template<size_t K, typename T> __forceinline unsigned movemask(const simd<K, T> &a) {
	     if constexpr (sizeof(T) == 1 && K ==   8) return __builtin_ia32_pmovmskb(simd<K, char>(a));
	else if constexpr (sizeof(T) == 1 && K ==  16) return __builtin_ia32_pmovmskb128(simd<K, char>(a).v);
	else if constexpr (sizeof(T) == 1 && K ==  32) return __builtin_ia32_pmovmskb256(simd<K, char>(a).v);
	else if constexpr (sizeof(T) == 4 && K ==   4) return __builtin_ia32_movmskps(simd<K, float>(a).v);
	else if constexpr (sizeof(T) == 4 && K ==   8) return __builtin_ia32_movmskps256(simd<K, float>(a).v);
	else if constexpr (sizeof(T) == 8 && K ==   2) return __builtin_ia32_movmskpd(simd<K, double>(a).v);
	else if constexpr (sizeof(T) == 8 && K ==   4) return __builtin_ia32_movmskpd256(simd<K, double>(a).v);
	else {
		unsigned result = 0;
		for (unsigned i = 0; i < K; i++) result |= ((a[i] < 0) << i);
		return result;
	}
}

// Ternary operations
template<size_t K, typename T> __forceinline simd<K, T> fmadd(const simd<K, T> &a, const simd<K, T> &b, const simd<K, T> &c) { return a * b + c; }

// Stream operator
template<size_t K, typename T> inline std::ostream& operator<<(std::ostream& cout, const simd<K, T>& a) {
	std::cout << "[" << a.v[0];
	for (unsigned i = 1; i < K; i++) std::cout << ", " << a.v[i]; 
	return std::cout << "]";
}


template <size_t K>
using simdf = simd<K, float>;

template <size_t K>
using simdi = simd<K, int>;

template <size_t K>
using simdu = simd<K, unsigned>;

using simd8f = simd<8, float>;
using simd4f = simd<4, float>;
using simd1f = simd<1, float>;
using simd8i = simd<8, int>;
using simd4i = simd<4, int>;
using simd1i = simd<1, int>;

};
