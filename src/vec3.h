#pragma once

#include "simd_generic.h"

#include <ostream>
#include <iostream>

using namespace smd;

namespace vec {

template <size_t K, typename T>
struct vec3 {
	enum { N  = 3 };
	
	union {
		struct {
			simd<K, T> x, y, z;
		};
		simd<K, T> cmpnt[N];
	};
	
	__forceinline vec3() {}
	__forceinline vec3(const simd<K, T> &a) : x(a), y(a), z(a) {}
	__forceinline vec3(const simd<K, T> &x, const simd<K, T> &y, const simd<K, T> &z) : x(x), y(y), z(z) {}
	__forceinline vec3(const         T  &x, const         T  &y, const         T  &z) : x(simd<K, T>::broadcast(x)), y(simd<K, T>::broadcast(y)), z(simd<K, T>::broadcast(z)) {}
	__forceinline vec3(const vec3<K, T> &a) : x(a.x), y(a.y), z(a.z) {}
	__forceinline vec3<K, T>& operator =(const vec3<K, T>& other) { x = other.x; y = other.y; z = other.z; return *this; }
	
	__forceinline const simd<K, T> & operator [](size_t index) const { /*assert(index < N);*/ return cmpnt[index]; }
	__forceinline       simd<K, T> & operator [](size_t index)       { /*assert(index < N);*/ return cmpnt[index]; }
	
	// Element-wise arithmetic operators
	__forceinline vec3<K, T> operator +(const vec3<K, T> &other) { return vec3(x + other.x, y + other.y, z + other.z); }
	__forceinline vec3<K, T> operator -(const vec3<K, T> &other) { return vec3(x - other.x, y - other.y, z - other.z); }
	__forceinline vec3<K, T> operator *(const vec3<K, T> &other) { return vec3(x * other.x, y * other.y, z * other.z); }
	__forceinline vec3<K, T> operator /(const vec3<K, T> &other) { return vec3(x / other.x, y / other.y, z / other.z); }
	
	__forceinline const vec3<K, T>& operator +=(const vec3<K, T> &other) { x += other.x; y += other.y; z += other.z; return *this;}
	__forceinline const vec3<K, T>& operator -=(const vec3<K, T> &other) { x -= other.x; y -= other.y; z -= other.z; return *this;}
	__forceinline const vec3<K, T>& operator *=(const vec3<K, T> &other) { x *= other.x; y *= other.y; z *= other.z; return *this;}
	__forceinline const vec3<K, T>& operator /=(const vec3<K, T> &other) { x /= other.x; y /= other.y; z /= other.z; return *this;}
	
	static __forceinline vec3<K, T> broadcast(const vec3<1, T> &a) { return vec3<K, T>(simd<K, T>::broadcast(a.x), simd<K, T>::broadcast(a.y), simd<K, T>::broadcast(a.z)); };
	
	__forceinline const vec3<K, T> sqrt() { return vec3<K, T>(x.sqrt(), y.sqrt(), z.sqrt()); }
	__forceinline const simd<K, T> length() { return this->dot(*this).sqrt(); }
	
	__forceinline const simd<K, T> dot(const vec3<K, T> &other) { return x * other.x + y * other.y + z * other.z; }
	__forceinline const vec3<K, T> cross(const vec3<K, T> &other) { return vec3<K, T>(y*other.z - other.y*z, z*other.x - other.z*x, x*other.y - other.x*y); }
};

template<size_t K, typename T>__forceinline vec3<K, int> operator ==(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, int>(a.x == b.x, a.y == b.y, a.z == b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  >(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, int>(a.x  > b.x, a.y  > b.y, a.z  > b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator >=(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, int>(a.x >= b.x, a.y >= b.y, a.z >= b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  <(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, int>(a.x  < b.x, a.y  < b.y, a.z  < b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator <=(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, int>(a.x <= b.x, a.y <= b.y, a.z <= b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator ==(const          T& a, const vec3<K, T>& b) { return vec3<K, int>(a   == b.x, a   == b.y, a   == b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  >(const          T& a, const vec3<K, T>& b) { return vec3<K, int>(a    > b.x, a    > b.y, a    > b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator >=(const          T& a, const vec3<K, T>& b) { return vec3<K, int>(a   >= b.x, a   >= b.y, a   >= b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  <(const          T& a, const vec3<K, T>& b) { return vec3<K, int>(a    < b.x, a    < b.y, a    < b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator <=(const          T& a, const vec3<K, T>& b) { return vec3<K, int>(a   <= b.x, a   <= b.y, a   <= b.z); }
template<size_t K, typename T>__forceinline vec3<K, int> operator ==(const vec3<K, T>& a, const          T& b) { return vec3<K, int>(a.x == b  , a.y == b  , a.z == b  ); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  >(const vec3<K, T>& a, const          T& b) { return vec3<K, int>(a.x  > b  , a.y  > b  , a.z  > b  ); }
template<size_t K, typename T>__forceinline vec3<K, int> operator >=(const vec3<K, T>& a, const          T& b) { return vec3<K, int>(a.x >= b  , a.y >= b  , a.z >= b  ); }
template<size_t K, typename T>__forceinline vec3<K, int> operator  <(const vec3<K, T>& a, const          T& b) { return vec3<K, int>(a.x  < b  , a.y  < b  , a.z  < b  ); }
template<size_t K, typename T>__forceinline vec3<K, int> operator <=(const vec3<K, T>& a, const          T& b) { return vec3<K, int>(a.x <= b  , a.y <= b  , a.z <= b  ); }

template<size_t K, typename T> __forceinline vec3<K, T> operator +(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x + b.x, a.y + b.y, a.z + b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator -(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x - b.x, a.y - b.y, a.z - b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator *(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x * b.x, a.y * b.y, a.z * b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator /(const vec3<K, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x / b.x, a.y / b.y, a.z / b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator +(const vec3<1, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x + b.x, a.y + b.y, a.z + b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator -(const vec3<1, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x - b.x, a.y - b.y, a.z - b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator *(const vec3<1, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x * b.x, a.y * b.y, a.z * b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator /(const vec3<1, T>& a, const vec3<K, T>& b) { return vec3<K, T>(a.x / b.x, a.y / b.y, a.z / b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator +(const          T& a, const vec3<K, T>& b) { return vec3<K, T>(a   + b.x, a   + b.y, a   + b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator -(const          T& a, const vec3<K, T>& b) { return vec3<K, T>(a   - b.x, a   - b.y, a   - b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator *(const          T& a, const vec3<K, T>& b) { return vec3<K, T>(a   * b.x, a   * b.y, a   * b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator /(const          T& a, const vec3<K, T>& b) { return vec3<K, T>(a   / b.x, a   / b.y, a   / b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator +(const vec3<K, T>& a, const vec3<1, T>& b) { return vec3<K, T>(a.x + b.x, a.y + b.y, a.z + b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator -(const vec3<K, T>& a, const vec3<1, T>& b) { return vec3<K, T>(a.x - b.x, a.y - b.y, a.z - b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator *(const vec3<K, T>& a, const vec3<1, T>& b) { return vec3<K, T>(a.x * b.x, a.y * b.y, a.z * b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator /(const vec3<K, T>& a, const vec3<1, T>& b) { return vec3<K, T>(a.x / b.x, a.y / b.y, a.z / b.z); }
template<size_t K, typename T> __forceinline vec3<K, T> operator +(const vec3<K, T>& a, const          T& b) { return vec3<K, T>(a.x + b  , a.y + b  , a.z + b  ); }
template<size_t K, typename T> __forceinline vec3<K, T> operator -(const vec3<K, T>& a, const          T& b) { return vec3<K, T>(a.x - b  , a.y - b  , a.z - b  ); }
template<size_t K, typename T> __forceinline vec3<K, T> operator *(const vec3<K, T>& a, const          T& b) { return vec3<K, T>(a.x * b  , a.y * b  , a.z * b  ); }
template<size_t K, typename T> __forceinline vec3<K, T> operator /(const vec3<K, T>& a, const          T& b) { return vec3<K, T>(a.x / b  , a.y / b  , a.z / b  ); }

template<size_t K, typename T> __forceinline vec3<K, T> fmadd(const vec3<K, T> &a, const vec3<K, T> &b, const vec3<K, T> &c) { return a * b + c; }

template<size_t K, typename T> __forceinline vec3<K, T> min(const vec3<K, T> &a, const vec3<K, T> &b) { return vec3<K, T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
template<size_t K, typename T> __forceinline vec3<K, T> max(const vec3<K, T> &a, const vec3<K, T> &b) { return vec3<K, T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }

template<size_t K,  typename T> __forceinline simd<K, T> hmin( const vec3<K, T>& a ) { return min(min(a.x, a.y), a.z); }
template<size_t K,  typename T> __forceinline simd<K, T> hmax( const vec3<K, T>& a ) { return max(max(a.x, a.y), a.z); }

template<size_t K,  typename T> __forceinline simd<K, int> argmax( const vec3<K, T>& a ) {
	
	simd<K, int> r0 = simd<K, int>::broadcast(0);
	simd<K, int> r1 = simd<K, int>::broadcast(1);
	simd<K, int> r2 = simd<K, int>::broadcast(2);
	
	simd<K, int> r = 
		a.y > a.x ? a.z > a.y ? r2 : r1 : a.z > a.x ? r2 : r0;
		
	return r;
}

template<size_t K, typename T> __forceinline vec3<K, T> abs(const vec3<K, T> &a) { return vec3<K, T>(abs(a.x), abs(a.y), abs(a.z)); }
template<size_t K, typename T> __forceinline vec3<K, T> rcp(const vec3<K, T> &a) { return vec3<K, T>(rcp(a.x), rcp(a.y), rcp(a.z)); }

template<size_t K, typename T> __forceinline vec3<K, T> ternary(const vec3<K, int> &c, const vec3<K, T> &a, const vec3<K, T> &b) {
	return vec3<K, T>(c.x ? a.x : b.x, c.y ? a.y : b.y, c.z ? a.z : b.z);
}

template<size_t K, typename T> __forceinline std::ostream& operator<<(std::ostream& cout, const vec3<K, T>& a) {
	return std::cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
}

template <size_t K = 1>
using vec3f = vec3<K, float>;

template <size_t K = 1>
using vec3i = vec3<K, int>;

template <size_t K = 1>
using vec3u = vec3<K, unsigned>;



struct vec3f1 {
	enum { N  = 3 };
	
	union {
		struct {
			float x, y, z;
		};
		float cmpnt[N];
	};
	
	__forceinline vec3f1() {}
	__forceinline vec3f1(const float &a) : x(a), y(a), z(a) {}
	__forceinline vec3f1(const float &x, const float &y, const float &z) : x(x), y(y), z(z) {}
	__forceinline vec3f1(const vec3f1 &a) : x(a.x), y(a.y), z(a.z) {}
	__forceinline vec3f1& operator =(const vec3f1& other) { x = other.x; y = other.y; z = other.z; return *this; }
	
	__forceinline const float & operator [](size_t index) const { /*assert(index < N);*/ return cmpnt[index]; }
	
	// Element-wise arithmetic operators
	__forceinline const vec3f1& operator +=(const vec3f1 &other) { x += other.x; y += other.y; z += other.z; return *this;}
	__forceinline const vec3f1& operator -=(const vec3f1 &other) { x -= other.x; y -= other.y; z -= other.z; return *this;}
	__forceinline const vec3f1& operator *=(const vec3f1 &other) { x *= other.x; y *= other.y; z *= other.z; return *this;}
	__forceinline const vec3f1& operator /=(const vec3f1 &other) { x /= other.x; y /= other.y; z /= other.z; return *this;}
	
	__forceinline const vec3f1  sqrt() { return vec3f1(sqrtf(x), sqrtf(x), sqrtf(x)); }
	__forceinline const float length() { return sqrtf(this->dot(*this)); }
	
	__forceinline const float  dot  (const vec3f1 &other) { return x * other.x + y * other.y + z * other.z; }
	__forceinline const vec3f1 cross(const vec3f1 &other) { return vec3f1(y*other.z - other.y*z, z*other.x - other.z*x, x*other.y - other.x*y); }
};

 __forceinline int operator ==(const vec3f1& a, const vec3f1& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

__forceinline vec3f1 operator +(const vec3f1& a, const vec3f1& b) { return vec3f1(a.x + b.x, a.y + b.y, a.z + b.z); }
__forceinline vec3f1 operator -(const vec3f1& a, const vec3f1& b) { return vec3f1(a.x - b.x, a.y - b.y, a.z - b.z); }
__forceinline vec3f1 operator *(const vec3f1& a, const vec3f1& b) { return vec3f1(a.x * b.x, a.y * b.y, a.z * b.z); }
__forceinline vec3f1 operator /(const vec3f1& a, const vec3f1& b) { return vec3f1(a.x / b.x, a.y / b.y, a.z / b.z); }
__forceinline vec3f1 operator +(const float & a, const vec3f1& b) { return vec3f1(a   + b.x, a   + b.y, a   + b.z); }
__forceinline vec3f1 operator -(const float & a, const vec3f1& b) { return vec3f1(a   - b.x, a   - b.y, a   - b.z); }
__forceinline vec3f1 operator *(const float & a, const vec3f1& b) { return vec3f1(a   * b.x, a   * b.y, a   * b.z); }
__forceinline vec3f1 operator /(const float & a, const vec3f1& b) { return vec3f1(a   / b.x, a   / b.y, a   / b.z); }
__forceinline vec3f1 operator +(const vec3f1& a, const float & b) { return vec3f1(a.x + b  , a.y + b  , a.z + b  ); }
__forceinline vec3f1 operator -(const vec3f1& a, const float & b) { return vec3f1(a.x - b  , a.y - b  , a.z - b  ); }
__forceinline vec3f1 operator *(const vec3f1& a, const float & b) { return vec3f1(a.x * b  , a.y * b  , a.z * b  ); }
__forceinline vec3f1 operator /(const vec3f1& a, const float & b) { return vec3f1(a.x / b  , a.y / b  , a.z / b  ); }

__forceinline vec3f1 fmadd(const vec3f1 &a, const vec3f1 &b, const vec3f1 &c) { return a * b + c; }

__forceinline vec3f1 min(const vec3f1 &a, const vec3f1 &b) { return vec3f1(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
__forceinline vec3f1 max(const vec3f1 &a, const vec3f1 &b) { return vec3f1(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }

__forceinline float hmin( const vec3f1& a ) { return std::min(std::min(a.x, a.y), a.z); }
__forceinline float hmax( const vec3f1& a ) { return std::max(std::max(a.x, a.y), a.z); }

__forceinline int argmax( const vec3f1& a ) {
	return a.y > a.x ? a.z > a.y ? 2 : 1 : a.z > a.x ? 2 : 0;
}

__forceinline vec3f1 abs(const vec3f1 &a) { return vec3f1(std::abs(a.x), std::abs(a.y), std::abs(a.z)); }
__forceinline vec3f1 rcp(const vec3f1 &a) { return vec3f1(1.0f/a.x, 1.0f/a.y, 1.0f/a.z); }

__forceinline std::ostream& operator<<(std::ostream& cout, const vec3f1& a) {
	return std::cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
}



};
