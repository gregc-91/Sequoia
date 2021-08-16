#pragma once

#include "simd_generic.h"
#include "vec3.h"

#include <cfloat>
#include <immintrin.h>

using namespace vec;
using namespace smd;

enum AABB_FLAGS {
	AABB_FLAGS_NONE = 0x00,
	AABB_FLAGS_LEAF = 0x01
};

struct Triangle {
	vec3f1 v0;
	vec3f1 v1;
	vec3f1 v2;
	
	Triangle() {};
	Triangle(const vec3f1 &v0, const vec3f1 &v1, const vec3f1 &v2) : v0(v0), v1(v1), v2(v2) {}
};

template <size_t K>
struct Node {
	vec3f<K> min;
	simdi<K> flags;
	vec3f<K> max;
	simdi<K> child;
};

struct Node1 {
	vec3f1 min;
	uint16_t count;
	uint16_t flags;
	vec3f1 max;
	uint32_t child;
	
	Node1() {}
};

// 32B node that holds two quantised AABBs
// Child pointers are limited to 24bit giving 16M nodes
// Suggestion to extend would be use instancing so the pointers are relative to an instance
// Suggestion to extend type would be to store an maintain the global AS type to infer leaf type
struct Node2 {
	vec3f1 centre;

	uint32_t a_child : 24;
	uint32_t size    : 8;
	uint32_t b_child : 24;
	uint32_t a_count : 3;
	uint32_t a_leaf  : 1;
	uint32_t b_count : 3;
	uint32_t b_leaf  : 1;
	
	uint8_t a_min_x;
	uint8_t a_max_x;
	uint8_t b_min_x;
	uint8_t b_max_x;
	
	uint8_t a_min_y;
	uint8_t a_max_y;
	uint8_t b_min_y;
	uint8_t b_max_y;
	
	uint8_t a_min_z;
	uint8_t a_max_z;
	uint8_t b_min_z;
	uint8_t b_max_z;
	
};

template <size_t K>
struct AABB {
	vec3f<K> min;
	vec3f<K> max;
	
	AABB() {}
	AABB(float min, float max) : min(simdf<K>::broadcast(min)), max(simdf<K>::broadcast(max)) {}
	AABB(simdf<K> min, simdf<K> max) : min(min), max(max) {}
	AABB(vec3f<K> min, vec3f<K> max) : min(min), max(max) {}
	AABB(simdf<K> &x, simdf<K> &y, simdf<K> &z, simdf<K> &X, simdf<K> &Y, simdf<K> &Z) : min(x, y, z), max(X, Y, Z) {}
	
	inline void grow(const vec3f<K> &p) { min = vec::min(min, p);     max = vec::max(max, p); }
	inline void grow(const AABB<K>  &a) { min = vec::min(min, a.min); max = vec::max(max, a.max); }
	
	inline AABB<K> combine(const vec3f<K> &p) { return AABB<K>(vec::min(min, p)    , vec::max(max, p)    ); }
	inline AABB<K> combine(const AABB<K>  &a) { return AABB<K>(vec::min(min, a.min), vec::max(max, a.max)); }
	
	inline simdf<K> sa() {
		vec3f<K> len = max - min;
		return (len.x*len.y + len.x*len.z + len.y*len.z) * 2.0f;
	}
	
	inline void reset() { min = simdf<K>::broadcast(FLT_MAX); max = simdf<K>::broadcast(-FLT_MAX); }
};

struct AABB1 {
	vec3f1 min;
	vec3f1 max;
	
	AABB1() {}
	AABB1(float min, float max) : min(min), max(max) {}
	AABB1(vec3f1 min, vec3f1 max) : min(min), max(max) {}
	AABB1(float &x, float &y, float &z, float &X, float &Y, float &Z) : min(x, y, z), max(X, Y, Z) {}
	AABB1(const Triangle &t) {
		min = vec::min(vec::min(t.v0, t.v1), t.v2);
		max = vec::max(vec::max(t.v0, t.v1), t.v2);
	}
	
	__forceinline void grow(const vec3f1 &p) { min = vec::min(min, p);     max = vec::max(max, p); }
	__forceinline void grow(const AABB1  &a) { min = vec::min(min, a.min); max = vec::max(max, a.max); }
	
	__forceinline AABB1 combine(const vec3f1 &p) const { return AABB1(vec::min(min, p)    , vec::max(max, p)    ); }
	__forceinline AABB1 combine(const AABB1  &a) const { return AABB1(vec::min(min, a.min), vec::max(max, a.max)); }
	
	__forceinline float sa() {
		vec3f1 len = max - min;
		return (len.x*len.y + len.x*len.z + len.y*len.z) * 2.0f;
	}
	
	__forceinline void reset() { min = vec3f1(FLT_MAX); max = vec3f1(-FLT_MAX); }
};

struct alignas(32) AABBh {
	simdf<4> min;
	simdf<4> max;
	
	AABBh() {}
	AABBh(float min, float max) : min(simdf<4>::broadcast(min)), max(simdf<4>::broadcast(max)) {}
	AABBh(simdf<4> min, simdf<4> max) : min(min), max(max) {}
	
	__forceinline void grow(const simdf<4> &p) { min.v = _mm_min_ps((__m128)min.v, (__m128)p.v); max.v = _mm_max_ps((__m128)max.v, (__m128)p.v); }
	__forceinline void grow(const AABBh  &a) { min.v = _mm_min_ps((__m128)min.v, (__m128)a.min.v); max.v = _mm_max_ps((__m128)max.v, (__m128)a.max.v); }
	
	__forceinline AABBh combine(const simdf<4> &p) { return AABBh(_mm_min_ps((__m128)min.v, (__m128)p.v)    , _mm_max_ps((__m128)max.v, (__m128)p.v)    ); }
	__forceinline AABBh combine(const AABBh  &a) { return AABBh(_mm_min_ps((__m128)min.v, (__m128)a.min.v), _mm_max_ps((__m128)max.v, (__m128)a.max.v)); }
	
	__forceinline float sa() {
		simdf<4> len = max - min;
		return (len[0]*len[1] + len[0]*len[2] + len[1]*len[2]);
	}
	
	__forceinline void reset() { min = simdf<4>(simdf<4>::broadcast(FLT_MAX)); max = simdf<4>(simdf<4>::broadcast(-FLT_MAX)); }
};

struct alignas(32) AABBm {
	__m128 min;
	__m128 max;
	
	AABBm() {}
	AABBm(float min, float max) : min(_mm_set_ps1(min)), max(_mm_set_ps1(max)) {}
	AABBm(__m128 min, __m128 max) : min(min), max(max) {}
	
	__forceinline void grow(const __m128 &p) { min = _mm_min_ps(min, p);     max = _mm_max_ps(max, p    ); }
	__forceinline void grow(const AABBm  &a) { min = _mm_min_ps(min, a.min); max = _mm_max_ps(max, a.max); }
	
	__forceinline AABBm combine(const __m128 &p) { return AABBm(_mm_min_ps(min, p)    , _mm_max_ps(max, p)    ); }
	__forceinline AABBm combine(const AABBm  &a) { return AABBm(_mm_min_ps(min, a.min), _mm_max_ps(max, a.max)); }
	
	__forceinline float sa() {
		__m128 len = _mm_sub_ps(max, min);
		float* lenf = (float*)&len;
		return (lenf[0]*lenf[1] + lenf[0]*lenf[2] + lenf[1]*lenf[2]);
	}
	
	__forceinline void reset() { min = _mm_set_ps1(FLT_MAX); max = _mm_set_ps1(-FLT_MAX); }
};

__forceinline int operator ==(const AABB1& a, const AABB1& b) { return a.min == b.min && a.max == b.max; }

struct Triangle64B {
	simdf<4> v0;
	simdf<4> v1;
	simdf<4> v2;
	simdf<4> pad;
};

struct PluckerTriangle {
	float  nu;   //used to store normal data 
	float  nv;   //used to store normal data 
	float  np;   //used to store vertex data 
	float  pu;   //used to store vertex data 
	float  pv;   //used to store vertex data 
	int    ci;   //used to store edges data 
	float  e0u;  //used to store edges data 
	float  e0v;  //used to store edges data 
	float  e1u;  //used to store edges data 
	float  e1v;  //used to store edges data 
	int    pad0; //padding 
	int    pad1; //padding 
};

struct Hierarchy {
	Node1*    node_base;
	unsigned  node_count;
	Triangle* tri_base;
	PluckerTriangle* plucker_tri_base;
	unsigned  tri_count;
	
	unsigned  root_index;
};

/*
inline PluckerTriangle TriangleToPluckerTriangle(Triangle &tri)
{
	PluckerTriangle t;
	
	// Compute the normal
	vec3f<1> e1 = tri.v1 - tri.v0;
	vec3f<1> e2 = tri.v2 - tri.v0;
	vec3f<1> n  = e1.cross(e2);
	
	// Compute the indices
	int w = argmax(abs(n));
	int u = w == 0 ? 1 : 0;
	int v = w == 2 ? 1 : 2;
	
	// Store the two normalised components
	t.nu = (n[u] / abs(n[w]));
	t.nv = (n[v] / abs(n[w]));
	
	// Store the vertex p
	t.pu = tri.v0[u];
	t.pv = tri.v0[v];
	t.np = (t.nu*tri.v0[u] + t.nv*tri.v0[v] + tri.v0[w]);
	
	// Store two normalised edge components
	t.e0u = ((w==1 ? -e1[u] : e1[u]) / n[w]); 
	t.e0v = ((w==1 ? -e1[v] : e1[v]) / n[w]); 
	t.e1u = ((w==1 ? -e2[u] : e2[u]) / n[w]); 
	t.e1v = ((w==1 ? -e2[v] : e2[v]) / n[w]);
	
	// Store the maximum index
	t.ci = w;
	
	return t;
}*/