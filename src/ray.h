#pragma once

#include "simd_generic.h"

using namespace smd;
using namespace vec;

template <size_t K>
struct Ray {
	
	vec3f<K> origin;
	simdf<K> tmin;
	vec3f<K> direction;
	simdf<K> tmax;
	
	// Constructors and assignment operators
	//__forceinline Ray<K>() {}
    //__forceinline Ray<K>(const Ray<K>& a) : origin(a.origin), tmin(a.tmin), direction(a.direction), tmax(a.tmax) {}
    //__forceinline Ray<K>(vec3f<K> origin, simdf<K> tmin, vec3f<K> direction, simdf<K> tmax) : origin(origin), tmin(tmin), direction(direction), tmax(tmax) {}
};

template <size_t K>
struct RayBundle {
	vec3f<1> dir_min;
	vec3f<1> dir_max;
	vec3f<1> pos_min;
	vec3f<1> pos_max;
	unsigned ids[K];
};

template <size_t K>
struct HitAttributes {
	simdi<K> hit;
	
	HitAttributes() : hit(simdi<K>::broadcast(0)) {}
};
