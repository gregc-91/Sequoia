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
