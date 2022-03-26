#pragma once

#include "aabb.h"

#include <vector>

typedef struct Bin {
	AABB1 p_aabb;
	int   count;
	int   pad;
	AABB1 c_aabb;
	int   pad1;
	int   pad2;
	
	Bin() {};
	
	Bin(float pmin, float pmax, float cmin, float cmax, int count) :
		p_aabb(pmin, pmax), count(count), c_aabb(cmin, cmax) {};
		
	Bin(const AABB1 &p_aabb, const AABB1 &c_aabb, const int count) :
		p_aabb(p_aabb), count(count), c_aabb(c_aabb) {};
	
	inline void reset() { p_aabb.reset(); c_aabb.reset(); count = 0; }
	
	inline Bin combine(const Bin &other) { return Bin(p_aabb.combine(other.p_aabb), c_aabb.combine(other.c_aabb), count + other.count); }
} Bin;

typedef struct alignas(64) Binh {
	AABBh p_aabb;
	AABBh c_aabb;
	int   count;
	int   pad0;
	int   pad1;
	int   pad2;
	int   pad3;
	int   pad4;
	int   pad5;
	int   pad6;
	
	Binh() {};
	
	Binh(float pmin, float pmax, float cmin, float cmax, int count) :
		p_aabb(pmin, pmax), c_aabb(cmin, cmax), count(count) {};
		
	Binh(const AABBh &p_aabb, const AABBh &c_aabb, const int count) :
		p_aabb(p_aabb), c_aabb(c_aabb), count(count) {};
	
	inline void reset() { p_aabb.reset(); c_aabb.reset(); count = 0; }
	
	inline Binh combine(const Binh &other) { return Binh(p_aabb.combine(other.p_aabb), c_aabb.combine(other.c_aabb), count + other.count); }
} Binh;

typedef struct Binm {
	AABBm p_aabb;
	AABBm c_aabb;
	int   count;
	int   pad0;
	int   pad1;
	int   pad2;
	int   pad3;
	int   pad4;
	int   pad5;
	int   pad6;
	
	inline void reset() { p_aabb.reset(); c_aabb.reset(); count = 0; }
	
	inline Binm combine(const Binm &other) { return { p_aabb.combine(other.p_aabb), c_aabb.combine(other.c_aabb), count + other.count, 0, 0, 0, 0, 0, 0, 0 }; }
} Binm;

template <size_t K>
struct Bink {
	AABB<K>  p_aabb;
	AABB<K>  c_aabb;
	simdi<K> count;
	
	inline void reset() { p_aabb.reset(); c_aabb.reset(); count = 0; }
	
	inline Bink<K> combine(const Bink<K> &other) { return { p_aabb.combine(other.p_aabb), c_aabb.combine(other.c_aabb), count + other.count }; }
};

class Builder
{
public:
	// A topdown recursive builder that uses openmp tasks to parallelise each split task
	static Hierarchy build_hierarchy(std::vector<Triangle> &primitives);

	// A builder based on partitioning a 32bit morton curve
	static Hierarchy build_hierarchy_morton(std::vector<Triangle> &primitives);

	// A builder based on partitioning a 64bit morton curve
	static Hierarchy build_hierarchy_morton2(std::vector<Triangle> &primitives);

	// A hybrid builder binning triangles to a grid and launching a build for each cell
	static Hierarchy build_hierarchy_grid(std::vector<Triangle> &primitives);

	// A version of grid that uses SIMD across x,y,z
	static Hierarchy build_hierarchy_grid_sse(std::vector<Triangle> &primitives);

	// A version of grid that uses SIMD across 8 input primitives
	static Hierarchy build_hierarchy_grid_m128(std::vector<Triangle> &primitives);

	//  A hybrid builder parallelising across primitives near the root and switching to recursive
	static Hierarchy build_hierarchy_horizontal(std::vector<Triangle> &primitives);

};