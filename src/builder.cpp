#include "builder.h"

#include "algorithms.h"

#include <algorithm>
#include <cfloat>
#include <omp.h>

#include <immintrin.h>

#define NUM_BINS 8
#define NUM_THREADS 64
#define GRID_DIM 4
#define NUM_CELLS GRID_DIM*GRID_DIM*GRID_DIM
#define MORTON_BITS 20

typedef int   v8si __attribute__ ((vector_size (32)));
typedef float v8sf __attribute__ ((vector_size (32)));

uint8_t demorton[64] = {
	0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x02, 0x03, 0x06, 0x07, 0x12, 0x13, 0x16, 0x17,
	0x08, 0x09, 0x0C, 0x0D, 0x18, 0x19, 0x1C, 0x1D, 0x0A, 0x0B, 0x0E, 0x0F, 0x1A, 0x1B, 0x1E, 0x1F,
	0x20, 0x21, 0x24, 0x25, 0x30, 0x31, 0x34, 0x35, 0x22, 0x23, 0x26, 0x27, 0x32, 0x33, 0x36, 0x37,
	0x28, 0x29, 0x2C, 0x2D, 0x38, 0x39, 0x3C, 0x3D, 0x2A, 0x2B, 0x2E, 0x2F, 0x3A, 0x3B, 0x3E, 0x3F
};

uint32_t Compact1By2(uint32_t x)
{
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

uint32_t count_nodes(Hierarchy &hierarchy, uint32_t index)
{
	Node1 &node = hierarchy.node_base[index];
	
	unsigned count = 1;
	
	if (!(node.flags & AABB_FLAGS_LEAF))
	{
		for (unsigned i = 0; i < node.count; i++)
		{
			count += count_nodes(hierarchy, node.child+i);
		}
	}
	
	return count;
}

void setup(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABB1> &aabbs, std::vector<float> &centres, AABB1 &p_aabb, AABB1 &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size()*4);
	triangleIds.resize(primitives.size());
	
	p_aabb = AABB1(FLT_MAX, -FLT_MAX);
	c_aabb = AABB1(FLT_MAX, -FLT_MAX);
	
	AABB1 p_aabb_thread[NUM_THREADS];
	AABB1 c_aabb_thread[NUM_THREADS];
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb_thread[i] = AABB1(FLT_MAX, -FLT_MAX);
		c_aabb_thread[i] = AABB1(FLT_MAX, -FLT_MAX);
	}
	
	#pragma omp parallel for
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		unsigned tid = omp_get_thread_num();
		
		auto &p = primitives[i];
		auto &a = aabbs[i];

		// Compute the aabb for this primitive
		a.min = min(min(p.v0, p.v1), p.v2);
		a.max = max(max(p.v0, p.v1), p.v2);
		
		// Compute the centroid for this primitive
		vec3f1 c = (a.min + a.max) * 0.5f;
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb_thread[tid].grow(a);
		c_aabb_thread[tid].grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i*4+0] = c.x;
		centres[i*4+1] = c.y;
		centres[i*4+2] = c.z;
	}
	
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb.grow(p_aabb_thread[i]);
		c_aabb.grow(c_aabb_thread[i]);
	}
}

#pragma omp declare reduction(growaabb  : AABB1 : omp_out = omp_out.combine(omp_in)) initializer (omp_priv=AABB1(FLT_MAX,-FLT_MAX))
#pragma omp declare reduction(growaabbh : AABBh : omp_out = omp_out.combine(omp_in)) initializer (omp_priv=AABBh(FLT_MAX,-FLT_MAX))
#pragma omp declare reduction(growaabbm : AABBm : omp_out = omp_out.combine(omp_in)) initializer (omp_priv=AABBm(FLT_MAX,-FLT_MAX))

#pragma omp declare reduction(appvec  : std::vector<unsigned> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
	
#pragma omp declare reduction(combin  : Bin  : omp_out = omp_in.count == 0 ? omp_out : omp_out.combine(omp_in)) initializer ( omp_priv = Bin (FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, 0) )
#pragma omp declare reduction(combinh : Binh : omp_out = omp_in.count == 0 ? omp_out : omp_out.combine(omp_in)) initializer ( omp_priv = Binh(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, 0) )
		
void setup_reduction(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABB1> &aabbs, std::vector<float> &centres, AABB1 &p_aabb, AABB1 &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size()*4);
	triangleIds.resize(primitives.size());
	
	p_aabb = AABB1(FLT_MAX, -FLT_MAX);
	c_aabb = AABB1(FLT_MAX, -FLT_MAX);
	
	#pragma omp parallel for reduction(growaabb : p_aabb, c_aabb) num_threads(8)
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		auto &p = primitives[i];
		auto &a = aabbs[i];

		// Compute the aabb for this primitive
		a.min = min(min(p.v0, p.v1), p.v2);
		a.max = max(max(p.v0, p.v1), p.v2);
		
		// Compute the centroid for this primitive
		vec3f1 c = (a.min + a.max) * 0.5f;
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb.grow(a);
		c_aabb.grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i*4+0] = c.x;
		centres[i*4+1] = c.y;
		centres[i*4+2] = c.z;
	}
}

void setup_sse(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABBh> &aabbs, std::vector<simdf<4>> &centres, AABBh &p_aabb, AABBh &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size());
	triangleIds.resize(primitives.size());
	
	p_aabb = AABBh(FLT_MAX, -FLT_MAX);
	c_aabb = AABBh(FLT_MAX, -FLT_MAX);
	
	AABBh p_aabb_thread[NUM_THREADS];
	AABBh c_aabb_thread[NUM_THREADS];
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb_thread[i] = AABBh(FLT_MAX, -FLT_MAX);
		c_aabb_thread[i] = AABBh(FLT_MAX, -FLT_MAX);
	}
	
	#pragma omp parallel for
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		unsigned tid = omp_get_thread_num();
		
		auto &p = primitives[i];
		auto &a = aabbs[i];
		
		simdf<4> v0s = simdf<4>::loadu(&p.v0.x);
		simdf<4> v1s = simdf<4>::loadu(&p.v1.x);
		simdf<4> v2s = simdf<4>::loadu(&p.v2.x);

		// Compute the aabb for this primitive
		a.min = min(min(v0s, v1s), v2s);
		a.max = max(max(v0s, v1s), v2s);
		
		// Compute the centroid for this primitive
		simdf<4> c = (a.min + a.max) * 0.5f;
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb_thread[tid].grow(a);
		c_aabb_thread[tid].grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i] = c;
	}
	
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb.grow(p_aabb_thread[i]);
		c_aabb.grow(c_aabb_thread[i]);
	}
}


void setup_sse_reduction(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABBh> &aabbs, std::vector<simdf<4>> &centres, AABBh &p_aabb, AABBh &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size());
	triangleIds.resize(primitives.size());
	
	p_aabb = AABBh(FLT_MAX, -FLT_MAX);
	c_aabb = AABBh(FLT_MAX, -FLT_MAX);
	
	#pragma omp parallel for reduction(growaabbh : p_aabb, c_aabb) num_threads(8)
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		auto &p = primitives[i];
		auto &a = aabbs[i];
		
		simdf<4> v0s = simdf<4>::loadu(&p.v0.x);
		simdf<4> v1s = simdf<4>::loadu(&p.v1.x);
		simdf<4> v2s = simdf<4>::loadu(&p.v2.x);

		// Compute the aabb for this primitive
		a.min = min(min(v0s, v1s), v2s);
		a.max = max(max(v0s, v1s), v2s);
		
		// Compute the centroid for this primitive
		simdf<4> c = (a.min + a.max) * 0.5f;
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb.grow(a);
		c_aabb.grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i] = c;
	}
}

void setup_m128(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABBm> &aabbs, std::vector<__m128> &centres, AABBm &p_aabb, AABBm &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size());
	triangleIds.resize(primitives.size());
	
	p_aabb = AABBm(FLT_MAX, -FLT_MAX);
	c_aabb = AABBm(FLT_MAX, -FLT_MAX);
	
	AABBm p_aabb_thread[NUM_THREADS];
	AABBm c_aabb_thread[NUM_THREADS];
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb_thread[i] = AABBm(FLT_MAX, -FLT_MAX);
		c_aabb_thread[i] = AABBm(FLT_MAX, -FLT_MAX);
	}

	__m128 half = _mm_set_ps1(0.5f);
	
	#pragma omp parallel for
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		unsigned tid = omp_get_thread_num();
		
		auto &p = primitives[i];
		auto &a = aabbs[i];
		
		__m128 v0s = _mm_loadu_ps(&p.v0.x);
		__m128 v1s = _mm_loadu_ps(&p.v1.x);
		__m128 v2s = _mm_loadu_ps(&p.v2.x);

		// Compute the aabb for this primitive
		a.min = _mm_min_ps(_mm_min_ps(v0s, v1s), v2s);
		a.max = _mm_max_ps(_mm_max_ps(v0s, v1s), v2s);
		
		// Compute the centroid for this primitive
		__m128 c = _mm_mul_ps(_mm_add_ps(a.min, a.max), half);
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb_thread[tid].grow(a);
		c_aabb_thread[tid].grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i] = c;
	}
	
	for (uint32_t i = 0; i < NUM_THREADS; i++)
	{
		p_aabb.grow(p_aabb_thread[i]);
		c_aabb.grow(c_aabb_thread[i]);
	}
}

void setup_m128_red(std::vector<Triangle> &primitives, std::vector<uint32_t> &triangleIds, std::vector<AABBm> &aabbs, std::vector<__m128> &centres, AABBm &p_aabb, AABBm &c_aabb)
{
	aabbs.resize(primitives.size());
	centres.resize(primitives.size());
	triangleIds.resize(primitives.size());
	
	p_aabb = AABBm(FLT_MAX, -FLT_MAX);
	c_aabb = AABBm(FLT_MAX, -FLT_MAX);
	
	__m128 half = _mm_set_ps1(0.5f);
	
	#pragma omp parallel for reduction(growaabbm : p_aabb, c_aabb) num_threads(8)
	for (uint32_t i = 0; i < primitives.size(); i++)
	{
		auto &p = primitives[i];
		auto &a = aabbs[i];
		
		__m128 v0s = _mm_loadu_ps(&p.v0.x);
		__m128 v1s = _mm_loadu_ps(&p.v1.x);
		__m128 v2s = _mm_loadu_ps(&p.v2.x);

		// Compute the aabb for this primitive
		a.min = _mm_min_ps(_mm_min_ps(v0s, v1s), v2s);
		a.max = _mm_max_ps(_mm_max_ps(v0s, v1s), v2s);
		
		// Compute the centroid for this primitive
		__m128 c = _mm_mul_ps(_mm_add_ps(a.min, a.max), half);
		
		// Update the overall primitive AABB and overall centres AABB
		p_aabb.grow(a);
		c_aabb.grow(c);

		// Initialise the triangle ID array
		triangleIds[i] = i;
		
		centres[i] = c;
	}
}

int choose_axis(AABB1 &aabb)
{
	vec3f1 length = aabb.max - aabb.min;
	
	int result = 0;
	result += 2 * (length.z > length.x && length.z > length.y);
	result += 1 * (length.y > length.x && length.y >= length.z);
	
	assert(result < 3);
	
	return result;
}

int choose_axis_sse(AABBh &aabb)
{
	simdf<4> length = aabb.max - aabb.min;
	
	int result = 0;
	result += 2 * (length[2] > length[0] && length[2] > length[1]);
	result += 1 * (length[1] > length[0] && length[1] >= length[2]);
	
	assert(result < 3);
	
	return result;
}

int choose_axis_m128(AABBm &aabb)
{
	__m128 lengthm = _mm_sub_ps(aabb.max, aabb.min);
	float* length = (float*)&lengthm;
	
	int result = 0;
	result += 2 * (length[2] > length[0] && length[2] > length[1]);
	result += 1 * (length[1] > length[0] && length[1] >= length[2]);
	
	return result;
}

inline void bin_centroids(	const std::vector<AABB1>              &aabbs, 
					const std::vector<float>             &centres, 
					std::vector<uint32_t>::iterator  begin, 
					std::vector<uint32_t>::iterator  end,
					AABB1                           &c_aabb, 
					int                              axis, 
					std::vector<Bin>                &bins)
{
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	for (unsigned i = 0; i < NUM_BINS; i++)
	{
		bins[i].reset();
	}
	
	
	for (auto it = begin; it < end; it++)
	{
		int bin_id = int(k1*(centres[(*it)*4+axis] - c_aabb.min[axis]));
		//assert(bin_id >= 0 && bin_id < NUM_BINS);
		
		bins[bin_id].p_aabb.grow(aabbs[*it]);
		bins[bin_id].c_aabb.grow(vec3f1(centres[(*it)*4], centres[(*it)*4+1], centres[(*it)*4+2]));
		bins[bin_id].count += 1;
	}
}

inline void bin_centroids_reduction(	const std::vector<AABB1>              &aabbs, 
					const std::vector<float>             &centres, 
					std::vector<uint32_t>::iterator  begin, 
					std::vector<uint32_t>::iterator  end,
					AABB1                           &c_aabb, 
					int                              axis, 
					std::vector<Bin>                &bins)
{
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	for (unsigned i = 0; i < NUM_BINS; i++)
	{
		bins[i].reset();
	}
	Bin* pBin = &bins[0];
	
	#pragma omp parallel for reduction(combin : pBin[:NUM_BINS]) num_threads(8)
	for (auto it = begin; it < end; it++)
	{
		int bin_id = int(k1*(centres[(*it)*4+axis] - c_aabb.min[axis]));
		//assert(bin_id >= 0 && bin_id < NUM_BINS);
		
		pBin[bin_id].p_aabb.grow(aabbs[*it]);
		pBin[bin_id].c_aabb.grow(vec3f1(centres[(*it)*4], centres[(*it)*4+1], centres[(*it)*4+2]));
		pBin[bin_id].count += 1;
	}
}

inline void bin_centroids_sse(	const std::vector<AABBh>        &aabbs, 
								const std::vector<simdf<4>>     &centres, 
								std::vector<uint32_t>::iterator  begin, 
								std::vector<uint32_t>::iterator  end,
								AABBh                           &c_aabb, 
								int                              axis, 
								std::vector<Binh>               &bins)
{
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	for (unsigned i = 0; i < NUM_BINS; i++) {
		bins[i].reset();
	}
	
	
	for (auto it = begin; it < end; it++) {
		int bin_id = int(k1*(centres[*it][axis] - c_aabb.min[axis]));
		//assert(bin_id >= 0 && bin_id < NUM_BINS);
		
		bins[bin_id].p_aabb.grow(aabbs[*it]);
		bins[bin_id].c_aabb.grow(centres[*it]);
		bins[bin_id].count += 1;
	}
}

inline void bin_centroids_m128(	const std::vector<AABBm>        &aabbs, 
								const std::vector<__m128>       &centres, 
								std::vector<uint32_t>::iterator  begin, 
								std::vector<uint32_t>::iterator  end,
								AABBm                           &c_aabb, 
								int                              axis, 
								std::vector<Binm>               &bins)
{
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (((float*)&c_aabb.max)[axis] - ((float*)&c_aabb.min)[axis]);
	
	for (unsigned i = 0; i < NUM_BINS; i++) {
		bins[i].reset();
	}
	
	for (auto it = begin; it < end; it++) {
		#if 0
			__m128 centre = _mm_mul_ps(_mm_add_ps(aabbs[*it].min, aabbs[*it].max), _mm_set_ps1(0.5f));
			int bin_id = int(k1*(((float*)&centre)[axis] - ((float*)&c_aabb.min)[axis]));
			
			bins[bin_id].p_aabb.grow(aabbs[*it]);
			bins[bin_id].c_aabb.grow(centre);
			bins[bin_id].count += 1;
		#else
			int bin_id = int(k1*(((float*)&centres[*it])[axis] - ((float*)&c_aabb.min)[axis]));
			
			bins[bin_id].p_aabb.grow(aabbs[*it]);
			bins[bin_id].c_aabb.grow(centres[*it]);
			bins[bin_id].count += 1;
		#endif
	}
}

int choose_plane(std::vector<Bin> &bins, AABB1 &left_p_aabb, AABB1 &right_p_aabb, AABB1 &left_c_aabb, AABB1 &right_c_aabb)
{
	int result = 0;
	Bin l2r[NUM_BINS-1];
	float best_score = FLT_MAX;
	
	// Initialise end bin
	l2r[0] = bins[0];
	
	// Linear pass left to right summing surface area
	for (int i = 1; i < NUM_BINS-1; i++)
	{
		l2r[i]  = l2r[i-1].combine(bins[i]);
	}
	
	// Linear pass right to left summing surface area
	Bin r2l = bins[NUM_BINS-1];
	for (int i = NUM_BINS-2; i >= 0; i--)
	{	
		float score = l2r[i].p_aabb.sa()*l2r[i].count + r2l.p_aabb.sa()*r2l.count;
		
		if (score < best_score)
		{
			best_score   = score;
			result       = i;
			left_p_aabb  = l2r[i].p_aabb;
			right_p_aabb = r2l.p_aabb;
			left_c_aabb  = l2r[i].c_aabb;
			right_c_aabb = r2l.c_aabb;
		}
		
		r2l = r2l.combine(bins[i]);
	}
	
	return result;
}

int choose_plane_sse(std::vector<Binh> &bins, AABBh &left_p_aabb, AABBh &right_p_aabb, AABBh &left_c_aabb, AABBh &right_c_aabb)
{
	int result = 0;
	Binh l2r[NUM_BINS-1];
	float best_score = FLT_MAX;
	
	// Initialise end bin
	l2r[0] = bins[0];
	
	// Linear pass left to right summing surface area
	for (int i = 1; i < NUM_BINS-1; i++)
	{
		l2r[i]  = l2r[i-1].combine(bins[i]);
	}
	
	// Linear pass right to left summing surface area
	Binh r2l = bins[NUM_BINS-1];
	for (int i = NUM_BINS-2; i >= 0; i--)
	{	
		float score = l2r[i].p_aabb.sa()*l2r[i].count + r2l.p_aabb.sa()*r2l.count;
		
		if (score < best_score)
		{
			best_score   = score;
			result       = i;
			left_p_aabb  = l2r[i].p_aabb;
			right_p_aabb = r2l.p_aabb;
			left_c_aabb  = l2r[i].c_aabb;
			right_c_aabb = r2l.c_aabb;
		}
		
		r2l = r2l.combine(bins[i]);
	}
	
	return result;
}

int choose_plane_m128(std::vector<Binm> &bins, AABBm &left_p_aabb, AABBm &right_p_aabb, AABBm &left_c_aabb, AABBm &right_c_aabb)
{
	int result = 0;
	Binm l2r[NUM_BINS-1];
	float best_score = FLT_MAX;
	
	// Initialise end bin
	l2r[0] = bins[0];
	
	// Linear pass left to right summing surface area
	for (int i = 1; i < NUM_BINS-1; i++)
	{
		l2r[i]  = l2r[i-1].combine(bins[i]);
	}
	
	// Linear pass right to left summing surface area
	Binm r2l = bins[NUM_BINS-1];
	for (int i = NUM_BINS-2; i >= 0; i--)
	{	
		float score = l2r[i].p_aabb.sa()*l2r[i].count + r2l.p_aabb.sa()*r2l.count;
		
		if (score < best_score)
		{
			best_score   = score;
			result       = i;
			left_p_aabb  = l2r[i].p_aabb;
			right_p_aabb = r2l.p_aabb;
			left_c_aabb  = l2r[i].c_aabb;
			right_c_aabb = r2l.c_aabb;
		}
		
		r2l = r2l.combine(bins[i]);
	}
	
	return result;
}

void partition(	int 	plane,
				int 	axis,
				AABB1  &c_aabb,
				const std::vector<float> &centres,
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				std::vector<uint32_t>::iterator &mid)
{
	// Use two iterators, one left to right and one right to left.
	// Increment the first until a value to the right of the plane is found
	// Increment the second until a value to the left of the plane is found
	// Swap the values and repeat until they meet
	std::vector<uint32_t>::iterator a = begin;
	std::vector<uint32_t>::iterator b = end-1;
	
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);

	while (a < b)
	{
		if (int(k1*(centres[(*a)*4+axis] - c_aabb.min[axis])) <=  plane && a <= b) {
			a++;
		} else if (int(k1*(centres[(*b)*4+axis] - c_aabb.min[axis])) >  plane && a <= b) {
			b--;
		} else if (a < b) {
			std::swap(*a, *b);
		}
	}
	mid = a;
}

static const int32_t permuteLesserInt[64] __attribute__((aligned(16))) = {
    0,1,2,3,
    1,2,3,0,
    0,2,3,0,
    2,3,0,0,
    0,1,3,0,
    1,3,0,0,
    0,3,0,0,
    3,0,0,0,
    0,1,2,0,
    1,2,0,0,
    0,2,0,0,
    2,0,0,0,
    0,1,0,0,
    1,0,0,0,
    0,0,0,0,
    0,0,0,0 
};

static const int32_t permuteGreaterInt[64] __attribute__((aligned(16))) = {
    0,0,0,0,
    0,0,0,0,
    1,0,0,0,
    0,1,0,0,
    2,0,0,0,
    0,2,0,0,
    1,2,0,0,
    0,1,2,0,
    3,0,0,0,
    0,3,0,0,
    1,3,0,0,
    0,1,3,0,
    2,3,0,0,
    0,2,3,0,
    1,2,3,0,
    0,1,2,3
};

static const int32_t sseActiveMaskInt[16] __attribute__((aligned(16))) = {
    0,0,0,0,
    -1,0,0,0,
    -1,-1,0,0,
    -1,-1,-1,0
};

static const int32_t activeMask[4] = {
	0, 1, 3, 7
};

__m128i* permuteLesser  = (__m128i*)permuteLesserInt;
__m128i* permuteGreater = (__m128i*)permuteGreaterInt;
__m128i* sseActiveMask  = (__m128i*)sseActiveMaskInt;

void partition_sse( const int 	plane,
					const int 	axis,
					const AABBh &c_aabb,
					const std::vector<simdf<4>> &centres,
					std::vector<uint32_t>::iterator begin, 
					std::vector<uint32_t>::iterator end,
					std::vector<uint32_t>           &scratch1,
					std::vector<uint32_t>           &scratch2,
					std::vector<uint32_t>::iterator &mid)
{	
	std::vector<uint32_t>::iterator left_it  = scratch1.begin();
	std::vector<uint32_t>::iterator right_it = scratch2.begin();
	
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
    __m128 k2     = _mm_broadcast_ss(&k1);
    __m128i p     = _mm_castps_si128(_mm_broadcast_ss((float*)&plane));
	
	auto it = begin;
	for (it = begin; it <= end-4; it += 4)
	{
		// Load 4 indices
		__m128i i = _mm_castps_si128(_mm_loadu_ps((float*)&(*it)));
		__m128i i4 = _mm_slli_epi32(i, 2);
		
		// Gather 4 float centres assuming stride of 4
		__m128 c = _mm_i32gather_ps(&centres[0][axis], i4, 4);

		// Compute the bin id of each
		__m128 cmin = _mm_broadcast_ss(&c_aabb.min[axis]);
		__m128 d = _mm_sub_ps(c, cmin);
		d = _mm_mul_ps(d, k2);
		__m128i ci = _mm_cvttps_epi32(d);
		
		// Compare against the plane: c > p ? 0xFFFFFFFF : 0
		// 0xFF is right side, 0 is left
		ci = _mm_cmpgt_epi32(ci, p);

		// Extract bitmask
		int r = _mm_movemask_ps(_mm_castsi128_ps(ci));

		// Consruct vectors with all elements lesser and all greater than the plane packed
		__m128 left  = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteLesser[r]);
		__m128 right = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteGreater[r]);

		int rc = _mm_popcnt_u32(r);
		int lc = 4 - rc;

		_mm_storeu_ps((float*)&(*left_it), left);
		_mm_storeu_ps((float*)&(*right_it), right);
		
		left_it += lc;
		right_it += rc;
	}
	
#if 1
	if (end != it)
	{
		unsigned count_left = end-it;

		__m128i i    = _mm_maskload_epi32((int*)&(*it), sseActiveMask[count_left]);
		__m128i i4   = _mm_slli_epi32(i, 2);
		__m128  c;
		__m128  mask = _mm_castsi128_ps(sseActiveMask[count_left]);
		c = _mm_mask_i32gather_ps(c, &centres[0][axis], i4, mask, 4);
		__m128  cmin = _mm_broadcast_ss(&c_aabb.min[axis]);
		__m128  d    = _mm_sub_ps(c, cmin);
		d = _mm_mul_ps(d, k2);
		__m128i ci = _mm_cvttps_epi32(d);
		ci = _mm_cmpgt_epi32(ci, p);
		
		int r = _mm_movemask_ps(_mm_castsi128_ps(ci));
		__m128 left  = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteLesser[r]);
		__m128 right = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteGreater[r]);
		
		int rc = _mm_popcnt_u32(r & activeMask[count_left]);
		int lc = count_left - rc;
			
		_mm_storeu_ps((float*)&(*left_it), left);
		_mm_storeu_ps((float*)&(*right_it), right);
		
		left_it += lc;
		right_it += rc;
	}
#else
	// Finish the rest serially
	for (; it < end; it++)
	{
		if (int(k1*(centres[(*it)*4+axis] - c_aabb.min[axis])) <=  plane) {
			*left_it = *it;
			left_it++;
		} else {
			*right_it = *it;
			right_it++;
		}
	}
#endif
	
	// Copy the temp arrays back into the main one
	unsigned left_count = left_it - scratch1.begin();
	//unsigned right_count = right_it - scratch2.begin();
	
	//memcpy(&(*begin), &(*scratch1.begin()), left_count*sizeof(unsigned));
	//memcpy(&(*(begin+left_count)), &(*scratch2.begin()), right_count*sizeof(unsigned));
	
	std::copy(scratch1.begin(), left_it, begin);
	std::copy(scratch2.begin(), right_it, begin + left_count);
	
	mid = begin + left_count;
}

void partition_m128( const int 	plane,
					const int 	axis,
					const AABBm &c_aabb,
					const std::vector<__m128> &centres,
					std::vector<uint32_t>::iterator begin, 
					std::vector<uint32_t>::iterator end,
					std::vector<uint32_t>           &scratch1,
					std::vector<uint32_t>           &scratch2,
					std::vector<uint32_t>::iterator &mid)
{	
	std::vector<uint32_t>::iterator left_it  = scratch1.begin();
	std::vector<uint32_t>::iterator right_it = scratch2.begin();
	
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
    __m128 k2     = _mm_broadcast_ss(&k1);
    __m128i p     = _mm_castps_si128(_mm_broadcast_ss((float*)&plane));
	
	auto it = begin;
	for (it = begin; it <= end-4; it += 4)
	{
		// Load 4 indices
		__m128i i = _mm_castps_si128(_mm_loadu_ps((float*)&(*it)));
		__m128i i4 = _mm_slli_epi32(i, 2);
		
		// Gather 4 float centres assuming stride of 4
		__m128 c = _mm_i32gather_ps(((float*)&centres[0]) + axis, i4, 4);

		// Compute the bin id of each
		__m128 cmin = _mm_broadcast_ss(&c_aabb.min[axis]);
		__m128 d = _mm_sub_ps(c, cmin);
		d = _mm_mul_ps(d, k2);
		__m128i ci = _mm_cvttps_epi32(d);
		
		// Compare against the plane: c > p ? 0xFFFFFFFF : 0
		// 0xFF is right side, 0 is left
		ci = _mm_cmpgt_epi32(ci, p);

		// Extract bitmask
		int r = _mm_movemask_ps(_mm_castsi128_ps(ci));

		// Consruct vectors with all elements lesser and all greater than the plane packed
		__m128 left  = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteLesser[r]);
		__m128 right = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteGreater[r]);

		int rc = _mm_popcnt_u32(r);
		int lc = 4 - rc;

		_mm_storeu_ps((float*)&(*left_it), left);
		_mm_storeu_ps((float*)&(*right_it), right);
		
		left_it += lc;
		right_it += rc;
	}
	
#if 1
	if (end != it)
	{
		unsigned count_left = end-it;

		__m128i i    = _mm_maskload_epi32((int*)&(*it), sseActiveMask[count_left]);
		__m128i i4   = _mm_slli_epi32(i, 2);
		__m128  c;
		__m128  mask = _mm_castsi128_ps(sseActiveMask[count_left]);
		c = _mm_mask_i32gather_ps(c, ((float*)&centres[0]) + axis, i4, mask, 4);
		__m128  cmin = _mm_broadcast_ss(&c_aabb.min[axis]);
		__m128  d    = _mm_sub_ps(c, cmin);
		d = _mm_mul_ps(d, k2);
		__m128i ci = _mm_cvttps_epi32(d);
		ci = _mm_cmpgt_epi32(ci, p);
		
		int r = _mm_movemask_ps(_mm_castsi128_ps(ci));
		__m128 left  = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteLesser[r]);
		__m128 right = _mm_permutevar_ps(_mm_castsi128_ps(i), permuteGreater[r]);
		
		int rc = _mm_popcnt_u32(r & activeMask[count_left]);
		int lc = count_left - rc;
			
		_mm_storeu_ps((float*)&(*left_it), left);
		_mm_storeu_ps((float*)&(*right_it), right);
		
		left_it += lc;
		right_it += rc;
	}
#else
	// Finish the rest serially
	for (; it < end; it++)
	{
		if (int(k1*(centres[(*it)*4+axis] - c_aabb.min[axis])) <=  plane) {
			*left_it = *it;
			left_it++;
		} else {
			*right_it = *it;
			right_it++;
		}
	}
#endif
	
	// Copy the temp arrays back into the main one
	unsigned left_count = left_it - scratch1.begin();
	//unsigned right_count = right_it - scratch2.begin();
	
	//memcpy(&(*begin), &(*scratch1.begin()), left_count*sizeof(unsigned));
	//memcpy(&(*(begin+left_count)), &(*scratch2.begin()), right_count*sizeof(unsigned));
	
	std::copy(scratch1.begin(), left_it, begin);
	std::copy(scratch2.begin(), right_it, begin + left_count);
	
	mid = begin + left_count;
}

#define BLOCKSIZE 64

void partition_branchless(
				int 	plane,
				int 	axis,
				AABB1  &c_aabb,
				const std::vector<float> &centres,
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				std::vector<uint32_t>::iterator &mid)
{
	// Use two iterators, one left to right and one right to left.
	// Increment the first until a value to the right of the plane is found
	// Increment the second until a value to the left of the plane is found
	// Swap the values and repeat until they meet
	std::vector<uint32_t>::iterator a = begin;
	std::vector<uint32_t>::iterator b = end-1;
	
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	unsigned left_buffer[BLOCKSIZE];
	unsigned right_buffer[BLOCKSIZE];
	unsigned num_l = 0, num_r = 0, start_l = 0, start_r = 0;
	
	// Use branchless blocks as long as they don't overlap
	while (a + BLOCKSIZE*2 < b)
	{
		if (num_l == 0)
		{
			start_l = 0;
			for (unsigned i = 0; i < BLOCKSIZE; i++)
			{
				int bucket = int(k1*(centres[*(a+i)*4+axis] - c_aabb.min[axis]));
				left_buffer[num_l] = i; num_l += (bucket > plane);
			}
		}
		
		if (num_r == 0)
		{
			start_r = 0;
			for (unsigned i = 0; i < BLOCKSIZE; i++)
			{
				int bucket = int(k1*(centres[*(b-i)*4+axis] - c_aabb.min[axis]));
				right_buffer[num_r] = i; num_r += (bucket <= plane);
			}
		}
		
		 // Swap the minimum number of elements
		int num = (std::min)(num_l, num_r);
		for (int i = 0; i < num; i++)
		{
			std::iter_swap(a+left_buffer[start_l+i], b-right_buffer[start_r+i]);
		}
			
		num_l -= num; num_r -= num;
		start_l += num; start_r += num;
		a += BLOCKSIZE*(num_l == 0);
		b -= BLOCKSIZE*(num_r == 0);
	}
	
	// Finish the last block with non-branchless method
	while (a < b)
	{
		if (int(k1*(centres[*a*4+axis] - c_aabb.min[axis])) <=  plane && a <= b)
			a++;
		else if (int(k1*(centres[*b*4+axis] - c_aabb.min[axis])) >  plane && a <= b)
			b--;
		else if (a < b)
			std::swap(*a, *b);
	}
	mid = a + (int(k1*(centres[*a*4+axis] - c_aabb.min[axis])) <=  plane);
}

#define BLOCKSIZE2 16

void partition_branchless2(
				int 	plane,
				int 	axis,
				AABB1  &c_aabb,
				const std::vector<vec3f1> &centres,
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				std::vector<uint32_t>::iterator &mid)
{
	// Use two iterators, one left to right and one right to left.
	// Increment the first until a value to the right of the plane is found
	// Increment the second until a value to the left of the plane is found
	// Swap the values and repeat until they meet
	std::vector<uint32_t>::iterator a = begin;
	std::vector<uint32_t>::iterator b = end-1;
	
	float epsilon = 1.1920929e-7; // 2^-23
	float k1      = NUM_BINS*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	unsigned left_buffer[BLOCKSIZE2];
	unsigned right_buffer[BLOCKSIZE2];
	unsigned num_l = 0, num_r = 0, start_l = 0, start_r = 0;
	
	// Use branchless blocks as long as they don't overlap
	while (a + BLOCKSIZE2*2 < b)
	{
		if (num_l == 0)
		{
			start_l = 0;
			for (unsigned i = 0; i < BLOCKSIZE2; i++)
			{
				int bucket = int(k1*(centres[*(a+i)][axis] - c_aabb.min[axis]));
				left_buffer[num_l] = i; num_l += (bucket > plane);
			}
		}
		
		if (num_r == 0)
		{
			start_r = 0;
			for (unsigned i = 0; i < BLOCKSIZE2; i++)
			{
				int bucket = int(k1*(centres[*(b-i)][axis] - c_aabb.min[axis]));
				right_buffer[num_r] = i; num_r += (bucket <= plane);
			}
		}
		
		 // Swap the minimum number of elements
		int num = (std::min)(num_l, num_r);
		for (int i = 0; i < num; i++)
		{
			std::iter_swap(a+left_buffer[start_l+i], b-right_buffer[start_r+i]);
		}
			
		num_l -= num; num_r -= num;
		start_l += num; start_r += num;
		a += BLOCKSIZE2*(num_l == 0);
		b -= BLOCKSIZE2*(num_r == 0);
	}
	
	// Finish the last block with non-branchless method
	while (a < b)
	{
		if (int(k1*(centres[*a][axis] - c_aabb.min[axis])) <=  plane && a <= b)
			a++;
		else if (int(k1*(centres[*b][axis] - c_aabb.min[axis])) >  plane && a <= b)
			b--;
		else if (a < b)
			std::swap(*a, *b);
	}
	mid = a + (int(k1*(centres[*a][axis] - c_aabb.min[axis])) <=  plane);
}

std::pair<unsigned, unsigned>
	recurse(	const std::vector<AABB1 >      &aabbs,
				const std::vector<float >      &centres,
				std::vector<Bin>                bins[],
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				AABB1                           p_aabb,
				AABB1                           c_aabb,
				Node1                           nodes[],
				unsigned                        free_index)
{	
	AABB1 left_p_aabb;
	AABB1 right_p_aabb;
	AABB1 left_c_aabb;
	AABB1 right_c_aabb;
	std::vector<uint32_t>::iterator mid;
	std::vector<uint32_t>::iterator mid2;
	unsigned count = end-begin;
	
	bool triangleThresholdReached = count <= 8;
	bool centroidBoundsTooSmall   = c_aabb.sa() <= 0.0f;
	
	if (triangleThresholdReached || centroidBoundsTooSmall)
	{
		// Create leaf nodes
		for (unsigned i = 0; i < count; i++, begin++)
		{
			Node1 &leaf_node = nodes[free_index++];
			leaf_node.min    = aabbs[*begin].min;
			leaf_node.max    = aabbs[*begin].max;
			leaf_node.child  = *begin;
			leaf_node.count  = 1;
			leaf_node.flags  = AABB_FLAGS_LEAF;
		}
		
		return std::make_pair(free_index-count, count);;
	}
	
	unsigned tid = omp_get_thread_num();
	
	// Choose the axis
	int axis = choose_axis(c_aabb);
	
	// Bin the centroids and populate the bin structs
	bin_centroids(aabbs, centres, begin, end, c_aabb, axis, bins[tid]);
	
	// Evaluate partitions
	int plane_id = choose_plane(bins[tid], left_p_aabb, right_p_aabb, left_c_aabb, right_c_aabb);

	// Reorder the triangle id array
#if 1
	partition(plane_id, axis, c_aabb, centres, begin, end, mid);
#else
	if (count > 128)
	partition_branchless(plane_id, axis, c_aabb, centres, begin, end, mid);
	else
	partition(plane_id, axis, c_aabb, centres, begin, end, mid);
#endif

	assert(mid != begin);
	assert(mid != end);
	
	// Todo: add in exit if split cost is worse than leaf cost
	unsigned left_count  = mid-begin;
	unsigned right_count = end-mid;
	unsigned pair_index = free_index;
	
	// Reserve space for the two child nodes
	unsigned left_index  = free_index;
	free_index += (left_count != 1);
	unsigned right_index = free_index;
	free_index += (right_count != 1);
	
	unsigned right_free_index = free_index + (left_count-1)*2 + (left_count==1);
	
	std::pair<unsigned, unsigned> left_child;
	std::pair<unsigned, unsigned> right_child;

	left_child = recurse(aabbs, centres, bins, begin, mid,  left_p_aabb,  left_c_aabb, nodes, free_index);
	right_child = recurse(aabbs, centres, bins,   mid, end, right_p_aabb, right_c_aabb, nodes, right_free_index);
	
	// Create the two split nodes
	if (left_count != 1)
	{
		Node1 &left_node = nodes[left_index];
		left_node.min   = left_p_aabb.min;
		left_node.max   = left_p_aabb.max;
		left_node.child = left_child.first;
		left_node.count = left_child.second;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1)
	{
		Node1 &right_node = nodes[right_index];
		right_node.min   = right_p_aabb.min;
		right_node.max   = right_p_aabb.max;
		right_node.child = right_child.first;
		right_node.count = right_child.second;
		right_node.flags = AABB_FLAGS_NONE;
	}

	return std::make_pair(pair_index, 2);
}

std::pair<unsigned, unsigned>
	recurse_sse(const std::vector<AABBh >      &aabbs,
				const std::vector<simdf<4>>    &centres,
				std::vector<Binh>               bins[],
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				std::vector<uint32_t>          &scratch1,
				std::vector<uint32_t>          &scratch2,
				AABBh                           p_aabb,
				AABBh                           c_aabb,
				Node1                           nodes[],
				unsigned                        free_index)
{	
	AABBh left_p_aabb;
	AABBh right_p_aabb;
	AABBh left_c_aabb;
	AABBh right_c_aabb;
	std::vector<uint32_t>::iterator mid;
	std::vector<uint32_t>::iterator mid2;
	unsigned count = end-begin;
	
	bool triangleThresholdReached = count <= 8;
	bool centroidBoundsTooSmall   = c_aabb.sa() <= 0.0f;

	if (triangleThresholdReached || centroidBoundsTooSmall)
	{
		// Create leaf nodes
		for (unsigned i = 0; i < count; i++, begin++)
		{
			Node1 &leaf_node = nodes[free_index++];
			leaf_node.min    = vec3f1(aabbs[*begin].min[0], aabbs[*begin].min[1], aabbs[*begin].min[2]);
			leaf_node.max    = vec3f1(aabbs[*begin].max[0], aabbs[*begin].max[1], aabbs[*begin].max[2]);
			leaf_node.child  = *begin;
			leaf_node.count  = 1;
			leaf_node.flags  = AABB_FLAGS_LEAF;
		}
		
		return std::make_pair(free_index-count, count);;
	}
	
	unsigned tid = omp_get_thread_num();
	
	// Choose the axis
	int axis = choose_axis_sse(c_aabb);
	
	// Bin the centroids and populate the bin structs
	bin_centroids_sse(aabbs, centres, begin, end, c_aabb, axis, bins[tid]);
	
	// Evaluate partitions
	int plane_id = choose_plane_sse(bins[tid], left_p_aabb, right_p_aabb, left_c_aabb, right_c_aabb);

	// Reorder the triangle id array
#if 1
	//partition(plane_id, axis, c_aabb, centres, begin, end, mid);
	partition_sse(plane_id, axis, c_aabb, centres, begin, end, scratch1, scratch2, mid);
#else
	if (count > 128)
	partition_branchless(plane_id, axis, c_aabb, centres, begin, end, mid);
	else
	partition(plane_id, axis, c_aabb, centres, begin, end, mid);
#endif

	assert(mid != begin);
	assert(mid != end);
	
	// Todo: add in exit if split cost is worse than leaf cost
	unsigned left_count  = mid-begin;
	unsigned right_count = end-mid;
	unsigned pair_index = free_index;
	
	// Reserve space for the two child nodes
	unsigned left_index  = free_index;
	free_index += (left_count != 1);
	unsigned right_index = free_index;
	free_index += (right_count != 1);
	
	unsigned right_free_index = free_index + (left_count-1)*2 + (left_count==1);
	
	std::pair<unsigned, unsigned> left_child;
	std::pair<unsigned, unsigned> right_child;

	left_child = recurse_sse(aabbs, centres, bins, begin, mid,  scratch1, scratch2, left_p_aabb,  left_c_aabb, nodes, free_index);
	right_child = recurse_sse(aabbs, centres, bins,   mid, end, scratch1, scratch2, right_p_aabb, right_c_aabb, nodes, right_free_index);
	
	// Create the two split nodes
	if (left_count != 1)
	{
		Node1 &left_node = nodes[left_index];
		left_node.min   = vec3f1(left_p_aabb.min[0], left_p_aabb.min[1], left_p_aabb.min[2]);
		left_node.max   = vec3f1(left_p_aabb.max[0], left_p_aabb.max[1], left_p_aabb.max[2]);
		left_node.child = left_child.first;
		left_node.count = left_child.second;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1)
	{
		Node1 &right_node = nodes[right_index];
		right_node.min   = vec3f1(right_p_aabb.min[0], right_p_aabb.min[1], right_p_aabb.min[2]);
		right_node.max   = vec3f1(right_p_aabb.max[0], right_p_aabb.max[1], right_p_aabb.max[2]);
		right_node.child = right_child.first;
		right_node.count = right_child.second;
		right_node.flags = AABB_FLAGS_NONE;
	}

	return std::make_pair(pair_index, 2);
}

std::pair<unsigned, unsigned>
	recurse_m128(const std::vector<AABBm >      &aabbs,
				const std::vector<__m128>      &centres,
				std::vector<Binm>               bins[],
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				std::vector<uint32_t>          &scratch1,
				std::vector<uint32_t>          &scratch2,
				AABBm                           p_aabb,
				AABBm                           c_aabb,
				Node1                           nodes[],
				unsigned                        free_index)
{	
	AABBm left_p_aabb;
	AABBm right_p_aabb;
	AABBm left_c_aabb;
	AABBm right_c_aabb;
	std::vector<uint32_t>::iterator mid;
	unsigned count = end-begin;
	
	bool triangleThresholdReached = count <= 8;
	bool centroidBoundsTooSmall   = c_aabb.sa() <= 0.0f;

	if (triangleThresholdReached || centroidBoundsTooSmall)
	{
		// Create leaf nodes
		for (unsigned i = 0; i < count; i++, begin++)
		{
			Node1 &leaf_node = nodes[free_index++];
			leaf_node.min    = vec3f1(aabbs[*begin].min[0], aabbs[*begin].min[1], aabbs[*begin].min[2]);
			leaf_node.max    = vec3f1(aabbs[*begin].max[0], aabbs[*begin].max[1], aabbs[*begin].max[2]);
			leaf_node.child  = *begin;
			leaf_node.count  = 1;
			leaf_node.flags  = AABB_FLAGS_LEAF;
		}
		
		return std::make_pair(free_index-count, count);;
	}
	
	unsigned tid = omp_get_thread_num();
	
	// Choose the axis
	int axis = choose_axis_m128(c_aabb);
	
	// Bin the centroids and populate the bin structs
	bin_centroids_m128(aabbs, centres, begin, end, c_aabb, axis, bins[tid]);
	
	// Evaluate partitions
	int plane_id = choose_plane_m128(bins[tid], left_p_aabb, right_p_aabb, left_c_aabb, right_c_aabb);

	// Reorder the triangle id array
	partition_m128(plane_id, axis, c_aabb, centres, begin, end, scratch1, scratch2, mid);

	assert(mid != begin);
	assert(mid != end);
	
	// Todo: add in exit if split cost is worse than leaf cost
	unsigned left_count  = mid-begin;
	unsigned right_count = end-mid;
	unsigned pair_index = free_index;
	
	// Reserve space for the two child nodes
	unsigned left_index  = free_index;
	free_index += (left_count != 1);
	unsigned right_index = free_index;
	free_index += (right_count != 1);
	
	unsigned right_free_index = free_index + (left_count-1)*2 + (left_count==1);
	
	std::pair<unsigned, unsigned> left_child;
	std::pair<unsigned, unsigned> right_child;

	left_child = recurse_m128(aabbs, centres, bins, begin, mid,  scratch1, scratch2, left_p_aabb,  left_c_aabb, nodes, free_index);
	right_child = recurse_m128(aabbs, centres, bins,   mid, end, scratch1, scratch2, right_p_aabb, right_c_aabb, nodes, right_free_index);
	
	// Create the two split nodes
	if (left_count != 1)
	{
		Node1 &left_node = nodes[left_index];
		left_node.min   = vec3f1(left_p_aabb.min[0], left_p_aabb.min[1], left_p_aabb.min[2]);
		left_node.max   = vec3f1(left_p_aabb.max[0], left_p_aabb.max[1], left_p_aabb.max[2]);
		left_node.child = left_child.first;
		left_node.count = left_child.second;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1)
	{
		Node1 &right_node = nodes[right_index];
		right_node.min   = vec3f1(right_p_aabb.min[0], right_p_aabb.min[1], right_p_aabb.min[2]);
		right_node.max   = vec3f1(right_p_aabb.max[0], right_p_aabb.max[1], right_p_aabb.max[2]);
		right_node.child = right_child.first;
		right_node.count = right_child.second;
		right_node.flags = AABB_FLAGS_NONE;
	}

	return std::make_pair(pair_index, 2);
}

std::pair<unsigned, unsigned>
	recurse_omp(const std::vector<AABB1 >      &aabbs,
				const std::vector<float >      &centres,
				std::vector<Bin>                bins[],
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				AABB1                           p_aabb,
				AABB1                           c_aabb,
				Node1                           nodes[],
				unsigned                        free_index)
{	
	AABB1 left_p_aabb;
	AABB1 right_p_aabb;
	AABB1 left_c_aabb;
	AABB1 right_c_aabb;
	std::vector<uint32_t>::iterator mid;
	unsigned count = end-begin;
	
	bool triangleThresholdReached = count <= 8;
	bool centroidBoundsTooSmall   = c_aabb.sa() <= 0;

	if (triangleThresholdReached || centroidBoundsTooSmall)
	{
		// Create leaf nodes
		for (unsigned i = 0; i < count; i++, begin++)
		{
			Node1 &leaf_node = nodes[free_index++];
			leaf_node.min    = aabbs[*begin].min;
			leaf_node.max    = aabbs[*begin].max;
			leaf_node.child  = *begin;
			leaf_node.count  = 1;
			leaf_node.flags  = AABB_FLAGS_LEAF;
		}
		
		return std::make_pair(free_index-count, count);;
	}
	
	unsigned tid = omp_get_thread_num();
	
	// Choose the axis
	int axis = choose_axis(c_aabb);
	
	// Bin the centroids and populate the bin structs
	bin_centroids(aabbs, centres, begin, end, c_aabb, axis, bins[tid]);
	
	// Evaluate partitions
	int plane_id = choose_plane(bins[tid], left_p_aabb, right_p_aabb, left_c_aabb, right_c_aabb);

	// Reorder the triangle id array
	partition(plane_id, axis, c_aabb, centres, begin, end, mid);
	
	assert(mid != begin);
	assert(mid != end);
	
	// Todo: add in exit if split cost is worse than leaf cost
	unsigned left_count  = mid-begin;
	unsigned right_count = end-mid;
	unsigned pair_index = free_index;
	
	// Reserve space for the two child nodes
	unsigned left_index  = free_index;
	free_index += (left_count != 1);
	unsigned right_index = free_index;
	free_index += (right_count != 1);
	
	unsigned right_free_index = free_index + (left_count-1)*2 + (left_count==1);
	
	std::pair<unsigned, unsigned> left_child;
	std::pair<unsigned, unsigned> right_child;

	if (left_count > 10000) {
		#pragma omp task shared(aabbs, centres, bins, nodes, left_child)
		left_child = recurse_omp(aabbs, centres, bins, begin, mid,  left_p_aabb,  left_c_aabb, nodes, free_index);
	} 
	else
	{
		left_child = recurse(aabbs, centres, bins, begin, mid,  left_p_aabb,  left_c_aabb, nodes, free_index);
	}
	
	if (right_count > 10000) {
		//#pragma omp task shared(aabbs, centres, bins, nodes, right_child)
		right_child = recurse_omp(aabbs, centres, bins,   mid, end, right_p_aabb, right_c_aabb, nodes, right_free_index);
	}
	else
	{
		right_child = recurse(aabbs, centres, bins,   mid, end, right_p_aabb, right_c_aabb, nodes, right_free_index);
	}
	
	#pragma omp taskwait
	
	// Create the two split nodes
	if (left_count != 1)
	{
		Node1 &left_node = nodes[left_index];
		left_node.min   = left_p_aabb.min;
		left_node.max   = left_p_aabb.max;
		left_node.child = left_child.first;//free_index;
		left_node.count = left_child.second;//2;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1)
	{
		Node1 &right_node = nodes[right_index];
		right_node.min   = right_p_aabb.min;
		right_node.max   = right_p_aabb.max;
		right_node.child = right_child.first;//right_free_index;
		right_node.count = right_child.second;//2;
		right_node.flags = AABB_FLAGS_NONE;
	}

	return std::make_pair(pair_index, 2);
}

std::pair<unsigned, unsigned>
	recurse_horizontal(const std::vector<AABB1 >      &aabbs,
				const std::vector<float >      &centres,
				std::vector<Bin>                bins[],
				std::vector<uint32_t>::iterator begin, 
				std::vector<uint32_t>::iterator end,
				AABB1                           p_aabb,
				AABB1                           c_aabb,
				Node1                           nodes[],
				unsigned                        free_index)
{	
	AABB1 left_p_aabb;
	AABB1 right_p_aabb;
	AABB1 left_c_aabb;
	AABB1 right_c_aabb;
	std::vector<uint32_t>::iterator mid;
	unsigned count = end-begin;
	
	unsigned tid = omp_get_thread_num();
	
	assert(count > 64);
	
	// Choose the axis
	int axis = choose_axis(c_aabb);
	
	// Bin the centroids and populate the bin structs
	bin_centroids_reduction(aabbs, centres, begin, end, c_aabb, axis, bins[tid]);
		
	// Evaluate partitions
	int plane_id = choose_plane(bins[tid], left_p_aabb, right_p_aabb, left_c_aabb, right_c_aabb);
		
	// Reorder the triangle id array
	partition(plane_id, axis, c_aabb, centres, begin, end, mid);
	
	assert(mid != begin);
	assert(mid != end);
	
	// Todo: add in exit if split cost is worse than leaf cost
	unsigned left_count  = mid-begin;
	unsigned right_count = end-mid;
	unsigned pair_index = free_index;
	
	// Reserve space for the two child nodes
	unsigned left_index  = free_index;
	free_index += (left_count != 1);
	unsigned right_index = free_index;
	free_index += (right_count != 1);
	
	unsigned right_free_index = free_index + (left_count-1)*2 + (left_count==1);
	
	std::pair<unsigned, unsigned> left_child;
	std::pair<unsigned, unsigned> right_child;

	if (left_count > 100000)
		#pragma omp task shared(aabbs, centres, bins, nodes, left_child)
		left_child = recurse_horizontal(aabbs, centres, bins, begin, mid,  left_p_aabb,  left_c_aabb, nodes, free_index);
	else
	{
		#pragma omp task shared(aabbs, centres, bins, nodes, left_child)
		left_child = recurse_omp(aabbs, centres, bins, begin, mid,  left_p_aabb,  left_c_aabb, nodes, free_index);
	}
	
	if (right_count > 100000)
		#pragma omp task shared(aabbs, centres, bins, nodes, right_child)
		right_child = recurse_horizontal(aabbs, centres, bins,   mid, end, right_p_aabb, right_c_aabb, nodes, right_free_index);
	else
	{
		#pragma omp task shared(aabbs, centres, bins, nodes, right_child)
		right_child = recurse_omp(aabbs, centres, bins, mid, end,  right_p_aabb,  right_c_aabb, nodes, right_free_index);
	}
	
	#pragma omp taskwait
	
	// Create the two split nodes
	if (left_count != 1)
	{
		Node1 &left_node = nodes[left_index];
		left_node.min   = left_p_aabb.min;
		left_node.max   = left_p_aabb.max;
		left_node.child = left_child.first;//free_index;
		left_node.count = left_child.second;//2;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1)
	{
		Node1 &right_node = nodes[right_index];
		right_node.min   = right_p_aabb.min;
		right_node.max   = right_p_aabb.max;
		right_node.child = right_child.first;//right_free_index;
		right_node.count = right_child.second;//2;
		right_node.flags = AABB_FLAGS_NONE;
	}

	return std::make_pair(pair_index, 2);
}

Hierarchy Builder::build_hierarchy(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	
	std::vector<AABB1 >   aabbs;
	std::vector<float >   centres;
	std::vector<Bin>      bins[NUM_THREADS];
	std::vector<uint32_t> triangleIds;
	AABB1 p_aabb;
	AABB1 c_aabb;
	
	Node1* nodes = new Node1[primitives.size()*2];
	
	for (auto &bin : bins)
		bin = std::vector<Bin>(NUM_BINS);
	
	omp_set_nested(1);
	
	// Initialise the triangle ids, primitive aabbs and centres, overall aabb and overall centres aabb
	setup_reduction(primitives, triangleIds, aabbs, centres, p_aabb, c_aabb);
	
	#pragma omp parallel
	{
		#pragma omp single
		{
			unsigned free_index = 0;
			std::pair<unsigned, unsigned> root = recurse_omp(aabbs, centres, bins, triangleIds.begin(), triangleIds.end(), p_aabb, c_aabb, nodes, free_index);
			 
			unsigned root_index = primitives.size()*2-1;
			nodes[root_index].child = root.first;
			nodes[root_index].count = root.second;
			nodes[root_index].min   = p_aabb.min;
			nodes[root_index].max   = p_aabb.max;
			nodes[root_index].flags = AABB_FLAGS_NONE;
			
			hierarchy.node_base  = nodes;
			hierarchy.root_index = root_index;
			hierarchy.tri_base = &primitives[0];
		}
	}
	
	return hierarchy;
}


Hierarchy Builder::build_hierarchy_horizontal(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	
	std::vector<AABB1 >   aabbs;
	std::vector<float >   centres;
	std::vector<Bin>      bins[NUM_THREADS];
	std::vector<uint32_t> triangleIds;
	AABB1 p_aabb;
	AABB1 c_aabb;
	
	Node1* nodes = new Node1[primitives.size()*2];
	
	for (auto &bin : bins)
		bin = std::vector<Bin>(NUM_BINS);
	
	omp_set_nested(1);
	
	// Initialise the triangle ids, primitive aabbs and centres, overall aabb and overall centres aabb
	setup(primitives, triangleIds, aabbs, centres, p_aabb, c_aabb);
	unsigned root_index = 0;
	
	#pragma omp parallel
	{
		#pragma omp single
		{
			unsigned free_index = 0;
			std::pair<unsigned, unsigned> root = recurse_horizontal(aabbs, centres, bins, triangleIds.begin(), triangleIds.end(), p_aabb, c_aabb, nodes, free_index);
		
			root_index = primitives.size()*2-1;
			nodes[root_index].child = root.first;
			nodes[root_index].count = root.second;
			nodes[root_index].min   = p_aabb.min;
			nodes[root_index].max   = p_aabb.max;
			nodes[root_index].flags = AABB_FLAGS_NONE;
			
			hierarchy.node_base  = nodes;
			hierarchy.root_index = root_index;
		}
	}
	
	return hierarchy;
}

void bin_centroids_to_grid(const std::vector<AABB1>   &aabbs,
						   const std::vector<float>       &centres,
						   const AABB1                    c_aabb,
						   std::vector<unsigned>          &triangles,
						   Bin                             bins[GRID_DIM][GRID_DIM][GRID_DIM],
						   unsigned                        thread_begin[NUM_CELLS+1])
{
	unsigned num_bins[3] = {GRID_DIM,GRID_DIM,GRID_DIM};
	float    epsilon     = 1.1920929e-7; // 2^-23
	float    k1      [3];
	
	std::vector<unsigned> grid[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	Bin thread_bins[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
	{
		bins[i][j][k].reset();
		for (unsigned x = 0; x < NUM_THREADS; x++)
		{
			thread_bins[x][i][j][k].reset();
			grid[x][i][j][k].reserve(64);
		}
	}
	
	for (unsigned axis = 0; axis < 3; axis++)
		k1[axis] = num_bins[axis]*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	// Bin the triangles to the grid
	#pragma omp parallel for shared(aabbs, centres, thread_bins)
	for (unsigned i = 0; i < centres.size()/4; i++)
	{
		int bin_id[3];
		unsigned tid = omp_get_thread_num();
			
		for (unsigned axis = 0; axis < 3; axis++)
		{
			bin_id[axis] = int(k1[axis]*(centres[i*4+axis] - c_aabb.min[axis]));
			assert(bin_id[axis] >= 0 && bin_id[axis] < GRID_DIM);
		}
		
		grid[tid][bin_id[2]][bin_id[1]][bin_id[0]].push_back(i);
		
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].p_aabb.grow(aabbs[i]);
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].c_aabb.grow(vec3f1(centres[i*4], centres[i*4+1], centres[i*4+2]));
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].count += 1;
	}
	
	// Combine the per thread bins
	for (unsigned t = 0; t < NUM_THREADS; t++)
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
		bins[i][j][k] = bins[i][j][k].combine(thread_bins[t][i][j][k]);
	
	// Prefix sum to compute the start of each thread
	thread_begin[NUM_CELLS] = centres.size()/4;
	unsigned prefix_sum = 0;
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		thread_begin[i] = prefix_sum;
		
		prefix_sum += bins[kz][ky][kx].count;
	}
	
	#pragma omp parallel for shared(grid, triangles)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		assert(kx < GRID_DIM);
		assert(ky < GRID_DIM);
		assert(kz < GRID_DIM);
		
		unsigned thread_sum = 0;
		for (unsigned t = 0; t < NUM_THREADS; t++)
		{
			std::copy(grid[t][kz][ky][kx].begin(), grid[t][kz][ky][kx].end(), triangles.begin() + thread_begin[i] + thread_sum);
			thread_sum += grid[t][kz][ky][kx].size();
		}
	}
}

void bin_centroids_to_grid_sse(const std::vector<AABBh>   &aabbs,
						   const std::vector<simdf<4>>    &centres,
						   const AABBh                     c_aabb,
						   std::vector<unsigned>          &triangles,
						   Binh                            bins[GRID_DIM][GRID_DIM][GRID_DIM],
						   unsigned                        thread_begin[NUM_CELLS+1])
{
	unsigned num_bins[3] = {GRID_DIM,GRID_DIM,GRID_DIM};
	float    epsilon     = 1.1920929e-7; // 2^-23
	float    k1      [3];
	
	std::vector<unsigned> grid[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	Binh thread_bins[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
	{
		bins[i][j][k].reset();
		for (unsigned x = 0; x < NUM_THREADS; x++)
		{
			thread_bins[x][i][j][k].reset();
			grid[x][i][j][k].reserve(64);
		}
	}
	
	for (unsigned axis = 0; axis < 3; axis++)
		k1[axis] = num_bins[axis]*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	// Bin the triangles to the grid
	#pragma omp parallel for shared(aabbs, centres, thread_bins)
	for (unsigned i = 0; i < centres.size(); i++)
	{
		int bin_id[3];
		unsigned tid = omp_get_thread_num();
			
		for (unsigned axis = 0; axis < 3; axis++)
		{
			bin_id[axis] = int(k1[axis]*(centres[i][axis] - c_aabb.min[axis]));
			assert(bin_id[axis] >= 0 && bin_id[axis] < 4);
		}
		
		grid[tid][bin_id[2]][bin_id[1]][bin_id[0]].push_back(i);
		
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].p_aabb.grow(aabbs[i]);
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].c_aabb.grow(centres[i]);
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].count += 1;
	}
	
	// Combine the per thread bins
	for (unsigned t = 0; t < NUM_THREADS; t++)
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
		bins[i][j][k] = bins[i][j][k].combine(thread_bins[t][i][j][k]);
	
	// Prefix sum to compute the start of each thread
	thread_begin[NUM_CELLS] = centres.size();
	unsigned prefix_sum = 0;
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		thread_begin[i] = prefix_sum;
		
		prefix_sum += bins[kz][ky][kx].count;
	}
	
	#pragma omp parallel for shared(grid, triangles)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		unsigned thread_sum = 0;
		for (unsigned t = 0; t < NUM_THREADS; t++)
		{
			std::copy(grid[t][kz][ky][kx].begin(), grid[t][kz][ky][kx].end(), triangles.begin() + thread_begin[i] + thread_sum);
			thread_sum += grid[t][kz][ky][kx].size();
		}
	}
}

void bin_centroids_to_grid_sse_reduction(const std::vector<AABBh>   &aabbs,
						   const std::vector<simdf<4>>    &centres,
						   const AABBh                     c_aabb,
						   std::vector<unsigned>          &triangles,
						   Binh                            bins[GRID_DIM][GRID_DIM][GRID_DIM],
						   unsigned                        thread_begin[NUM_CELLS+1])
{
	unsigned num_bins[3] = {GRID_DIM,GRID_DIM,GRID_DIM};
	float    epsilon     = 1.1920929e-7; // 2^-23
	float    k1      [3];
	
	std::vector<unsigned> grid[GRID_DIM][GRID_DIM][GRID_DIM];
	
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
	{
		bins[i][j][k].reset();
		grid[i][j][k].reserve(64);
	}
	
	for (unsigned axis = 0; axis < 3; axis++)
		k1[axis] = num_bins[axis]*(1-epsilon) / (c_aabb.max[axis] - c_aabb.min[axis]);
	
	// Bin the triangles to the grid
	#pragma omp parallel for shared(aabbs, centres) reduction(appvec : grid[:GRID_DIM][:GRID_DIM][:GRID_DIM]) reduction(combinh : bins[:GRID_DIM][:GRID_DIM][:GRID_DIM]) num_threads(16)
	for (unsigned i = 0; i < centres.size(); i++)
	{
		int bin_id[3];
		//unsigned tid = omp_get_thread_num();
			
		for (unsigned axis = 0; axis < 3; axis++)
		{
			bin_id[axis] = int(k1[axis]*(centres[i][axis] - c_aabb.min[axis]));
			//assert(bin_id[axis] >= 0 && bin_id[axis] < GRID_DIM);
		}
		
		grid[bin_id[2]][bin_id[1]][bin_id[0]].push_back(i);
		bins[bin_id[2]][bin_id[1]][bin_id[0]].p_aabb.grow(aabbs[i]);
		bins[bin_id[2]][bin_id[1]][bin_id[0]].c_aabb.grow(centres[i]);
		bins[bin_id[2]][bin_id[1]][bin_id[0]].count += 1;
	}
	
	// Prefix sum to compute the start of each thread
	thread_begin[NUM_CELLS] = centres.size();
	unsigned prefix_sum = 0;
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		thread_begin[i] = prefix_sum;
		prefix_sum += bins[kz][ky][kx].count;
	}
	
	#pragma omp parallel for shared(grid, triangles)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		std::copy(grid[kz][ky][kx].begin(), grid[kz][ky][kx].end(), triangles.begin() + thread_begin[i]);
	}
}

void bin_centroids_to_grid_m128(const std::vector<AABBm>     &aabbs,
						        const std::vector<__m128>    &centres,
						        const AABBm                   c_aabb,
						        std::vector<unsigned>        &triangles,
						        Binm                          bins[GRID_DIM][GRID_DIM][GRID_DIM],
						        unsigned                      thread_begin[NUM_CELLS+1])
{
	float  epsilon  = 1.1920929e-7; // 2^-23
	__m128 num_bins = _mm_set_ps1((float)GRID_DIM);
	__m128 eps = _mm_set_ps1(1-epsilon);
	__m128 k1;
	
	std::vector<unsigned> grid[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	Binm thread_bins[NUM_THREADS][GRID_DIM][GRID_DIM][GRID_DIM];
	
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
	{
		bins[i][j][k].reset();
		for (unsigned x = 0; x < NUM_THREADS; x++)
		{
			thread_bins[x][i][j][k].reset();
			grid[x][i][j][k].reserve(64);
		}
	}
	
	k1 = _mm_div_ps(_mm_mul_ps(num_bins, eps), _mm_sub_ps(c_aabb.max, c_aabb.min));
	
	// Bin the triangles to the grid
	#pragma omp parallel for shared(aabbs, centres, thread_bins)
	for (unsigned i = 0; i < centres.size(); i++)
	{
		unsigned tid = omp_get_thread_num();
			
		__m128i bin_idm = _mm_cvttps_epi32(_mm_mul_ps(k1, _mm_sub_ps(centres[i], c_aabb.min)));
		int* bin_id = (int*)&bin_idm;
		
		assert(bin_id[0] >= 0 && bin_id[0] < GRID_DIM);
		assert(bin_id[1] >= 0 && bin_id[0] < GRID_DIM);
		assert(bin_id[2] >= 0 && bin_id[0] < GRID_DIM);
		
		grid[tid][bin_id[2]][bin_id[1]][bin_id[0]].push_back(i);
		
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].p_aabb.grow(aabbs[i]);
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].c_aabb.grow(centres[i]);
		thread_bins[tid][bin_id[2]][bin_id[1]][bin_id[0]].count += 1;
	}
	
	// Combine the per thread bins
	for (unsigned t = 0; t < NUM_THREADS; t++)
	for (unsigned i = 0; i < GRID_DIM; i++)
	for (unsigned j = 0; j < GRID_DIM; j++)
	for (unsigned k = 0; k < GRID_DIM; k++)
		bins[i][j][k] = bins[i][j][k].combine(thread_bins[t][i][j][k]);
	
	// Prefix sum to compute the start of each thread
	thread_begin[NUM_CELLS] = centres.size();
	unsigned prefix_sum = 0;
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		thread_begin[i] = prefix_sum;
		prefix_sum += bins[kz][ky][kx].count;
	}
	
	#pragma omp parallel for shared(grid, triangles)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		unsigned thread_sum = 0;
		for (unsigned t = 0; t < NUM_THREADS; t++)
		{
			std::copy(grid[t][kz][ky][kx].begin(), grid[t][kz][ky][kx].end(), triangles.begin() + thread_begin[i] + thread_sum);
			thread_sum += grid[t][kz][ky][kx].size();
		}
	}
}

Hierarchy Builder::build_hierarchy_grid_sse(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	
	std::vector<AABBh >   aabbs;
	std::vector<simdf<4>> centres;
	std::vector<Binh>     bins[NUM_THREADS];
	std::vector<uint32_t> triangleIds;
	unsigned              thread_begin[NUM_CELLS+1];
	AABBh p_aabb;
	AABBh c_aabb;
	Binh grid_bins[GRID_DIM][GRID_DIM][GRID_DIM];
	
	Node1* nodes = new Node1[primitives.size()*2+NUM_CELLS*2];
	
	for (auto &bin : bins)
		bin = std::vector<Binh>(NUM_BINS);
	
	// Initialise the triangle ids, primitive aabbs and centres, overall aabb and overall centres aabb
	setup_sse_reduction(primitives, triangleIds, aabbs, centres, p_aabb, c_aabb);
	
	bin_centroids_to_grid_sse(aabbs, centres, c_aabb, triangleIds, grid_bins, thread_begin);
	//bin_centroids_to_grid_sse_reduction(aabbs, centres, c_aabb, triangleIds, grid_bins, thread_begin);

	unsigned offset = NUM_CELLS*2;
	
	#pragma omp parallel for shared(aabbs, centres, triangleIds)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		unsigned thread_count = thread_begin[i+1] - thread_begin[i];
		std::vector<unsigned> scratch1(thread_count);
		std::vector<unsigned> scratch2(thread_count);
		
		std::pair<unsigned, unsigned> child = recurse_sse(aabbs, centres, bins, 
			triangleIds.begin() + thread_begin[i], 
			triangleIds.begin() + thread_begin[i+1],
			scratch1,
			scratch2,
			grid_bins[kz][ky][kx].p_aabb,
			grid_bins[kz][ky][kx].c_aabb,
			nodes,
			offset + thread_begin[i]*2);
			
		nodes[i+NUM_CELLS-1].min   = vec3f1(grid_bins[kz][ky][kx].p_aabb.min[0], grid_bins[kz][ky][kx].p_aabb.min[1], grid_bins[kz][ky][kx].p_aabb.min[2]);
		nodes[i+NUM_CELLS-1].max   = vec3f1(grid_bins[kz][ky][kx].p_aabb.max[0], grid_bins[kz][ky][kx].p_aabb.max[1], grid_bins[kz][ky][kx].p_aabb.max[2]);
		nodes[i+NUM_CELLS-1].child = child.first;
		nodes[i+NUM_CELLS-1].count = child.second;
		nodes[i+NUM_CELLS-1].flags = AABB_FLAGS_NONE;
	}
	
	// Fill in the top of tree nodes
	for (int i = NUM_CELLS-2; i >= 0; i--)
	{
		nodes[i].min   = min(nodes[i*2+1].min, nodes[i*2+2].min);
		nodes[i].max   = max(nodes[i*2+1].max, nodes[i*2+2].max);
		nodes[i].child = i*2+1;
		nodes[i].count = 2;
		nodes[i].flags = AABB_FLAGS_NONE;
	}
	
	hierarchy.node_base  = nodes;
	hierarchy.root_index = 0;
	hierarchy.tri_base = &primitives[0];
			
	return hierarchy;
}

Hierarchy Builder::build_hierarchy_grid_m128(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	
	std::vector<AABBm >   aabbs;
	std::vector<__m128>   centres;
	std::vector<Binm>     bins[NUM_THREADS];
	std::vector<uint32_t> triangleIds;
	unsigned              thread_begin[NUM_CELLS+1];
	AABBm p_aabb;
	AABBm c_aabb;
	Binm grid_bins[GRID_DIM][GRID_DIM][GRID_DIM];
	
	Node1* nodes = new Node1[primitives.size()*2+NUM_CELLS*2];
	
	for (auto &bin : bins)
		bin = std::vector<Binm>(NUM_BINS);
	
	// Initialise the triangle ids, primitive aabbs and centres, overall aabb and overall centres aabb
	setup_m128_red(primitives, triangleIds, aabbs, centres, p_aabb, c_aabb);
	
	bin_centroids_to_grid_m128(aabbs, centres, c_aabb, triangleIds, grid_bins, thread_begin);

	unsigned offset = NUM_CELLS*2;
	
	#pragma omp parallel for shared(aabbs, centres, triangleIds)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		unsigned thread_count = thread_begin[i+1] - thread_begin[i];
		std::vector<unsigned> scratch1(thread_count);
		std::vector<unsigned> scratch2(thread_count);
		
		std::pair<unsigned, unsigned> child = recurse_m128(aabbs, centres, bins, 
			triangleIds.begin() + thread_begin[i], 
			triangleIds.begin() + thread_begin[i+1],
			scratch1,
			scratch2,
			grid_bins[kz][ky][kx].p_aabb,
			grid_bins[kz][ky][kx].c_aabb,
			nodes,
			offset + thread_begin[i]*2);
			
		nodes[i+NUM_CELLS-1].min   = vec3f1(grid_bins[kz][ky][kx].p_aabb.min[0], grid_bins[kz][ky][kx].p_aabb.min[1], grid_bins[kz][ky][kx].p_aabb.min[2]);
		nodes[i+NUM_CELLS-1].max   = vec3f1(grid_bins[kz][ky][kx].p_aabb.max[0], grid_bins[kz][ky][kx].p_aabb.max[1], grid_bins[kz][ky][kx].p_aabb.max[2]);
		nodes[i+NUM_CELLS-1].child = child.first;
		nodes[i+NUM_CELLS-1].count = child.second;
		nodes[i+NUM_CELLS-1].flags = AABB_FLAGS_NONE;
	}
	
	// Fill in the top of tree nodes
	for (int i = NUM_CELLS-2; i >= 0; i--)
	{
		nodes[i].min   = min(nodes[i*2+1].min, nodes[i*2+2].min);
		nodes[i].max   = max(nodes[i*2+1].max, nodes[i*2+2].max);
		nodes[i].child = i*2+1;
		nodes[i].count = 2;
		nodes[i].flags = AABB_FLAGS_NONE;
	}
	
	hierarchy.node_base  = nodes;
	hierarchy.root_index = 0;
			
	return hierarchy;
}

Hierarchy Builder::build_hierarchy_grid(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	
	std::vector<AABB1>    aabbs;
	std::vector<float>    centres;
	std::vector<Bin>      bins[NUM_THREADS];
	std::vector<uint32_t> triangleIds;
	unsigned              thread_begin[NUM_CELLS+1];
	AABB1 p_aabb;
	AABB1 c_aabb;
	Bin  grid_bins[GRID_DIM][GRID_DIM][GRID_DIM];
	
	Node1* nodes = new Node1[primitives.size()*2+NUM_CELLS*2];
	
	for (auto &bin : bins)
		bin = std::vector<Bin>(NUM_BINS);
	
	// Initialise the triangle ids, primitive aabbs and centres, overall aabb and overall centres aabb
	setup(primitives, triangleIds, aabbs, centres, p_aabb, c_aabb);
	
	bin_centroids_to_grid(aabbs, centres, c_aabb, triangleIds, grid_bins, thread_begin);

	unsigned offset = NUM_CELLS*2;
	
	#pragma omp parallel for shared(aabbs, centres, triangleIds)
	for (unsigned i = 0; i < NUM_CELLS; i++)
	{
		unsigned kx = Compact1By2(i>>0);
		unsigned ky = Compact1By2(i>>1);
		unsigned kz = Compact1By2(i>>2);
		
		std::pair<unsigned, unsigned> child = recurse(aabbs, centres, bins, 
			triangleIds.begin() + thread_begin[i], 
			triangleIds.begin() + thread_begin[i+1],
			grid_bins[kz][ky][kx].p_aabb,
			grid_bins[kz][ky][kx].c_aabb,
			nodes,
			offset + thread_begin[i]*2);
			
		nodes[i+NUM_CELLS-1].min   = p_aabb.min;
		nodes[i+NUM_CELLS-1].max   = p_aabb.max;
		nodes[i+NUM_CELLS-1].child = child.first;
		nodes[i+NUM_CELLS-1].count = child.second;
		nodes[i+NUM_CELLS-1].flags = AABB_FLAGS_NONE;
	}
	
	// Fill in the top of tree nodes
	for (int i = NUM_CELLS-2; i >= 0; i--)
	{
		nodes[i].min   = min(nodes[i*2+1].min, nodes[i*2+2].min);
		nodes[i].max   = max(nodes[i*2+1].max, nodes[i*2+2].max);
		nodes[i].child = i*2+1;
		nodes[i].count = 2;
		nodes[i].flags = AABB_FLAGS_NONE;
	}
	
	hierarchy.node_base  = nodes;
	hierarchy.root_index = 0;
			
	return hierarchy;
}

typedef struct alignas(8) RadixInput32 {
	uint32_t code;
	unsigned index;
	
	__forceinline operator uint32_t() const { return code; }
	__forceinline bool operator<(const RadixInput32 &m) const { return code < m.code; }
} RadixInput32;

void setup_morton(std::vector<Triangle> &primitives, std::vector<RadixInput32> &morton_codes, AABB1 &scene_aabb)
{
	scene_aabb.reset();
	
	for (unsigned i = 0; i < primitives.size(); i++) {
		auto &p = primitives[i];
		morton_codes[i].index = i;
		AABB1 p_aabb;
		p_aabb.min = min(min(p.v0, p.v1), p.v2);
		p_aabb.max = max(max(p.v0, p.v1), p.v2);
		scene_aabb.grow(p_aabb);
	}
	
	#pragma omp parallel for shared(primitives, morton_codes) schedule(static, 1024) num_threads(8)
	for (unsigned i = 0; i < primitives.size(); i++) {
		//int tid = omp_get_thread_num();
		auto &p = primitives[i];
		vec3f1 bb_min = min(min(p.v0, p.v1), p.v2);
		vec3f1 bb_max = max(max(p.v0, p.v1), p.v2);
		vec3f1 centre = (bb_min + bb_max) * 0.5f;
		
		centre = (centre - scene_aabb.min) / (scene_aabb.max - scene_aabb.min);
		
		uint32_t x = centre.x * 0x3FF;
		uint32_t y = centre.y * 0x3FF;
		uint32_t z = centre.z * 0x3FF;
		
		assert(x <= 0x3FF);
		assert(y <= 0x3FF);
		assert(z <= 0x3FF);
		
		morton_codes[i].code = bitInterleave32(x, y, z);
	}
}

typedef struct alignas(16) RadixInput64 {
	uint64_t code;
	unsigned index;
	
	__forceinline operator uint64_t() const { return code; }
	__forceinline bool operator<(const RadixInput64 &m) const { return code < m.code; }
} RadixInput64;

void setup_morton2(std::vector<Triangle> &primitives, std::vector<RadixInput64> &morton_codes, AABB1 &scene_aabb)
{
	scene_aabb.reset();
	
	for (unsigned i = 0; i < primitives.size(); i++) {
		auto &p = primitives[i];
		morton_codes[i].index = i;
		AABB1 p_aabb;
		p_aabb.min = min(min(p.v0, p.v1), p.v2);
		p_aabb.max = max(max(p.v0, p.v1), p.v2);
		scene_aabb.grow(p_aabb);
	}
	
	#pragma omp parallel for shared(primitives, morton_codes) schedule(static, 1024) num_threads(8)
	for (unsigned i = 0; i < primitives.size(); i++) {
		auto &p = primitives[i];
		vec3f1 bb_min = min(min(p.v0, p.v1), p.v2);
		vec3f1 bb_max = max(max(p.v0, p.v1), p.v2);
		vec3f1 centre = (bb_min + bb_max) * 0.5f;
		
		centre = (centre - scene_aabb.min) / (scene_aabb.max - scene_aabb.min);
		
		uint64_t x = centre.x * 0xFFFFF;
		uint64_t y = centre.y * 0xFFFFF;
		uint64_t z = centre.z * 0xFFFFF;
		
		assert(x <= 0xFFFFF);
		assert(y <= 0xFFFFF);
		assert(z <= 0xFFFFF);
		
		morton_codes[i].code = bitInterleave64(x, y, z);
	}
}

unsigned find_split_morton(std::vector<RadixInput32>::iterator begin, std::vector<RadixInput32>::iterator end)
{
	assert(begin < end);
	
	uint32_t firstCode = (*begin).code;
	uint32_t lastCode  = (*(end-1)).code;
	
	if (firstCode == lastCode)
        return ((end - begin) >> 1);
	
	int commonPrefix = _lzcnt_u32(firstCode ^ lastCode);

	int split = 0; // initial guess
	int last  = end-begin-1;
    int step  = last;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position
		
        if (newSplit < last)
        {
            uint32_t splitCode = (*(begin+newSplit)).code;
            int splitPrefix = _lzcnt_u32(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix) {
                split = newSplit; // accept proposal
			}
        }
    }
    while (step > 1);
	
	return split;
}

unsigned find_split_morton2(std::vector<RadixInput64>::iterator begin, std::vector<RadixInput64>::iterator end)
{
	assert(begin < end);
	
	uint64_t firstCode = (*begin).code;
	uint64_t lastCode  = (*(end-1)).code;
	
	if (firstCode == lastCode)
        return ((end - begin) >> 1);
	
	int commonPrefix = _lzcnt_u64(firstCode ^ lastCode);

	int split = 0; // initial guess
	int last  = end-begin-1;
    int step  = last;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position
		
        if (newSplit < last)
        {
            uint64_t splitCode = (*(begin+newSplit)).code;
            int splitPrefix = _lzcnt_u64(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix) {
                split = newSplit; // accept proposal
			}
        }
    }
    while (step > 1);
	
	return split;
}

unsigned recurse_morton(std::vector<RadixInput32>::iterator begin, std::vector<RadixInput32>::iterator end, Node1* nodes, unsigned free_index)
{
	unsigned count = end-begin;
	
	if (count < 8) {
		// Create leaves
		for (unsigned i = 0; i < count; i++, free_index++) {
			Node1 &leaf = nodes[free_index];
			leaf.child = (*(begin+i)).index;
			leaf.count = 1;
			leaf.flags = AABB_FLAGS_LEAF;
		}
		
		return count;
	}
	
	unsigned split = find_split_morton(begin, end);
	
	unsigned left_count  = split+1;
	unsigned right_count = count-left_count;
	unsigned new_nodes   = (left_count != 1) + (right_count != 1);
	unsigned right_free_index = free_index + new_nodes + (left_count-1)*2 + (left_count==1);
	
	// Todo handle 1 node split
	unsigned left_child_count;
	unsigned right_child_count;

	if (left_count > 512) {
		#pragma omp task shared(nodes, left_child_count)
		left_child_count = recurse_morton(begin, begin+split+1, nodes, free_index+new_nodes);
	} else {
		left_child_count = recurse_morton(begin, begin+split+1, nodes, free_index+new_nodes);
	}

	if (right_count > 512) {
		#pragma omp task shared(nodes, right_child_count)
		right_child_count = recurse_morton(begin+split+1, end, nodes, right_free_index);
	} else {
		right_child_count = recurse_morton(begin+split+1, end, nodes, right_free_index);
	}
	
	#pragma omp taskwait
	
	if (left_count != 1) {
		Node1 &left_node = nodes[free_index];
		left_node.child = free_index+new_nodes;
		left_node.count = left_child_count;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1) {
		Node1 &right_node = nodes[free_index + (left_count != 1)];
		right_node.child = right_free_index;
		right_node.count = right_child_count;
		right_node.flags = AABB_FLAGS_NONE;
	}
	
	return 2;
}

unsigned leaves_created = 0;

unsigned recurse_morton2(std::vector<RadixInput64>::iterator begin, std::vector<RadixInput64>::iterator end, Node1* nodes, unsigned free_index)
{
	unsigned count = end-begin;
	
	if (count < 8) {
		// Create leaves
		for (unsigned i = 0; i < count; i++, free_index++) {
			Node1 &leaf = nodes[free_index];
			leaf.child = (*(begin+i)).index;
			leaf.count = 1;
			leaf.flags = AABB_FLAGS_LEAF;
			leaves_created++;
		}
		
		return count;
	}
	
	unsigned split = find_split_morton2(begin, end);
	
	unsigned left_count  = split+1;
	unsigned right_count = count-left_count;
	unsigned new_nodes   = (left_count != 1) + (right_count != 1);
	unsigned right_free_index = free_index + new_nodes + (left_count-1)*2 + (left_count==1);
	
	// Todo handle 1 node split
	unsigned left_child_count;
	unsigned right_child_count;
	
	if (left_count > 512) {
		#pragma omp task shared(nodes, left_child_count)
		left_child_count = recurse_morton2(begin, begin+split+1, nodes, free_index+new_nodes);
	} else {
		left_child_count = recurse_morton2(begin, begin+split+1, nodes, free_index+new_nodes);
	}

	if (right_count > 512) {
		#pragma omp task shared(nodes, right_child_count)
		right_child_count = recurse_morton2(begin+split+1, end, nodes, right_free_index);
	} else {
		right_child_count = recurse_morton2(begin+split+1, end, nodes, right_free_index);
	}
	
	#pragma omp taskwait
	
	if (left_count != 1) {
		Node1 &left_node = nodes[free_index];
		left_node.child = free_index+new_nodes;
		left_node.count = left_child_count;
		left_node.flags = AABB_FLAGS_NONE;
	}
	
	if (right_count != 1) {
		Node1 &right_node = nodes[free_index + (left_count != 1)];
		right_node.child = right_free_index;
		right_node.count = right_child_count;
		right_node.flags = AABB_FLAGS_NONE;
	}
	
	return 2;
}

Hierarchy Builder::build_hierarchy_morton(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	Node1* nodes = new Node1[primitives.size()*2];
	std::vector<RadixInput32> scratch(primitives.size());
	std::vector<RadixInput32> morton_codes(primitives.size());
	AABB1 scene_aabb;
	
	// Create morton codes
	setup_morton(primitives, morton_codes, scene_aabb);
	
	// Sort indices by morton code
	radixSort<uint32_t, RadixInput32>(morton_codes.begin(), morton_codes.end(), scratch.begin(), scratch.end(), 8);

	// Morton recurse
	unsigned count;
	#pragma omp parallel shared(morton_codes, nodes, count) num_threads(16)
	{
		#pragma omp single
		{
			count = recurse_morton(morton_codes.begin(), morton_codes.end(), nodes, 1);
		}
	}
	
	Node1 &root = nodes[0];
	root.child = 1;
	root.count = count;
	root.flags = AABB_FLAGS_NONE;
	
	hierarchy.node_base  = nodes;
	hierarchy.root_index = 0;
	hierarchy.tri_base = &primitives[0];
	
	return hierarchy;
}

Hierarchy Builder::build_hierarchy_morton2(std::vector<Triangle> &primitives)
{
	Hierarchy hierarchy;
	Node1* nodes = new Node1[primitives.size()*2];
	std::vector<RadixInput64> scratch(primitives.size());
	std::vector<RadixInput64> morton_codes(primitives.size());
	AABB1 scene_aabb;
	
	// Create morton codes
	setup_morton2(primitives, morton_codes, scene_aabb);
	
	// Sort indices by morton code
	radixSort<uint64_t, RadixInput64>(morton_codes.begin(), morton_codes.end(), scratch.begin(), scratch.end(), 8);

	// Morton recurse
	unsigned count;
	#pragma omp parallel shared(morton_codes, nodes, count) num_threads(16)
	{
		#pragma omp single
		{
			count = recurse_morton2(morton_codes.begin(), morton_codes.end(), nodes, 1);
		}
	}
	
	printf("Recurse done\n");
	
	Node1 &root = nodes[0];
	root.child = 1;
	root.count = count;
	root.flags = AABB_FLAGS_NONE;
	
	hierarchy.node_base  = nodes;
	hierarchy.root_index = 0;
	hierarchy.tri_base = &primitives[0];
	
	return hierarchy;
}
