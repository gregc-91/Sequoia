
#include "simd_generic.h"
#include "vec3.h"
#include "ray.h"
#include "aabb.h"
#include "builder.h"
#include "tracer.h"
#include "algorithms.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include <stdlib.h>

#include <omp.h>

#include <immintrin.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int K = 8;

#define MAX_LENGTH 256
#define WHITESPACE " \t\n"

//
// Faster than builtin strtok but maybe less robust
//
char strtok_buf[256];
char* gstrtok(char** s, const char* delims)
{
	char* begin = *s;
	for (; **s != '\0'; (*s)++)
	{
		for (const char* d = delims; *d != '\0'; ++d)
		{
			if (**s == *d)
			{
				memcpy(strtok_buf, begin, *s-begin);
				strtok_buf[*s-begin] = '\0';
				(*s)++;
				return strtok_buf;
			}
		}
	}

	return NULL;
}

std::vector<Triangle> loadOBJFromFile(std::string filename)
{
	FILE *fp;
	char* line = (char*)malloc(MAX_LENGTH);
	char* token = NULL;
	float f[4];
	char* a[4];
	int idx[4];

	std::vector<vec3f1 > vertex_buffer;
	std::vector<unsigned> index_buffer;
	
	index_buffer.reserve(200000);
	vertex_buffer.reserve(200000);

	fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open OBJ file %s!\n", filename.c_str());
		exit(1);
	}

	while (fgets(line, MAX_LENGTH, fp)) {
		char* line_copy = line;
		token = gstrtok(&line_copy, " \t\n\r");
		
		if (token == NULL || strcmp(token, "#") == 0)
			continue;

		else if (strcmp(token, "v") == 0) {
			f[0] = (float)atof(gstrtok(&line_copy, WHITESPACE));
			f[1] = (float)atof(gstrtok(&line_copy, WHITESPACE));
			f[2] = (float)atof(gstrtok(&line_copy, WHITESPACE));

			vertex_buffer.push_back(vec3f1(f[0], f[1], f[2]));
		}
		else if (strcmp(token, "f") == 0) {
			for (int i = 0; i < 3; i++) {
				a[i] = gstrtok(&line_copy, WHITESPACE);
				idx[i] = atoi(a[i]);
				idx[i] += idx[i] < 0 ? (unsigned)vertex_buffer.size() : -1;
			}
			index_buffer.push_back(idx[0]);
			index_buffer.push_back(idx[1]);
			index_buffer.push_back(idx[2]);

			if ((a[3] = gstrtok(&line_copy, WHITESPACE)) != NULL) {
				idx[1] = idx[2];
				idx[2] = atoi(a[3]);
				idx[2] += idx[2] < 0 ? (unsigned)vertex_buffer.size() : -1;

				index_buffer.push_back(idx[0]);
				index_buffer.push_back(idx[1]);
				index_buffer.push_back(idx[2]);
			}
		}
	}
	printf("Geometry\n");
	printf("  faces:        %d\n", (unsigned)index_buffer.size()/3);
	printf("  verts:        %d\n", (unsigned)vertex_buffer.size());
	fclose(fp);

	std::vector<Triangle> triangles;
	triangles.reserve(index_buffer.size() / 3);
	for (unsigned i = 0; i < index_buffer.size(); i += 3)
	{
		triangles.push_back(
			Triangle(vertex_buffer[index_buffer[i + 0]],
				vertex_buffer[index_buffer[i + 1]],
				vertex_buffer[index_buffer[i + 2]]));
	}
	return triangles;
}

simdi<K> intersect_ray_tri(Hierarchy &hierarchy, Ray<K> &ray, unsigned index)
{
	PluckerTriangle &tri = hierarchy.plucker_tri_base[index];
	
	// Load precomputed tri params and replicate to simd
	const simdf<K> nu  = simdf<K>::broadcast(tri.nu);
	const simdf<K> np  = simdf<K>::broadcast(tri.np);
	const simdf<K> nv  = simdf<K>::broadcast(tri.nv);
	const simdf<K> pu  = simdf<K>::broadcast(tri.pu);
	const simdf<K> pv  = simdf<K>::broadcast(tri.pv);
	const simdf<K> e0u = simdf<K>::broadcast(tri.e0u);
	const simdf<K> e0v = simdf<K>::broadcast(tri.e0v);
	const simdf<K> e1u = simdf<K>::broadcast(tri.e1u);
	const simdf<K> e1v = simdf<K>::broadcast(tri.e1v);
	
	//indices computed from from ‘ci’ field of TriAccel: 
	const int w = tri.ci;
	const int u = w == 0 ? 1 : 0;
	const int v = w == 2 ? 1 : 2;
	
	//temporary variables 
	simdf<K> det, dett, detu, detv, du, dv, ou, ov;
	simdi<K> tmpdet0, tmpdet1; 
	 
	/* ----ray-packet/triangle hit test ---- */ 
	//dett = np -(ou*nu+ov*nv+ow) 
	dett = np - ray.origin[w];  
	ou   = pu - ray.origin[u];
	ov   = pv - ray.origin[v];
	dett = dett + nu * ou; 
	dett = dett + nv * ov;
	
	//det =du*nu+dv*nv+dw 
	du = ray.direction[u]; 
	dv = ray.direction[v];
	det = nu * du + nv * dv + ray.direction[w]; 
	
	//Du = du*dett - (pu-ou)*det  
	du  = du - ou * det;
	
	//Dv = dv*dett -  (pv-ov)*det 
	dv  = dv - ov * det; 
	
	//detu = (e1vDu – e1u*Dv) 
	detu = e1v*du - e1u*dv; 
	
	//detv = (e0uDv – e0v*Du) 
	detv = e0u*dv - e0v*du; 
	
	/* Having det, detu and detv values in hands we can then 
	compute the mask indicating whether each of 4 values ‘det - detu 
	– detv’, ‘detu’ and ‘detv’ all have the same sign indicating that 
	corresponding rays in packet hit the triangle (see section 4.2.1)*/
	tmpdet0 = (simdi<K>)(det - detu - detv);
	tmpdet0 = tmpdet0 ^ (simdi<K>)detu;
	tmpdet1 = (simdi<K>)detv ^ (simdi<K>)detu;
	tmpdet0 = !(tmpdet0 | tmpdet1);
	
	return tmpdet0;
}

simd<1, int> argmax_test(vec3f<1> a) {
	return argmax(a);
}

bool intersect(float* aabb, float* origin, float* direction) {
    float tx1 = (aabb[0] - origin[0])/direction[0];
    float tx2 = (aabb[3] - origin[0])/direction[0];
    float ty1 = (aabb[1] - origin[1])/direction[1];
    float ty2 = (aabb[4] - origin[1])/direction[1];
	float tz1 = (aabb[2] - origin[2])/direction[2];
    float tz2 = (aabb[5] - origin[2])/direction[2];
	
	float front = std::max(std::max(std::min(tx1, tx2), std::min(ty1, ty2)), std::min(tz1, tz2));
	float back  = std::min(std::min(std::max(tx1, tx2), std::max(ty1, ty2)), std::max(tz1, tz2));
 
    return back >= front;
}

simd<K, int> intersect_ray_aabb(simdf<K>* aabb, simdf<K>* origin, simdf<K>* direction) {
    simdf<K> tx1 = (aabb[0] - origin[0])/direction[0];
    simdf<K> tx2 = (aabb[3] - origin[0])/direction[0];
    simdf<K> ty1 = (aabb[1] - origin[1])/direction[1];
    simdf<K> ty2 = (aabb[4] - origin[1])/direction[1];
	simdf<K> tz1 = (aabb[2] - origin[2])/direction[2];
    simdf<K> tz2 = (aabb[5] - origin[2])/direction[2];
	
	simdf<K> front = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2));
	simdf<K> back  = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2));
 
    return back >= front;
}

simdi<K> intersect_ray_aabb2(simdf<1>* aabb, simdf<K>* origin, simdf<K>* direction) {
	
	simdf<K> inv_dir_x = rcp(direction[0]);
	simdf<K> inv_dir_y = rcp(direction[1]);
	simdf<K> inv_dir_z = rcp(direction[2]);
	
	simdf<K> oid_x = origin[0] * inv_dir_x;
	simdf<K> oid_y = origin[1] * inv_dir_y;
	simdf<K> oid_z = origin[2] * inv_dir_z;

    simdf<K> tx1 = fmadd(simdf<K>::broadcast(aabb[0]), oid_x, inv_dir_x);
    simdf<K> tx2 = fmadd(simdf<K>::broadcast(aabb[3]), oid_x, inv_dir_x);
    simdf<K> ty1 = fmadd(simdf<K>::broadcast(aabb[1]), oid_y, inv_dir_y);
    simdf<K> ty2 = fmadd(simdf<K>::broadcast(aabb[4]), oid_y, inv_dir_y);
	simdf<K> tz1 = fmadd(simdf<K>::broadcast(aabb[2]), oid_z, inv_dir_z);
    simdf<K> tz2 = fmadd(simdf<K>::broadcast(aabb[5]), oid_z, inv_dir_z);
	
	simdf<K> front = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2));
	simdf<K> back  = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2));
 
    return back >= front;
}

simdi<K> intersect_ray_aabb(Node<1> &aabb, Ray<K> &ray) {
	vec3f<K> t1 = (aabb.min - ray.origin) / ray.direction;
	vec3f<K> t2 = (aabb.max - ray.origin) / ray.direction;
	
	vec3f<K> tmin = min(t1, t2);
	vec3f<K> tmax = max(t1, t2);
	
	simdf<K> front = hmax(tmin);
	simdf<K> back  = hmin(tmax);
	
    return back >= front;
}

simdi<K> intersect_ray_aabb2(Node<1> &aabb, Ray<K> &ray) {
	
	vec3f<K> inv_dir = rcp(ray.direction);
	vec3f<K> oid = ray.origin * inv_dir;
	
	//vec3f<K> t1 = fmadd(vec3f<K>::broadcast(*aabb_min), oid, inv_dir);
	//vec3f<K> t2 = fmadd(vec3f<K>::broadcast(*aabb_max), oid, inv_dir);
	
	vec3f<K> t1, t2;
	
	// Uses fewer xmm registers, is this faster?
	t1.x = fmadd(simdf<K>::broadcast(aabb.min.x), oid.x, inv_dir.x);
    t2.x = fmadd(simdf<K>::broadcast(aabb.max.x), oid.x, inv_dir.x);
    t1.y = fmadd(simdf<K>::broadcast(aabb.min.y), oid.y, inv_dir.y);
    t2.y = fmadd(simdf<K>::broadcast(aabb.max.y), oid.y, inv_dir.y);
	t1.z = fmadd(simdf<K>::broadcast(aabb.min.z), oid.z, inv_dir.z);
    t2.z = fmadd(simdf<K>::broadcast(aabb.max.z), oid.z, inv_dir.z);
	
	vec3f<K> tmin = min(t1, t2);
	vec3f<K> tmax = max(t1, t2);
	
	simdf<K> front = hmax(tmin);
	simdf<K> back  = hmin(tmax);
	
   return back >= front;
}

typedef struct alignas(8) RadixInput {
	unsigned code;
	unsigned index;
	
	__forceinline operator unsigned() const { return code; }
	__forceinline bool operator<(const RadixInput &m) const { return code < m.code; }
} RadixInput;

typedef struct alignas(16) RadixInput64 {
	uint64_t code;
	unsigned index;
	
	__forceinline operator uint64_t() const { return code; }
	__forceinline bool operator<(const RadixInput &m) const { return code < m.code; }
} RadixInput64;

void sort_test()
{
	#define NUM_VALS 1000000
	std::vector<RadixInput> input(NUM_VALS);
	std::vector<RadixInput> scratch(NUM_VALS);
	std::vector<RadixInput64> input2(NUM_VALS);
	std::vector<RadixInput64> scratch2(NUM_VALS);

	for (unsigned i = 0; i < NUM_VALS; i++) {
		input[i].code = rand();
		input[i].index = i;
		
		input2[i].code = rand();
		input2[i].index = i;
	}
	
	std::vector<RadixInput> input3 = input;
	
	clock_t t, t2; 
	
	t = clock(); 
	radixSort<unsigned, RadixInput>(input.begin(), input.end(), scratch.begin(), scratch.end(), (unsigned)8);
	t = clock() - t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC;
	
	t2 = clock();
	std::sort(input3.begin(), input3.end());
	t2 = clock() - t2;
	double time_taken2 = ((double)t2)/CLOCKS_PER_SEC;
	
	t = clock(); 
	radixSort<uint64_t, RadixInput64>(input2.begin(), input2.end(), scratch2.begin(), scratch2.end(), (unsigned)8);
	t = clock() - t;
	double time_taken3 = ((double)t)/CLOCKS_PER_SEC;
	
	for (unsigned i = 0; i < input.size(); i++) {
		assert(input[i] == input3[i]);
	}
	
	printf("radixsort   time %f\n", time_taken);
	printf("radixsort64 time %f\n", time_taken3);
	printf("std::sort   time %f\n", time_taken2);
}

uint32_t count_nodes_r(Hierarchy &hierarchy, uint32_t index)
{
	Node1 &node = hierarchy.node_base[index];
	unsigned count = 1;
	if (!(node.flags & AABB_FLAGS_LEAF)) {
		for (unsigned i = 0; i < node.count; i++) {
			count += count_nodes_r(hierarchy, node.child+i);
		}
	}
	return count;
}

uint32_t count_nodes(Hierarchy &hierarchy)
{
	return count_nodes_r(hierarchy, hierarchy.root_index);
}

void rt_test()
{
	uint32_t w = 512, h = 512;
	//uint32_t tile_w = 16, tile_h = 16;
	uint8_t* image_buffer = new uint8_t[w*h*3];
	//float* depth_buffer = new float[w*h];
	clock_t t; 
	const unsigned k = 8;
	
	printf("Alignment checks\n");
	printf("  simd1f %llu\n", alignof(simd1f));
	printf("  simd4f %llu\n", alignof(simd4f));
	printf("  simd8f %llu\n", alignof(simd8f));
	printf("  vec3f8 %llu\n", alignof(vec3f<8>));
	
	std::vector<Triangle> primitives = loadOBJFromFile(std::string("c:/users/greg/documents/bunny.obj"));
	
	t = clock(); 
	Hierarchy hierarchy = Builder::build_hierarchy_grid_sse(primitives);
	printf("Build time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	t = clock(); 
	printf("Nodes counted %d\n", count_nodes(hierarchy));
	printf("Count time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);

	unsigned kx = floor(sqrt(k));
	unsigned ky = k / kx;
	
	t = clock(); 
	for (unsigned iter = 0; iter < 500; iter++) {
	#pragma omp parallel for
	for (unsigned j = 0; j < h; j+= ky) {
		for (unsigned i = 0; i < w; i+= kx) {
			Ray<k> ray;
			unsigned m = 0;
			for (unsigned my = 0; my < ky; my++) {
				for (unsigned mx = 0; mx < kx; mx++) {
					float x = (float)(i+mx)*0.25/w - 0.125f;
					float y = (float)(j+my)*0.25/h;
					ray.origin[0][m] = x;
					ray.origin[1][m] = y;
					ray.origin[2][m] = -1.0f;
					ray.direction[0][m] = 0.0f;
					ray.direction[1][m] = 0.0f;
					ray.direction[2][m] = 1.0f;
					ray.tmin = simdf<k>::broadcast(0.0f);
					ray.tmax = simdf<k>::broadcast(5.0f);
					m++;
				}
			}
			HitAttributes<k> attributes;
			
			traverse_packet_stack(hierarchy, hierarchy.root_index, 1, ray, attributes);
			m = 0;
			for (unsigned my = 0; my < ky; my++) {
				for (unsigned mx = 0; mx < kx; mx++) {
					if (attributes.hit[m]) {
						memset(&image_buffer[((j+my)*w+i+mx)*3], 128, 3);
					} else {
						memset(&image_buffer[((j+my)*w+i+mx)*3], 0, 3);
					}
					m++;
				}
			}
		}
	}
	}
	printf("Traverse time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	stbi_write_bmp("out.bmp", w, h, 3, image_buffer);
}

uint32_t morton_lut[64] = {
	0x00,0x01,0x04,0x05,0x02,0x03,0x06,0x07,0x08,0x09,0x0C,0x0D,0x0A,0x0B,0x0E,0x0F,
	0x10,0x11,0x14,0x15,0x12,0x13,0x16,0x17,0x18,0x19,0x1C,0x1D,0x1A,0x1B,0x1E,0x1F,
	0x20,0x21,0x24,0x25,0x22,0x23,0x26,0x27,0x28,0x29,0x2C,0x2D,0x2A,0x2B,0x2E,0x2F,
	0x30,0x31,0x34,0x35,0x32,0x33,0x36,0x37,0x38,0x39,0x3C,0x3D,0x3A,0x3B,0x3E,0x3F
};

void bundle_test()
{
	uint32_t w = 512, h = 512;
	uint32_t tile_w = 4, tile_h = 4;
	uint8_t* image_buffer = new uint8_t[w*h*3];
	//float* depth_buffer = new float[w*h];
	clock_t t; 
	const unsigned K = 16;
	const unsigned N = 8;
	const unsigned M = K/N;
	
	printf("Alignment checks\n");
	printf("  simd1f %llu\n", alignof(simd1f));
	printf("  simd4f %llu\n", alignof(simd4f));
	printf("  simd8f %llu\n", alignof(simd8f));
	printf("  vec3f8 %llu\n", alignof(vec3f<8>));
	
	std::vector<Triangle> primitives = loadOBJFromFile(std::string("c:/users/greg/documents/bunny.obj"));
	
	t = clock(); 
	Hierarchy hierarchy = Builder::build_hierarchy_grid_sse(primitives);
	printf("Build time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	t = clock(); 
	printf("Nodes counted %d\n", count_nodes(hierarchy));
	printf("Count time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	t = clock(); 
	for (unsigned iter = 0; iter < 1000; iter++) {
	#pragma omp parallel for
	for (unsigned j = 0; j < h; j+= tile_h) {
		for (unsigned i = 0; i < w; i+= tile_w) {
			RayBundle<K> bundle;
			Ray<N> rays[M];
			
			bundle.dir_min = vec3f<1>(FLT_MAX, FLT_MAX, FLT_MAX);
			bundle.dir_max = vec3f<1>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			
			unsigned k = 0;
			for (unsigned m = 0; m < M; m++)
			for (unsigned n = 0; n < N; n++, k++) {

				unsigned mx = morton_lut[k] % tile_w;
				unsigned my = morton_lut[k] / tile_w;
				
				float x = (float)(i+mx)*0.25/w - 0.125f;
				float y = (float)(j+my)*0.25/h;
				
				vec3f<1> origin(x, y, -1.0f);
				vec3f<1> direction(0.0f, 0.0f, 1.0f);
				
				rays[m].origin[0][n] = origin.x;
				rays[m].origin[1][n] = origin.y;
				rays[m].origin[2][n] = origin.z;
				rays[m].direction[0][n] = direction.x;
				rays[m].direction[1][n] = direction.y;
				rays[m].direction[2][n] = direction.z;
				rays[m].tmin = simdf<N>::broadcast(0.0f);
				rays[m].tmax = simdf<N>::broadcast(5.0f);
				
				bundle.ids[m] = m;
				bundle.pos_min = min(bundle.pos_min, origin);
				bundle.pos_max = max(bundle.pos_max, origin);
				bundle.dir_min = min(bundle.dir_min, direction);
				bundle.dir_max = max(bundle.dir_max, direction);
			}
			HitAttributes<N> attributes[M];
			traverse_bundle(hierarchy, hierarchy.root_index, 1, bundle, rays, attributes);
			
			k = 0;
			for (unsigned m = 0; m < M; m++)
			for (unsigned n = 0; n < N; n++, k++) {
				
				unsigned mx = morton_lut[k] % tile_w;
				unsigned my = morton_lut[k] / tile_w;
				if (attributes[m].hit[n]) {
					memset(&image_buffer[((j+my)*w+i+mx)*3], 128, 3);
				} else {
					memset(&image_buffer[((j+my)*w+i+mx)*3], 0, 3);
				}
			}
		}
	}
	}
	printf("Traverse time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	stbi_write_bmp("out.bmp", w, h, 3, image_buffer);

	print_traverse_stats();
}

void usage()
{
	printf("Usage:\n");
	printf("    sequoia <scene.obj> [<args>]\n");
	printf("        args:\n");
	printf("            builder <type> : Specify the builder to use. One of the following:\n");
	printf("                recursive      : A topdown recursive builder that uses openmp tasks to parallelise each split task\n");
	printf("                horizontal     : A hybrid builder parallelising across primitives near the root and switching to recursive\n");
	printf("                grid           : A hybrid builder binning triangles to a grid and launching a build for each cell\n");
	printf("                grid_sse       : A version of grid that uses SIMD across x,y,z\n");
	printf("                grid_avx       : A version of grid that uses SIMD across 8 input primitives\n");
	printf("                morton         : A builder based on partitioning a morton curve\n");
	printf("            tracer  <type> : Specify the traverser to use. One of the following:\n");
	printf("                basic          : A single ray traverser\n");
	printf("                packet <n>     : A packet traverser of n rays\n");
	printf("                bundle <n>     : A bundle traverser of n rays\n");
	printf("            threads <n>    : Maximum number of threads\n");
}
	
int main(int argc, char** argv)
{
	omp_set_num_threads(1);
	
	//sort_test();
	//rt_test();
	bundle_test();
	exit(0);
	
	clock_t t, t2; 
		
	t = clock(); 
		
	std::vector<Triangle> primitives = loadOBJFromFile(std::string("c:/users/greg/documents/sponza.obj"));
	
	//#define NUM_PRIMS 16
	//std::vector<Triangle> tmp(NUM_PRIMS);
	//std::copy(primitives.begin(), primitives.begin()+NUM_PRIMS, tmp.begin());
	//std::copy(primitives.begin(), primitives.end(), tmp.begin()+primitives.size());
	//primitives = tmp;

    t2 = clock();
	auto omp_t1 = omp_get_wtime();
	
	//Builder::build_hierarchy(primitives);
	//Builder::build_hierarchy_horizontal(primitives);
	//Builder::build_hierarchy_grid(primitives);
	//Builder::build_hierarchy_grid_sse(primitives);
	//Builder::build_hierarchy_grid_m128(primitives);
	//Builder::build_hierarchy_morton(primitives);
#if 1
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
	Builder::build_hierarchy_grid_sse(primitives);
#endif
	
	auto omp_t2 = omp_get_wtime();
	
    t = clock() - t; 
	t2 = clock() - t2;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    double build_time = ((double)t2)/CLOCKS_PER_SEC;
	
	printf("Time taken:  %f\n", time_taken);
	printf("Build time:  %f\n", build_time);
	printf("omp time:    %f\n", omp_t2-omp_t1);
	printf("Triangles/s: %f\n", primitives.size() / build_time);
}