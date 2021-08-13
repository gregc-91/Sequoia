
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
uint32_t kNumThreads = 16;
bool benchmark_mode = false;

#define MAX_LENGTH 256
#define WHITESPACE " \t\n"

enum class BuilderType {
	recursive,
	horizontal,
	grid,
	grid_sse,
	grid_avx,
	morton
};

enum class TraverserType {
	basic,
	packet,
	bundle
};

struct Arguments {
	BuilderType builder_type;
	TraverserType traverser_type;
	uint32_t packet_width;
	uint32_t bundle_width;
};

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

	omp_set_num_threads(64);
	
	t = clock(); 
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
	printf("Traverse time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	stbi_write_bmp("out1.bmp", w, h, 3, image_buffer);
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

	omp_set_num_threads(64);
	
	t = clock(); 
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
	printf("Traverse time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	stbi_write_bmp("out.bmp", w, h, 3, image_buffer);

	print_traverse_stats();
}

void benchmark()
{
	clock_t t[10];
	uint32_t i = 0;
		
	t[i++] = clock(); 	
	std::vector<Triangle> primitives = loadOBJFromFile(std::string("c:/users/greg/documents/sponza.obj"));
    t[i++] = clock(); 

	Builder::build_hierarchy(primitives);
	t[i++] = clock(); 
	Builder::build_hierarchy_horizontal(primitives);
	t[i++] = clock(); 
	Builder::build_hierarchy_grid(primitives);
	t[i++] = clock(); 
	Builder::build_hierarchy_grid_sse(primitives);
	t[i++] = clock(); 
	Builder::build_hierarchy_grid_m128(primitives);
	t[i++] = clock(); 
	Builder::build_hierarchy_morton(primitives);
	t[i++] = clock(); 
	
	printf("Load time: %f\n", (t[1] - t[0]) / (double) CLOCKS_PER_SEC);
	printf("\n");
	printf("            recursive horizontal     grid   grid_sse  grid_avx   morton\n");
	printf("build time:     % 3.0fms      % 3.0fms    % 3.0fms      % 3.0fms     % 3.0fms    % 3.0fms\n",
		1000.0f * (t[2] - t[1]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[3] - t[2]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[4] - t[3]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[5] - t[4]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[6] - t[5]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[7] - t[6]) / (double) CLOCKS_PER_SEC
	);
}

void usage()
{
	printf("Usage:\n");
	printf("    sequoia <scene.obj> [<args>]\n");
	printf("        args:\n");
	printf("            -builder <type> : Specify the builder to use. One of the following:\n");
	printf("                recursive      : A topdown recursive builder that uses openmp tasks to parallelise each split task\n");
	printf("                horizontal     : A hybrid builder parallelising across primitives near the root and switching to recursive\n");
	printf("                grid           : A hybrid builder binning triangles to a grid and launching a build for each cell\n");
	printf("                grid_sse       : A version of grid that uses SIMD across x,y,z\n");
	printf("                grid_avx       : A version of grid that uses SIMD across 8 input primitives\n");
	printf("                morton         : A builder based on partitioning a morton curve\n");
	printf("            -tracer  <type> : Specify the traverser to use. One of the following:\n");
	printf("                basic          : A single ray traverser\n");
	printf("                packet <n>     : A packet traverser of n rays\n");
	printf("                bundle <n>     : A bundle traverser of n rays\n");
	printf("            -threads <n>    : Maximum number of threads\n");
	printf("            -benchmark      : run a benchmark mode testing all builders and traversers\n");
}

Arguments parse_arguments(int argc, char** argv)
{
	Arguments args;
	args.builder_type = BuilderType::recursive;
	args.traverser_type = TraverserType::basic;
	args.packet_width = 8;

	int i = 1;
	while (i < argc) {
		
		if (std::string(argv[i]).compare("-builder") == 0) {
			assert(++i < argc);
			if (std::string(argv[i]).compare("recursive") == 0) {
				args.builder_type = BuilderType::recursive;
			}
			else if (std::string(argv[i]).compare("horizontal") == 0) {
				args.builder_type = BuilderType::horizontal;
			}
			else if (std::string(argv[i]).compare("grid") == 0) {
				args.builder_type = BuilderType::grid;
			}
			else if (std::string(argv[i]).compare("grid_sse") == 0) {
				args.builder_type = BuilderType::grid_sse;
			}
			else if (std::string(argv[i]).compare("grid_avx") == 0) {
				args.builder_type = BuilderType::grid_avx;
			}
			else if (std::string(argv[i]).compare("morton") == 0) {
				args.builder_type = BuilderType::morton;
			} 
			else {
				printf("Error: invalid builder type\n");
				exit(0);
			}
		}
		else if (std::string(argv[i]).compare("-tracer") == 0) {
			assert(++i < argc);

			if (std::string(argv[i]).compare("basic") == 0) {
				args.traverser_type = TraverserType::basic;
			}
			else if (std::string(argv[i]).compare("packet") == 0) {
				assert(++i < argc);
				args.traverser_type = TraverserType::packet;
				args.packet_width = atoi(argv[i]);
				assert(args.packet_width == 1 || args.packet_width == 2 || args.packet_width == 4 || 
					   args.packet_width == 8 || args.packet_width == 16);
			}
			else if (std::string(argv[i]).compare("bundle") == 0) {
				assert(++i < argc);
				args.traverser_type = TraverserType::bundle;
				args.bundle_width = atoi(argv[i]);
				assert(args.bundle_width == 16 || args.bundle_width == 64);
			}
			else {
				printf("Error: invalid tracer type\n");
				exit(0);
			}
		}
		else if (std::string(argv[i]).compare("-threads") == 0) {
			assert(++i < argc);
			kNumThreads = atoi(argv[i]);
		}
		else if (std::string(argv[i]).compare("-benchmark") == 0) {
			benchmark_mode = true;
		}
		i++;
	}

	return args;
}
	
int main(int argc, char** argv)
{
	Arguments agrs = parse_arguments(argc, argv);

	if (benchmark_mode) {
		benchmark();
		exit(0);
	}

	omp_set_num_threads(64);
	
	rt_test();
	bundle_test();
	exit(0);
}