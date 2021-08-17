
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
#include <numeric>
#include <stdlib.h>

#include <omp.h>

#include <immintrin.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int K = 8;
uint32_t kNumThreads = 64;
bool benchmark_mode = false;
std::string filename;

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
	uint32_t w;
	uint32_t h;
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

Hierarchy run_builder(Arguments args, std::vector<Triangle> &primitives)
{
	switch (args.builder_type) {
	case BuilderType::recursive:
		return Builder::build_hierarchy(primitives);
	case BuilderType::horizontal:
		return Builder::build_hierarchy_horizontal(primitives);
	case BuilderType::grid:
		return Builder::build_hierarchy_grid(primitives);
	case BuilderType::grid_sse:
		return Builder::build_hierarchy_grid_sse(primitives);
	case BuilderType::grid_avx:
		return Builder::build_hierarchy_grid_m128(primitives);
	case BuilderType::morton:
		return Builder::build_hierarchy_morton(primitives);
	default:
		assert(0 && "Error: Unknown builder type");
	}
	assert(0);
}

uint8_t* run_tracer(Arguments args, Hierarchy &hierarchy, vec3f1 camera)
{
	switch (args.traverser_type) {
	case TraverserType::basic:
		return trace(hierarchy, args.w, args.h, camera);
	case TraverserType::packet:
		switch (args.packet_width) {
		case 1: return trace_packet<1>(hierarchy, args.w, args.h, camera);
		case 2: return trace_packet<2>(hierarchy, args.w, args.h, camera);
		case 4: return trace_packet<4>(hierarchy, args.w, args.h, camera);
		case 8: return trace_packet<8>(hierarchy, args.w, args.h, camera);
		case 16: return trace_packet<16>(hierarchy, args.w, args.h, camera);
		default: assert(0 && "Error: Invalid packet width");
		}
	case TraverserType::bundle:
		switch (args.bundle_width) {
		case 16: return trace_bundle<16>(hierarchy, args.w, args.h, camera);
		case 64: return trace_bundle<64>(hierarchy, args.w, args.h, camera);
		default: assert(0 && "Error: Invalid bundle size");
		}
	default: assert(0 && "Error: Unknown traverser type");
	}
}

void run(Arguments args)
{
	clock_t t;

	// Load the scene
	std::vector<Triangle> primitives = loadOBJFromFile(filename);

	// Compute the bounding box
	AABB1 scene_aabb = std::reduce(primitives.begin(), primitives.end(), AABB1(FLT_MAX, -FLT_MAX), 
		[](const AABB1 &a, const AABB1 &b)->AABB1 { return a.combine(b); }
	);

	// Compute the camera position
	vec3f1 centre = (scene_aabb.min + scene_aabb.max) * 0.5f;
	vec3f1 camera = vec3f1(centre.x, centre.y, 2*scene_aabb.min.z - centre.z);

	// Build the hierarchy
	t = clock();
	Hierarchy hierarchy = run_builder(args, primitives);
	printf("Build time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);

	// Count the nodes
	t = clock(); 
	printf("Nodes counted %d\n", count_nodes(hierarchy));
	printf("Count time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);

	// Traverse the hierarchy
	t = clock();
	uint8_t* image = run_tracer(args, hierarchy, camera);
	printf("Traverse time: %f\n", ((double)clock() - t)/CLOCKS_PER_SEC);
	
	stbi_write_bmp("render.bmp", args.w, args.h, 3, image);

	print_traverse_stats();
}

void benchmark()
{
	clock_t t[16];
	uint32_t i = 0;
		
	t[i++] = clock(); 	
	std::vector<Triangle> primitives = loadOBJFromFile(filename);
	AABB1 scene_aabb = std::reduce(primitives.begin(), primitives.end(), AABB1(FLT_MAX, -FLT_MAX), 
		[](const AABB1 &a, const AABB1 &b)->AABB1 { return a.combine(b); }
	);
	vec3f1 centre = (scene_aabb.min + scene_aabb.max) * 0.5f;
    t[i++] = clock(); 

	Hierarchy hierarchy = Builder::build_hierarchy(primitives);
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

	uint32_t w = 1920*2;
	uint32_t h = 1080*2;
	uint8_t* image1 = trace(hierarchy, w, h, vec3f1(centre.x, centre.y, 2*scene_aabb.min.z - centre.z));
	t[i++] = clock(); 
	uint8_t* image2 = trace_packet<8>(hierarchy, w, h, vec3f1(centre.x, centre.y, 2*scene_aabb.min.z - centre.z));
	t[i++] = clock(); 
	uint8_t* image3 = trace_bundle<64>(hierarchy, w, h, vec3f1(centre.x, centre.y, 2*scene_aabb.min.z - centre.z));
	t[i++] = clock(); 
	
	printf("Load time: %.2fs\n", (t[1] - t[0]) / (double) CLOCKS_PER_SEC);
	printf("\n");
	printf("             recursive horizontal       grid   grid_sse   grid_avx     morton\n");
	printf("build time: % 8.0fms % 8.0fms % 8.0fms % 8.0fms % 8.0fms % 8.0fms\n",
		1000.0f * (t[2] - t[1]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[3] - t[2]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[4] - t[3]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[5] - t[4]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[6] - t[5]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[7] - t[6]) / (double) CLOCKS_PER_SEC
	);
	printf("tri/s     : % 9.0fm % 9.0fm % 9.0fm % 9.0fm % 9.0fm % 9.0fm\n",
		(float)primitives.size() / (1000000.0f * (t[2] - t[1]) / (double) CLOCKS_PER_SEC),
		(float)primitives.size() / (1000000.0f * (t[3] - t[2]) / (double) CLOCKS_PER_SEC),
		(float)primitives.size() / (1000000.0f * (t[4] - t[3]) / (double) CLOCKS_PER_SEC),
		(float)primitives.size() / (1000000.0f * (t[5] - t[4]) / (double) CLOCKS_PER_SEC),
		(float)primitives.size() / (1000000.0f * (t[6] - t[5]) / (double) CLOCKS_PER_SEC),
		(float)primitives.size() / (1000000.0f * (t[7] - t[6]) / (double) CLOCKS_PER_SEC)
	);
	printf("\n");
	printf("             basic packet bundle\n");
	printf("trace time: % 4.0fms % 4.0fms % 4.0fms\n",
		1000.0f * (t[8] - t[7]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[9] - t[8]) / (double) CLOCKS_PER_SEC,
		1000.0f * (t[10] - t[9]) / (double) CLOCKS_PER_SEC
	);
	printf("mrps      : % 6.1f % 6.1f % 6.1f\n", 
		(w*h/1000000.0f) / ((t[8] - t[7]) / (double) CLOCKS_PER_SEC),
		(w*h/1000000.0f) / ((t[9] - t[8]) / (double) CLOCKS_PER_SEC),
		(w*h/1000000.0f) / ((t[10] - t[9]) / (double) CLOCKS_PER_SEC)
	);

	stbi_write_bmp("benchmark1.bmp", w, h, 3, image1);
	stbi_write_bmp("benchmark2.bmp", w, h, 3, image2);
	stbi_write_bmp("benchmark3.bmp", w, h, 3, image3);

	delete[] image1;
	delete[] image2;
	delete[] image3;
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
	printf("            -res <x> <y>    : Set the output resolution\n");
	printf("            -benchmark      : run a benchmark mode testing all builders and traversers\n");
}

Arguments parse_arguments(int argc, char** argv)
{
	Arguments args;
	args.builder_type = BuilderType::recursive;
	args.traverser_type = TraverserType::basic;
	args.packet_width = 8;
	args.w = 1920;
	args.h = 1080;

	if (argc < 2) {
		printf("Error: not enough arguments\n");
		usage();
		exit(0);
	}
	filename = argv[1];

	int i = 2;
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
		else if (std::string(argv[i]).compare("-res") == 0) {
			assert(++i < argc);
			args.w = atoi(argv[i]);
			assert(++i < argc);
			args.h = atoi(argv[i]);
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
	Arguments args = parse_arguments(argc, argv);

	omp_set_num_threads(kNumThreads);

	if (benchmark_mode) {
		benchmark();
		exit(0);
	}

	run(args);
	exit(0);
}