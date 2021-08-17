#include "tracer.h"

uint64_t all_miss_tests = 0;
uint64_t all_miss_culls = 0;
uint64_t all_hit_tests = 0;
uint64_t all_hit_hits = 0;
uint64_t aabb_tests = 0;

void print_traverse_stats()
{
	std::cout << "all_miss_tests: " << all_miss_tests << std::endl;
	std::cout << "all_miss_culls: " << all_miss_culls << std::endl;
	std::cout << "all_hit_tests : " << all_hit_tests << std::endl;
	std::cout << "all_hit_hits  : " << all_hit_hits << std::endl;
	std::cout << "aabb_tests    : " << aabb_tests << std::endl;
}

static uint32_t morton_lut[64] = {
	0x00,0x01,0x04,0x05,0x02,0x03,0x06,0x07,0x08,0x09,0x0C,0x0D,0x0A,0x0B,0x0E,0x0F,
	0x10,0x11,0x14,0x15,0x12,0x13,0x16,0x17,0x18,0x19,0x1C,0x1D,0x1A,0x1B,0x1E,0x1F,
	0x20,0x21,0x24,0x25,0x22,0x23,0x26,0x27,0x28,0x29,0x2C,0x2D,0x2A,0x2B,0x2E,0x2F,
	0x30,0x31,0x34,0x35,0x32,0x33,0x36,0x37,0x38,0x39,0x3C,0x3D,0x3A,0x3B,0x3E,0x3F
};

void traverse(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<1> &ray, HitAttributes<1> &attributes)
{
	for (unsigned i = 0; i < count; i++) {
		Node1 &node = hierarchy.node_base[index+i];

		int hit = intersect_ray_aabb<1>(node, ray, attributes);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (hit) {
			if (is_leaf) {
				for (unsigned j = 0; j < node.count; j++) {
					attributes.hit |= intersect_ray_triangle(hierarchy.tri_base[node.child+j], ray, attributes);
				}
			} else {
				traverse(hierarchy, node.child, node.count, ray, attributes);
			}
		}
	}
}

template<size_t K>
void traverse_packet(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<K> &ray, HitAttributes<K> &attributes)
{
	for (unsigned i = 0; i < count; i++) {
		Node1 &node = hierarchy.node_base[index+i];
		
		int hit = intersect_ray_aabb(node, ray, attributes);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (hit) {
			if (is_leaf) {
				for (unsigned j = 0; j < node.count; j++) {
					attributes.hit |= intersect_ray_triangle(hierarchy.tri_base[node.child+j], ray, attributes);
				}
			} else {
				traverse_packet(hierarchy, node.child, node.count, ray, attributes);
			}
		}
	}
}

template<size_t K>
void traverse_packet_stack(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<K> &ray, HitAttributes<K> &attributes)
{
	struct StackEntry {
		unsigned index;
		unsigned count;
	};
	
	const unsigned max_stack_size = 128;
	StackEntry stack[max_stack_size];
	unsigned stack_index = 1;

	stack[0] = {index, count};
	
    while (true) //pop:
    {
		if (unlikely(stack_index == 0)) break;
		StackEntry entry = stack[--stack_index];
		
		for (unsigned i = 0; i < entry.count; i++) {
			Node1 &node = hierarchy.node_base[entry.index+i];

			int hit = intersect_ray_aabb(node, ray, attributes);
			bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
			
			if (hit) {
				if (is_leaf) {
					for (unsigned j = 0; j < node.count; j++) {
						attributes.hit |= intersect_ray_triangle(hierarchy.tri_base[node.child+j], ray, attributes);
					}
				} else {
					stack[stack_index++] = {node.child, node.count};
					assert(stack_index < max_stack_size);
				}
			}
		}
    }
}

template<size_t K, size_t N>
void traverse_bundle(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<K> &bundle, Ray<N> *rays, HitAttributes<N> *attributes)
{
	for (unsigned i = 0; i < count; i++) {
		Node1 &node = hierarchy.node_base[index+i];

		bool any_hit, all_hit, mid_hit;
		intersect_bundle_aabb(bundle, node, any_hit, mid_hit, all_hit);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (mid_hit) {
			if (is_leaf) {
				// Intersect all the triangles individually
				for (unsigned k = 0; k < K/N; k++) {
					for (unsigned j = 0; j < node.count; j++) {
						attributes[k].hit |= intersect_ray_triangle(hierarchy.tri_base[node.child+j], rays[k], attributes[k]);
					}
				}
			} else {
				// Carry on traversing as a bundle
				traverse_bundle(hierarchy, node.child, node.count, bundle, rays, attributes);
			}
		} else if (any_hit) {
			// Break the bundle up and traverse individual rays
			for (unsigned k = 0; k < K/N; k++) {
				traverse_packet_stack(hierarchy, index+i, 1, rays[k], attributes[k]);
			}
		}
	}
}

template
void traverse_packet<1>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<1> &ray, HitAttributes<1> &attributes);

template
void traverse_packet<2>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<2> &ray, HitAttributes<2> &attributes);

template
void traverse_packet<4>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<4> &ray, HitAttributes<4> &attributes);

template
void traverse_packet<8>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<8> &ray, HitAttributes<8> &attributes);

template
void traverse_packet_stack<1>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<1> &ray, HitAttributes<1> &attributes);

template
void traverse_packet_stack<2>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<2> &ray, HitAttributes<2> &attributes);

template
void traverse_packet_stack<4>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<4> &ray, HitAttributes<4> &attributes);

template
void traverse_packet_stack<8>(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<8> &ray, HitAttributes<8> &attributes);

template
void traverse_bundle<16>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<16> &bundle, Ray<1> *bundle_rays, HitAttributes<1> *attributes);
template
void traverse_bundle<16>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<16> &bundle, Ray<8> *bundle_rays, HitAttributes<8> *attributes);
template
void traverse_bundle<32>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<32> &bundle, Ray<1> *bundle_rays, HitAttributes<1> *attributes);
template
void traverse_bundle<32>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<32> &bundle, Ray<8> *bundle_rays, HitAttributes<8> *attributes);
template
void traverse_bundle<64>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<64> &bundle, Ray<1> *bundle_rays, HitAttributes<1> *attributes);
template
void traverse_bundle<64>(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<64> &bundle, Ray<8> *bundle_rays, HitAttributes<8> *attributes);

uint8_t* trace(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin)
{
	uint8_t* image_buffer = new uint8_t[w*h*3];

	#pragma omp parallel for
	for (uint32_t j = 0; j < h; j++) {
		for (uint32_t i = 0; i < w; i++) {
			Ray<1> ray;
			ray.origin[0][0] = origin.x;
			ray.origin[1][0] = origin.y;
			ray.origin[2][0] = origin.z;

			vec3f1 coord = vec3f1((float)i, (float)j, 1);
			vec3f1 ndc = 2 * ((coord + 0.5f) / vec3f1(w, h, 1)) - 1;
			vec3f1 p = (ndc.x * vec3f1(1, 0, 0)) + (ndc.y * vec3f1(0, 1, 0)) + (1.0f * vec3f1(0, 0, 1));
			ray.direction[0][0] = p.x / p.length();
			ray.direction[1][0] = p.y / p.length();
			ray.direction[2][0] = p.z / p.length();

			ray.tmin = simdf<1>::broadcast(0.0f);
			ray.tmax = simdf<1>::broadcast(10000.0f);

			HitAttributes<1> attributes;
			
			traverse(hierarchy, hierarchy.root_index, 1, ray, attributes);
			if (attributes.hit[0]) {
				memset(&image_buffer[(j*w+i)*3], 128, 3);
			} else {
				memset(&image_buffer[(j*w+i)*3], 0, 3);
			}
		}
	}

	return image_buffer;
}

template<size_t K>
uint8_t* trace_packet(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin)
{
	uint8_t* image_buffer = new uint8_t[w*h*3];
	unsigned kx = floor(sqrt(K));
	unsigned ky = K / kx;

	#pragma omp parallel for
	for (unsigned j = 0; j < h; j+= ky) {
		for (unsigned i = 0; i < w; i+= kx) {
			Ray<K> ray;
			unsigned m = 0;
			for (unsigned my = 0; my < ky; my++) {
				for (unsigned mx = 0; mx < kx; mx++) {
					ray.origin[0][m] = origin.x;
					ray.origin[1][m] = origin.y;
					ray.origin[2][m] = origin.z;

					vec3f1 coord = vec3f1((float)i + mx, (float)j + my, 1);
					vec3f1 ndc = 2 * ((coord + 0.5f) / vec3f1(w, h, 1)) - 1;
					vec3f1 p = (ndc.x * vec3f1(1, 0, 0)) + (ndc.y * vec3f1(0, 1, 0)) + (1.0f * vec3f1(0, 0, 1));
					ray.direction[0][m] = p.x / p.length();
					ray.direction[1][m] = p.y / p.length();
					ray.direction[2][m] = p.z / p.length();

					ray.tmin = simdf<K>::broadcast(0.0f);
					ray.tmax = simdf<K>::broadcast(10000.0f);
					m++;
				}
			}
			HitAttributes<K> attributes;
			
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

	return image_buffer;
}

template
uint8_t* trace_packet<1>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);
template
uint8_t* trace_packet<2>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);
template
uint8_t* trace_packet<4>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);
template
uint8_t* trace_packet<8>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);
template
uint8_t* trace_packet<16>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);

template<size_t K>
uint8_t* trace_bundle(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin)
{
	const int N = 8;
	uint8_t* image_buffer = new uint8_t[w*h*3];
	uint32_t tile_w = sqrt(K);
	uint32_t tile_h = K / tile_w;
	uint32_t M = K / N;

	#pragma omp parallel for
	for (unsigned j = 0; j < h; j+= tile_h) {
		for (unsigned i = 0; i < w; i+= tile_w) {
			RayBundle<K> bundle;
			Ray<N> rays[M];
			
			bundle.dir_min = vec3f<1>(FLT_MAX, FLT_MAX, FLT_MAX);
			bundle.dir_max = vec3f<1>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			bundle.pos_min = vec3f<1>(FLT_MAX, FLT_MAX, FLT_MAX);
			bundle.pos_max = vec3f<1>(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			unsigned k = 0;
			for (unsigned m = 0; m < M; m++)
			for (unsigned n = 0; n < N; n++, k++) {

				unsigned mx = morton_lut[k] % tile_w;
				unsigned my = morton_lut[k] / tile_w;
				
				rays[m].origin[0][n] = origin.x;
				rays[m].origin[1][n] = origin.y;
				rays[m].origin[2][n] = origin.z;

				vec3f1 coord = vec3f1((float)i + mx, (float)j + my, 1);
				vec3f1 ndc = 2 * ((coord + 0.5f) / vec3f1(w, h, 1)) - 1;
				vec3f1 p = (ndc.x * vec3f1(1, 0, 0)) + (ndc.y * vec3f1(0, 1, 0)) + (1.0f * vec3f1(0, 0, 1));
				rays[m].direction[0][n] = p.x / p.length();
				rays[m].direction[1][n] = p.y / p.length();
				rays[m].direction[2][n] = p.z / p.length();

				rays[m].tmin = simdf<N>::broadcast(0.0f);
				rays[m].tmax = simdf<N>::broadcast(10000.0f);
				vec3f<1> direction(rays[m].direction[0][n], rays[m].direction[1][n], rays[m].direction[2][n]);
				vec3f<1> ori(origin.x, origin.y, origin.z);
				
				bundle.ids[m] = m;
				bundle.pos_min = min(bundle.pos_min, ori);
				bundle.pos_max = max(bundle.pos_max, ori);
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

	return image_buffer;
}

template
uint8_t* trace_bundle<16>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);
template
uint8_t* trace_bundle<64>(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);


