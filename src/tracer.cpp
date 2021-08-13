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

void traverse(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<1> &ray, HitAttributes<1> &attributes)
{
	for (unsigned i = 0; i < count; i++) {
		Node1 &node = hierarchy.node_base[index+i];
		
		simdi<1> hit = intersect_ray_aabb<1>(node, ray, attributes);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (any(hit)) {
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
		
		simdi<K> hit = intersect_ray_aabb(node, ray, attributes);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (movemask(hit)) {
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
			
			simdi<K> hit = intersect_ray_aabb(node, ray, attributes);
			bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
			
			if (movemask(hit)) {
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

		bool all_hit = intersect_bundle_aabb(bundle, node);
		bool is_leaf =!! (node.flags & AABB_FLAGS_LEAF);
		
		if (all_hit) {
			
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
		} else {
			// Break the bundle up and traverse individual rays
			for (unsigned k = 0; k < K/N; k++) {
				traverse_packet_stack(hierarchy, index+1, 1, rays[k], attributes[k]);
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


