#ifndef __TRACER_H__
#define __TRACER_H__

#include "aabb.h"
#include "ray.h"

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

void print_traverse_stats();

template <size_t K>
static int intersect_ray_aabb(Node1 &aabb, Ray<K> &ray, HitAttributes<K> &attributes) {
	
	vec3f<K> inv_dir = rcp(ray.direction);
	vec3f<K> t1, t2;
	vec3f<K> q1, q2;
	
	t1.x = simdf<K>((simdf<K>::broadcast(aabb.min.x)-ray.origin.x) * inv_dir.x);
	t2.x = simdf<K>((simdf<K>::broadcast(aabb.max.x)-ray.origin.x) * inv_dir.x);
	t1.y = simdf<K>((simdf<K>::broadcast(aabb.min.y)-ray.origin.y) * inv_dir.y);
	t2.y = simdf<K>((simdf<K>::broadcast(aabb.max.y)-ray.origin.y) * inv_dir.y);
	t1.z = simdf<K>((simdf<K>::broadcast(aabb.min.z)-ray.origin.z) * inv_dir.z);
	t2.z = simdf<K>((simdf<K>::broadcast(aabb.max.z)-ray.origin.z) * inv_dir.z);
	
	vec3f<K> tmin = min(t1, t2);
	vec3f<K> tmax = max(t1, t2);
	
	simdf<K> front = hmax(tmin);
	simdf<K> back  = hmin(tmax);

	return movemask(simdi<K>((back >= front) & (back <= ray.tmax)));
}

// Assumes the bundle shares an origin
template <size_t K>
static bool intersect_bundle_aabb(RayBundle<K> &bundle, Node1 &aabb, bool &any_hit, bool &mid_hit, bool& all_hit) {
	
	simdf<4> dir_min = simdf<4>::loadu(&bundle.dir_min.x);
	simdf<4> dir_max = simdf<4>::loadu(&bundle.dir_max.x);
	simdf<4> inv_min_dir = rcp(dir_min);
	simdf<4> inv_max_dir = rcp(dir_max);
	simdf<4> inv_mid_dir = rcp((dir_min + dir_max) * simdf<4>::broadcast(0.5f));
	simdf<4> aabb_min = simdf<4>::loadu(&aabb.min.x);
	simdf<4> aabb_max = simdf<4>::loadu(&aabb.max.x);
	simdf<4> pos = simdf<4>::loadu(&bundle.pos_min.x);
	simdf<4> t1a = (aabb_min - pos) * inv_min_dir;
	simdf<4> t1b = (aabb_min - pos) * inv_max_dir;
	simdf<4> t2a = (aabb_max - pos) * inv_min_dir;
	simdf<4> t2b = (aabb_max - pos) * inv_max_dir;
	simdf<4> t1 = (aabb_min - pos) * inv_mid_dir;
	simdf<4> t2 = (aabb_max - pos) * inv_mid_dir;
	simdf<4> t1min = min(t1a, t1b);
	simdf<4> t1max = max(t1a, t1b);
	simdf<4> t2min = min(t2a, t2b);
	simdf<4> t2max = max(t2a, t2b);
	simdf<4> tmin_mid = min(t1, t2);
	simdf<4> tmin_any = min(t1min, t2min);
	simdf<4> tmin_all = min(t1max, t2max);
	simdf<4> tmax_mid = max(t1, t2);
	simdf<4> tmax_any = max(t1max, t2max);
	simdf<4> tmax_all = max(t1min, t2min);

	float front_mid = std::max(std::max(tmin_mid[0], tmin_mid[1]), tmin_mid[2]);
	float front_any = std::max(std::max(tmin_any[0], tmin_any[1]), tmin_any[2]);
	float front_all = std::max(std::max(tmin_all[0], tmin_all[1]), tmin_all[2]);
	float back_mid = std::min(std::min(tmax_mid[0], tmax_mid[1]), tmax_mid[2]);
	float back_any = std::min(std::min(tmax_any[0], tmax_any[1]), tmax_any[2]);
	float back_all = std::min(std::min(tmax_all[0], tmax_all[1]), tmax_all[2]);

	mid_hit = back_mid >= front_mid;
	all_hit = back_all >= front_all;
	any_hit = back_any >= front_any;
	return any_hit;
}

template <size_t K>
static simdi<K> intersect_ray_triangle(Triangle &tri, Ray<K> &ray, HitAttributes<K> &attributes)
{
	simdi<K> miss_mask = simdi<K>::broadcast(0);
    const float EPSILON = 0.000000001f;
	simdf<K> epsilon = simdf<K>::broadcast(EPSILON);
	
    vec3f<K> v0(tri.v0.x, tri.v0.y, tri.v0.z);
    vec3f<K> v1(tri.v1.x, tri.v1.y, tri.v1.z); 
    vec3f<K> v2(tri.v2.x, tri.v2.y, tri.v2.z);
    vec3f<K> edge1, edge2, h, s, q;
    simdf<K> a,f,u,v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = ray.direction.cross(edge2);
    a = edge1.dot(h);
	
    miss_mask |= ((a > -epsilon) & (a < epsilon)); // This ray is parallel to this triangle. 
	
    f = 1.0f/a;
    s = ray.origin - v0;
    u = f * s.dot(h);
    miss_mask |= ((u < 0.0f) | (u > 1.0f));
	
    q = s.cross(edge1);
    v = f * ray.direction.dot(q);
    miss_mask |= ((v < 0.0f) | ((u + v) > 1.0f));

    // At this stage we can compute t to find out where the intersection point is on the line.
    simdf<K> t = f * edge2.dot(q);
	
	//simdi<K> m = !miss_mask;
	
	return (!miss_mask) & (t > epsilon);
}

void traverse(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<1> &ray, HitAttributes<1> &attributes);

template<size_t K>
void traverse_packet(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<K> &ray, HitAttributes<K> &attributes);

template<size_t K>
void traverse_packet_stack(Hierarchy &hierarchy, uint32_t index, uint32_t count, Ray<K> &ray, HitAttributes<K> &attributes);

template<size_t K, size_t N>
void traverse_bundle(Hierarchy &hierarchy, uint32_t index, uint32_t count, RayBundle<K> &bundle, Ray<N> *bundle_rays, HitAttributes<N> *attributes);

uint8_t* trace(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);

template<size_t K>
uint8_t* trace_packet(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);

template<size_t K>
uint8_t* trace_bundle(Hierarchy &hierarchy, uint32_t w, uint32_t h, vec3f1 origin);

#endif