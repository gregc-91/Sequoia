#ifndef __TRACER_H__
#define __TRACER_H__

#include "aabb.h"
#include "ray.h"

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

void print_traverse_stats();

template <size_t K>
static simdi<K> intersect_ray_aabb(Node1 &aabb, Ray<K> &ray, HitAttributes<K> &attributes) {
	
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
	
	return simdi<K>((back >= front) & (back <= ray.tmax));
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

#endif