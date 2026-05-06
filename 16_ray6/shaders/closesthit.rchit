#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

struct RayPayload {
	vec3 albedo;
	float t;
	vec3 normal;
	float emission;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

hitAttributeEXT vec2 attribs;

layout(binding = 4, set = 0) buffer Vertices { vec4 v[]; } vertices;
layout(binding = 5, set = 0) buffer Indices { uint i[]; } indices;

// Vertex layout: 3 vec4 = 12 floats per vertex.
//   d0 = (pos.xyz, normal.x)
//   d1 = (normal.yz, uv.xy)
//   d2 = (albedo.rgb, emission_strength)
struct Vertex {
	vec3 pos;
	vec3 normal;
	vec3 albedo;
	float emission;
};

Vertex unpack(uint index)
{
	vec4 d0 = vertices.v[3 * index + 0];
	vec4 d1 = vertices.v[3 * index + 1];
	vec4 d2 = vertices.v[3 * index + 2];

	Vertex v;
	v.pos = d0.xyz;
	v.normal = vec3(d0.w, d1.x, d1.y);
	v.albedo = d2.xyz;
	v.emission = d2.w;
	return v;
}

void main()
{
	ivec3 idx = ivec3(
		indices.i[3 * gl_PrimitiveID + 0],
		indices.i[3 * gl_PrimitiveID + 1],
		indices.i[3 * gl_PrimitiveID + 2]);

	Vertex v0 = unpack(idx.x);
	Vertex v1 = unpack(idx.y);
	Vertex v2 = unpack(idx.z);

	vec3 bc = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * bc.x + v1.normal * bc.y + v2.normal * bc.z);

	payload.albedo = v0.albedo;
	payload.emission = v0.emission;
	payload.normal = normal;
	payload.t = gl_RayTmaxEXT;
}
