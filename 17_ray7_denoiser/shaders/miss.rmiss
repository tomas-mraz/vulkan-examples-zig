#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
	vec3 albedo;
	float t;
	vec3 normal;
	float emission;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
	payload.t = -1.0;
	payload.albedo = vec3(0.0);
	payload.emission = 0.0;
	payload.normal = vec3(0.0);
}
