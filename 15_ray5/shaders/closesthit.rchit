#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 0) callableDataEXT vec3 outColor;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

void main()
{
	// Dispatch a different callable shader per geometry index.
	// The BLAS contains three separate triangle geometries, so
	// gl_GeometryIndexEXT picks callable 0, 1 or 2 from the SBT.
	executeCallableEXT(gl_GeometryIndexEXT, 0);

	hitValue = outColor;
}
