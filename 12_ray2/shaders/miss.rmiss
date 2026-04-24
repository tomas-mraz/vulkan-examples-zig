/* Copyright (c) 2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : require

#include "payload.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
	if ((payload.flags & PAYLOAD_FLAG_REFLECTION) != 0u) {
		// Procedural sky + sun for reflection rays. Without this, glass would only
		// reflect the matte helmet and the white background, which is visually flat.
		vec3 dir = normalize(gl_WorldRayDirectionEXT);
		float t = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
		vec3 sky = mix(vec3(0.55, 0.7, 0.95), vec3(0.15, 0.35, 0.85), t);
		float sun = pow(max(0.0, dot(dir, GLASS_LIGHT_DIR)), 800.0);
		sky += vec3(8.0, 7.0, 5.5) * sun;
		payload.radiance = sky;
	} else {
		payload.radiance = vec3(1.0);
	}
}
