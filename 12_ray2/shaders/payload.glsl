/* Shared ray payload definition and glass parameters. */

struct Payload {
	vec3 radiance;            // color from the eventual closest-hit or miss
	uint seed;                // RNG state for alpha-cutout decisions in any-hit
	float glassCoverage;      // [0,1] accumulated as the ray passes glass surfaces
	float glassFirstT;        // ray-distance to nearest glass surface, -1 if none recorded
	vec3 glassFirstNormal;    // world-space normal at nearest glass surface
	uint flags;               // see PAYLOAD_FLAG_*
};

// Set on rays that should skip glass-surface recording (reflection rays use this
// to avoid spawning reflections-of-reflections, which would exceed recursion depth).
const uint PAYLOAD_FLAG_SKIP_GLASS_RECORD = 1u;

// Set on reflection rays — miss shader uses a procedural sky for these so glass
// surfaces have something visually rich to reflect (the scene itself is empty).
const uint PAYLOAD_FLAG_REFLECTION = 2u;

// Per-surface glass opacity. With front+back faces of the visor both traversed,
// compound coverage = 1 - (1 - GLASS_OPACITY)^2. Must match glass_opacity in ray2_renderer.zig.
const float GLASS_OPACITY = 0.5;

// Subtle blue tint applied to whatever is behind the glass.
const vec3 GLASS_TINT = vec3(0.85, 0.92, 1.0);

// Schlick base reflectance. Physically ~0.04 for clean glass, but boosted slightly so
// head-on environment reflections are visible against a low-contrast scene.
const float GLASS_F0 = 0.08;

// Fixed key-light direction (world space) used for the Phong-style specular glint.
// This produces a bright highlight on the glass that moves visibly as the model rotates,
// which is the cue the eye uses to read a surface as "glassy".
const vec3 GLASS_LIGHT_DIR = normalize(vec3(0.5, 0.7, 0.4));
const float GLASS_SHININESS = 80.0;
const float GLASS_GLINT_INTENSITY = 1.6;
