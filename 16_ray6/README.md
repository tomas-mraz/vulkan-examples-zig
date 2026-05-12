# 16_ray6 — Global Illumination via Path Tracing

Vulkan ray-tracing example demonstrating **diffuse global illumination** using a
brute-force Monte Carlo path tracer with progressive frame accumulation.

Builds on the framework from `14_ray4` and `15_ray5` (single BLAS / TLAS, RT
pipeline, SBT). No glTF loading — geometry is a hardcoded Cornell box so the
GI effect (color bleed from the red/green walls) is unmistakable.

---

## Idea in one paragraph

Each pixel shoots **one** primary ray per frame. On a diffuse hit the closest-hit
shader picks a **cosine-weighted random direction** in the hemisphere around the
surface normal and recurses (iteratively, in a loop) up to a fixed bounce depth.
Hitting an emissive surface terminates the path and returns the accumulated
throughput × emission. The result is written into a `rgba32f` accumulation image
and averaged with the prior frames; convergence happens over tens to hundreds of
frames while the camera is static. When the camera moves, the accumulator is
reset.

---

## Scope of this example

### In scope (skeleton — first pass)

- Cornell box geometry (white floor / ceiling / back, red left wall, green right
  wall, emissive area light on the ceiling, two diffuse boxes inside)
- Single BLAS with all triangles, single TLAS instance
- Per-vertex material info: `albedo` + `emission` baked into vertex color +
  alpha-channel marker
- `rgba32f` accumulation storage image, separate from the swapchain copy target
- UBO with `viewInverse`, `projInverse`, `frame_index` (RNG seed),
  `accum_count` (reset to 0 on movement, otherwise increments)
- Cosine-weighted hemisphere sampling (Malley's method — sample disk, project up)
- PCG hash RNG seeded from `(pixel.x, pixel.y, frame_index)`
- Iterative bounce loop (no recursion in shader), max depth 4
- Lambert BRDF only
- Reset-on-resize handling for both storage images

### Implementation in the second phase

- **Next-event estimation (NEE)** — explicit area-light sampling at every diffuse
  bounce, with a shadow ray and area→solid-angle pdf conversion. Massive variance
  reducer; the scene now converges in tens of frames instead of hundreds.
- **Russian roulette** — kicks in from bounce 3, terminates by max-channel
  throughput (clamped to [0.05, 1.0]). `MAX_BOUNCES` raised to 8.
- **Tone mapping** — Reinhard `c/(1+c)` per channel followed by gamma 2.2 on the
  display copy. Without this the colored bleed onto white walls clamped to white.
- **Pixel jitter** — primary-ray sub-pixel offset uses random `vec2(rnd(), rnd())`
  instead of fixed +0.5; gives free anti-aliasing through accumulation.
- **Per-vertex barycentric interpolation** of albedo and emission in the
  closest-hit shader (the original code took only `v0`).
- **Two diffuse boxes inside the Cornell box** — tall box near the red wall,
  short box near the green wall. They make color bleeding obvious on their
  side faces.
- **Glossy / mirror BRDF** on the top of the short box. Encoded as
  `emission < -0.5` sentinel with `roughness = -emission - 1.0`. The path
  tracer dispatches to either a delta lobe (`roughness < 1e-3`: perfect
  reflection, no NEE, throughput tinted by albedo) or a GGX microfacet path
  (Trowbridge-Reitz `D`, height-correlated Smith `G2`, Schlick `F` with
  `F0 = albedo`). The GGX path uses VNDF importance sampling (Heitz 2018)
  and combines NEE with the BSDF sample via the **power-heuristic MIS** —
  necessary for low-roughness lobes where pure NEE has near-zero BRDF and
  pure BSDF sampling rarely hits the small ceiling light. Direct-emission
  accounting on the next emissive hit is tracked through a `pdfBsdfPrev`
  sentinel float (`< 0` count, `0` skip, `> 0` MIS-weight).
- **Free-fly camera** — WSAD + mouse-look. Cursor is captured (FPS-style).
  See _Controls_ below.
- **Denoising** (SVGF / OIDN) — too large to fit alongside the path tracer here.
- **Texturing**, environment maps / HDRI miss shader.

---

## Controls

| key                    | action                                              |
|------------------------|-----------------------------------------------------|
| `W` / `S`              | move forward / back along the ground plane          |
| `A` / `D`              | strafe left / right                                 |
| `Space` / `Left Ctrl`  | move up / down (world Y)                            |
| `Left/Right Shift`     | speed boost (3×)                                    |
| **mouse**              | look around (cursor is captured)                    |
| `Esc`                  | quit                                                |

Any view change resets `accum_count` to 0 — the accumulator is overwritten
by the next sample, so movement gives an instantly-fresh image (with the
expected single-sample noise) that resharpens within a few seconds when you
stop moving.

---

## Architecture (file layout)

```
16_ray6/
├── README.md                # this file
├── build.zig
├── build.zig.zon            # depends only on vulkan_ash (no zgltf)
├── Makefile
├── shaders/
│   ├── raygen.rgen          # primary-ray launch, accumulator read/write
│   ├── miss.rmiss           # black background (closed box)
│   ├── closesthit.rchit     # Lambert path-tracer step + RNG
│   └── common.glsl          # shared structs + RNG (PCG)  ← optional
└── src/
    ├── main.zig             # ash.DesktopHost + Session + Renderer
    └── ray6_renderer.zig    # geometry, BLAS/TLAS, pipeline, SBT, accumulator
```

### Descriptor set layout

| binding | type                       | stage                     | purpose                          |
|---------|----------------------------|---------------------------|----------------------------------|
| 0       | acceleration_structure_khr | rgen, rchit               | TLAS                             |
| 1       | storage_image (rgba32f)    | rgen                      | accumulation image (read+write)  |
| 2       | storage_image              | rgen                      | display image (write, copied to swapchain) |
| 3       | uniform_buffer             | rgen, rchit, rmiss        | camera + frame counters          |
| 4       | storage_buffer             | rchit                     | vertex SSBO                      |
| 5       | storage_buffer             | rchit                     | index SSBO                       |

### Per-frame draw flow

1. Compare current camera matrix against previous frame's; if changed → reset
   `accum_count = 0` (and clear accumulation image).
2. Update UBO with `viewInverse`, `projInverse`, `frame_index++`, `accum_count`.
3. `cmdTraceRaysKHR` — each pixel runs the path tracer once, mixes its result
   into the accumulation image as
   `accum = (accum * accum_count + sample) / (accum_count + 1)`.
4. Convert accumulator → display image (linear → sRGB cast for now).
5. Copy display image into swapchain, present.

### Vertex layout

Same 12 f32 / vertex as `14_ray4` to keep the unpack helper reusable:

```
d0 = (pos.x, pos.y, pos.z, normal.x)
d1 = (normal.y, normal.z, uv.x,    uv.y)
d2 = (albedo.r, albedo.g, albedo.b, emission_strength)
```

Material is encoded by the sign of `emission_strength`:

| value      | meaning                                                              |
|------------|----------------------------------------------------------------------|
| `== 0`     | diffuse Lambert surface, BRDF = albedo / π                           |
| `> 0`      | emissive (multiplied by `albedo` for tint, terminates the path)      |
| `< -0.5`   | glossy GGX with `roughness = -emission - 1.0` (clamped to `[1e-4, 1]`). `F0 = albedo` (metallic-tinted) |
| `< -10`    | dielectric glass with `IOR = -emission * 0.1`. Schlick Fresnel splits perfect reflect/refract; TIR handled |

---

## Implementation roadmap

1. **Skeleton**
   - [x] Repo scaffolding, build files, shader build wiring
   - [x] Cornell box geometry hardcoded in `ray6_renderer.zig`
   - [x] Two diffuse boxes inside (tall by the red wall, short by the green)
   - [x] Accumulation storage image + display image
   - [x] RT pipeline + SBT (rgen / miss / chit)
   - [x] Raygen with pixel-level accumulation arithmetic
   - [x] Closest-hit with cosine hemisphere bounce + iterative loop
   - [x] PCG RNG + frame seed
2. **Visual polish & noise reduction**
   - [x] NEE: sample the area light explicitly at every bounce
   - [x] Russian roulette (kicks in after bounce 3, max bounces 8)
   - [x] Reinhard tone-mapping + gamma encode on display copy
   - [x] Pixel jitter for free anti-aliasing through accumulation
   - [x] Per-vertex barycentric albedo + emission interpolation
   - [x] Reset accumulation when view matrix changes
3. **Stretch**
   - [x] Mirror BRDF lobe (top of the short box)
   - [x] Free-fly camera (WSAD + mouse, cursor captured)
   - [x] Glossy GGX lobe with VNDF sampling and NEE+BSDF MIS
   - [x] Refractive glass (Snell + Fresnel)
   - [ ] Environment map / HDRI miss shader
