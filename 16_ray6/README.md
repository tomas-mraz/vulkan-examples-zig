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

- **Next-event estimation (NEE)** — explicit area-light sampling at every bounce.
  The big variance reducer; ~50 lines of GLSL. Add after the brute-force version
  is visibly working.
- **Russian roulette** termination — replaces fixed bounce depth, ~10 lines.
- **Tone mapping** (Reinhard / ACES) on the display copy. The skeleton uses raw
  linear → sRGB swapchain.
- **Camera controls** — mouse-look or WASD. Skeleton uses a slow auto-orbit so
  you can verify accumulation reset works.
- **Specular / glossy materials** — pure diffuse for now.
- **Denoising** (SVGF / OIDN) — out of scope for this repo.

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

`emission_strength == 0` → diffuse surface; `> 0` → emissive (multiplies `albedo`
to give radiance, terminates the path).

---

## Implementation roadmap

1. **Skeleton** _(this commit)_
   - [x] Repo scaffolding, build files, shader build wiring
   - [x] Cornell box geometry hardcoded in `ray6_renderer.zig`
   - [x] Accumulation storage image + display image
   - [x] RT pipeline + SBT (rgen / miss / chit)
   - [x] Raygen with pixel-level accumulation arithmetic
   - [x] Closest-hit with cosine hemisphere bounce + iterative loop
   - [x] PCG RNG + frame seed
   - [x] Slow camera orbit so reset behavior is observable
2. **Visual polish & noise reduction**
   - [x] NEE: sample the area light explicitly at every bounce
   - [x] Russian roulette (kicks in after bounce 3)
   - [x] Reinhard tone-mapping + gamma encode on display copy
   - [x] Pixel jitter for free anti-aliasing through accumulation
   - [x] Per-vertex barycentric albedo + emission interpolation
   - [x] Reset accumulation when view matrix changes
3. **Stretch**
   - [ ] Glossy / mirror BSDF lobe
   - [ ] Free-fly camera (WASD + mouse)
