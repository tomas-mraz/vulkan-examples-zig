# 17_ray7_denoiser — Path Tracing + Real-Time Denoiser

Vulkan ray-tracing example built on top of `16_ray6` — same Cornell-box path
tracer (Lambert + GGX + glass, NEE + MIS, Russian roulette) with a
**real-time SVGF-light denoiser** layered on top of the 1-SPP signal.

The progressive frame accumulator from `16_ray6` is replaced with a
**temporal reproject + spatial à-trous wavelet** filter chain that runs as
compute passes after `cmdTraceRaysKHR`. Goal: usable image during camera
motion (where pure accumulation has nothing to average), not just when the
view is static.

Vendor-neutral GLSL — runs on any Vulkan device that supports
`VK_KHR_ray_tracing_pipeline` + compute (desktop, also Adreno 740+ / Xclipse
920 / Mali Immortalis-G715+ on Android). No external denoiser library;
reference algorithms are SVGF (Schied et al. 2017) and the Khronos
`Vulkan-Samples` `hybrid_ray_tracing` sample.

---

## Idea in one paragraph

Each pixel shoots **one primary ray per frame** through the path tracer
inherited from `16_ray6` (Lambert + GGX + glass, NEE + MIS, Russian roulette,
max 8 bounces). The 1-SPP output is **demodulated by the primary-hit albedo**
and fed into a **two-stage denoiser**: a temporal reprojection pass that
re-uses the previous frame's denoised result via motion vectors (with
disocclusion rejection against the previous-frame G-buffer), followed by
**N à-trous wavelet iterations** with normal/depth/luminance edge-stopping
weights. The denoiser output is **re-modulated by albedo**, tone-mapped, and
copied to the swapchain. There is no progressive accumulator — every frame
is fully formed, including the frame right after a camera teleport.

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
17_ray7_denoiser/
├── README.md                    # this file
├── build.zig
├── build.zig.zon                # depends only on vulkan_ash (no zgltf)
├── Makefile
├── shaders/
│   ├── raygen.rgen              # 1-SPP path trace + G-buffer write + albedo demod
│   ├── miss.rmiss               # black background (closed box)
│   ├── closesthit.rchit         # Lambert / GGX / glass path step + RNG
│   ├── temporal_reproject.comp  # motion-vector reproject + disocclusion + accum
│   ├── atrous.comp              # edge-stopping à-trous iteration (run N×)
│   └── remodulate.comp          # × albedo + Reinhard + gamma + swapchain copy
└── src/
    ├── main.zig                 # ash.DesktopHost + Session + Renderer
    └── renderer.zig             # geometry, BLAS/TLAS, RT + compute pipelines
```

### Descriptor set layout

A single `set 0` shared across RT and compute pipelines. Most images are
declared `r/w` so the same set works as both input and output across the
pipeline phases.

| binding | type                       | stage                     | purpose                                          |
|---------|----------------------------|---------------------------|--------------------------------------------------|
| 0       | acceleration_structure_khr | rgen, rchit               | TLAS                                             |
| 1       | storage_image (rgba16f)    | rgen, comp                | `radiance_demodulated` — 1-SPP path-trace output |
| 2       | storage_image (rgba16f)    | rgen, comp                | `gbuffer_normal_depth` — world-space N + hitT    |
| 3       | storage_image (rg16f)      | rgen, comp                | `gbuffer_motion` — screen-space motion vector    |
| 4       | storage_image (rgba8)      | rgen, comp                | `albedo` — first-hit albedo (for re-modulation)  |
| 5       | storage_image (rgba16f)    | comp                      | `history_curr` — temporal accum output           |
| 6       | storage_image (rgba16f)    | comp                      | `history_prev` — previous-frame temporal accum   |
| 7       | storage_image (rgba16f)    | comp                      | `gbuffer_prev` — previous-frame N + z (disoccl.) |
| 8       | storage_image (rgba16f)    | comp                      | `atrous_ping` — à-trous ping-pong A              |
| 9       | storage_image (rgba16f)    | comp                      | `atrous_pong` — à-trous ping-pong B              |
| 10      | storage_image              | comp                      | `display` — final tonemap, copied to swapchain   |
| 11      | uniform_buffer             | rgen, rchit, rmiss, comp  | camera + matrices + frame counters               |
| 12      | storage_buffer             | rchit                     | vertex SSBO                                      |
| 13      | storage_buffer             | rchit                     | index SSBO                                       |

### Per-frame draw flow

1. Detect camera move (current vs. previous view matrix). If changed →
   `disocclusion_force = 1` in the UBO (forces history rejection on every
   pixel of this frame's temporal pass).
2. Update UBO: `viewInverse`, `projInverse`, **previous-frame `viewProj`**
   (needed for motion vectors), `frame_index++`, `disocclusion_force`.
3. **`cmdTraceRaysKHR`** — 1 SPP per pixel. Raygen writes
   `radiance_demodulated`, and on the primary hit also fills
   `gbuffer_normal_depth`, `gbuffer_motion`, `albedo`.
4. **`temporal_reproject.comp`** — reads `radiance_demodulated`,
   `history_prev`, `gbuffer_prev`; writes `history_curr` and `atrous_ping`.
5. **`atrous.comp` × N** (typically N=4, stride `1,2,4,8`, ping-pong between
   `atrous_ping` / `atrous_pong`).
6. **`remodulate.comp`** — reads denoised radiance × `albedo`, Reinhard
   tonemap, gamma 2.2, writes `display`.
7. Copy `display` into swapchain, present.
8. Swap roles of `history_curr ↔ history_prev` and
   `gbuffer_normal_depth ↔ gbuffer_prev` for the next frame.

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

## Denoiser

The denoiser is a **SVGF-light** chain: temporal reprojection over motion
vectors followed by 3–5 spatial à-trous wavelet iterations with edge-stopping
weights. "Light" because we skip the per-pixel moment tracking and variance
estimation of full SVGF — instead the à-trous filter is parameterised by a
single luminance-based edge weight and constant strengths per iteration. This
keeps the implementation under ~600 lines of GLSL while still giving a usable
1-SPP image during camera motion (which a pure accumulator cannot).

### Why a denoiser at all

`16_ray6`'s progressive accumulator converges beautifully but only when the
camera is *static* — every camera change resets `accum_count` and you see the
raw 1-SPP noise until tens to hundreds of frames have integrated. A denoiser
trades a small bias and some over-blur for a usable image at 1 SPP **every
frame**, which is what real-time path tracing actually needs.

### Pipeline overview

```
  cmdTraceRaysKHR                 →  radiance_demodulated + G-buffer
        │
        ▼
  temporal_reproject.comp         →  history_curr  (= α·new + (1-α)·history_prev)
        │
        ▼
  atrous.comp  (stride 1)         →  atrous_pong
  atrous.comp  (stride 2)         →  atrous_ping
  atrous.comp  (stride 4)         →  atrous_pong
  atrous.comp  (stride 8)         →  atrous_ping
        │
        ▼
  remodulate.comp                 →  × albedo, tonemap, gamma → display
```

### Phase 1 — Path tracer extensions

Single small change to the existing path tracer: **demodulate albedo at the
primary hit** and write the G-buffer.

- `radiance_demodulated = radiance / max(albedo_primary_hit, ε)` — denoiser
  operates on the low-frequency illumination signal, so it doesn't blur away
  texture/material detail (re-applied in `remodulate.comp`).
- G-buffer write on first hit only: `gbuffer_normal_depth` (`rgba16f`,
  `.xyz` = world-space normal, `.w` = ray hit distance `hitT`),
  `gbuffer_motion` (`rg16f`, computed from previous-frame `viewProj` × current
  world-space hit point), `albedo` (`rgba8`, first-hit material colour).
  World-space normal + hit distance keeps the G-buffer camera-independent —
  cross-frame disocclusion tests then work directly off invariant quantities.
- Glass / mirror primary hits: write the first non-specular hit's data into
  the G-buffer (otherwise the denoiser tries to filter through perfect
  reflections, which is wrong). Marker bit in `albedo.a` can disable
  denoising on these pixels — fall through to copy raw 1-SPP, accepting
  noise. (Alternative: separate specular-history buffers, full-SVGF-style.
  Out of scope for the lite version.)

### Phase 2 — Temporal reproject

For each output pixel `p`:

1. `uv_prev = uv - motion_vec` — where this surface was last frame.
2. **Disocclusion test** at `uv_prev`:
   - Normal similarity: `dot(normal_curr, normal_prev) > 0.9`
   - Depth similarity: `|depth_curr - depth_prev| / depth_curr < 0.05`
   - In-bounds check: `0 ≤ uv_prev ≤ 1`
   - Either fails (or `disocclusion_force == 1`) ⇒ history rejected.
3. **Blend:** `history_curr = (1-α) · history_prev + α · radiance_demod`
   with `α = 0.1` on history-accepted, `α = 1.0` on history-rejected.
4. Write `history_curr` and also seed `atrous_ping` with the same value for
   phase 3.

The reprojection sample uses 2×2 bilinear on `history_prev`; each of the four
taps is independently rejected if its `gbuffer_prev` fails the disocclusion
test, and remaining valid taps are renormalised. This stops bleeding across
silhouette edges during motion.

### Phase 3 — À-trous wavelet

N iterations (N ∈ {3, 4, 5}; default 4). Each iteration runs the same
compute shader with a different `stride` push constant (`1, 2, 4, 8, …`).
Within a 5×5 footprint at the given stride, the filter is:

```
filtered(p) = Σ h(q-p) · w(p,q) · radiance(q)   /   Σ h(q-p) · w(p,q)
```

where `h` is a fixed 5×5 B3-spline kernel and the edge-stopping weight is

```
w(p,q) = w_normal · w_depth · w_luminance
       = pow(max(0, dot(N_p, N_q)), σ_n)
         · exp(-|z_p - z_q| / (σ_z · |∇z| + ε))
         · exp(-|L_p - L_q| / σ_l)
```

with `σ_n = 128`, `σ_z = 1`, `σ_l = 4` as starting defaults (tunable). The
luminance term `L = 0.2126·R + 0.7152·G + 0.0722·B` and the gradient
`∇z` is approximated from the depth difference to immediate horizontal /
vertical neighbours.

Two reasons for à-trous instead of straight Gaussian: (1) cost grows linearly
with iterations instead of quadratically with kernel size — five iterations
of 5×5 cover an effective ~80-pixel footprint at the cost of 5 × 25 = 125
taps, vs. 80×80 = 6400 taps for a single-pass Gaussian; (2) the stride-based
expansion preserves high-frequency detail because narrow features that
survive iteration 1 (stride 1) are protected by the edge weights at later
iterations (stride 4+) and not blurred away.

The first iteration's output (stride 1) is written back to `history_curr` as
the value used for the *next* frame's temporal accumulation — feedback this
denoised signal, not the raw `radiance_demodulated`. This is what makes the
filter "temporal-spatial" rather than just "spatial per frame".

### Phase 4 — Remodulate

Final compute pass:

```
denoised_rgb = atrous_pong.rgb   // last à-trous iteration
out_rgb      = denoised_rgb * albedo
tonemap_rgb  = out_rgb / (1 + out_rgb)         // Reinhard
display_rgb  = pow(tonemap_rgb, vec3(1/2.2))   // gamma encode
```

Written to `display`, then copied to swapchain by the same blit path as
`16_ray6`.

### Disocclusion handling and history rejection

Three orthogonal failure modes for the temporal history:

| failure                        | cause                              | response                                    |
|--------------------------------|------------------------------------|---------------------------------------------|
| `uv_prev` outside `[0,1]`      | content rolled in from off-screen  | reject (α=1)                                |
| normal/depth mismatch          | surface behind / in front changed  | reject (α=1)                                |
| camera matrix changed *at all* | teleport, FOV change, resize       | `disocclusion_force` = 1, reject everywhere |

The first two are local; the third is global and is detected on the CPU by
comparing the just-built view matrix against last frame's. (We could let
per-pixel disocclusion handle even teleports, but on a hard cut every pixel
fails anyway and saving the per-pixel work by setting the global force flag
is essentially free.)

### What this denoiser will *not* do well

- **Caustics / mirror reflections / refractions.** SVGF assumes filtering
  works in the screen-space neighbourhood of a pixel, but a sharp specular
  surface gives geometrically unrelated radiance to neighbouring pixels.
  Marker bit on the specular path keeps those pixels noisy rather than
  smeared.
- **Fast disocclusion (rapid camera spin).** Whole regions get `α=1`, which
  reveals raw 1-SPP noise until à-trous spreads it out — visibly noisy for
  ~1-2 frames after fast motion. Full SVGF mitigates this with variance
  estimation that allows wider filtering when history is unstable; we
  approximate it with bumped à-trous iteration count, but not perfectly.
- **Low-frequency colour bleed convergence.** The brute-force accumulator in
  `16_ray6` is *unbiased* — given enough samples it converges to the truth.
  The denoiser is *biased* (over-blurs) and never quite gets there even when
  the camera sits still. Trade-off is fundamental.

### Tunable parameters (UBO / push constants)

| name                 | default | meaning                                          |
|----------------------|---------|--------------------------------------------------|
| `temporal_alpha`     | 0.1     | new-sample weight in temporal blend              |
| `atrous_iterations`  | 4       | number of à-trous passes                         |
| `sigma_normal`       | 128     | normal edge-stop power                           |
| `sigma_depth`        | 1.0     | depth edge-stop scale                            |
| `sigma_luminance`    | 4.0     | luminance edge-stop scale                        |
| `feedback_iteration` | 1       | which à-trous output feeds next frame's history  |

Exposed as runtime keys (F5 / F6 / F7 …) so the effect of each can be
A/B'd on the fly. Pressing `H` toggles between **noisy 1-SPP** (denoiser
bypassed) and **denoised** output for direct comparison.

### Inspiration / reference

- Schied et al., *"Spatiotemporal Variance-Guided Filtering"* (HPG 2017) —
  the original SVGF paper.
- Khronos `Vulkan-Samples` repo, sample `hybrid_ray_tracing` — a working
  per-vendor-neutral reference for the temporal + spatial structure.
- ARM Mali / Qualcomm Adreno RT denoising samples — referenced for mobile
  performance considerations (kernel sizes, format choices).

This is a **clean-room implementation**, not a port — algorithm follows the
paper, code is original GLSL.

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
3. **Stretch (inherited from `16_ray6`)**
   - [x] Mirror BRDF lobe (top of the short box)
   - [x] Free-fly camera (WSAD + mouse, cursor captured)
   - [x] Glossy GGX lobe with VNDF sampling and NEE+BSDF MIS
   - [x] Refractive glass (Snell + Fresnel)
   - [ ] Environment map / HDRI miss shader
4. **Denoiser — new in 17**
   - [x] G-buffer storage images (normal+depth, motion, albedo) + raygen writes
   - [x] Albedo demodulation in raygen + remodulation in final compute pass
   - [x] Previous-frame `viewProj` in UBO + motion-vector math
   - [x] Debug F-key cycle through G-buffer outputs (`F1`=normal, `F2`=depth,
         `F3`=motion, `F4`=albedo, `F5`=noisy 1-SPP, `F6`=denoised composite)
   - [x] `temporal_reproject.comp` — 2×2 bilinear history fetch + per-tap
         disocclusion test + α-blend
   - [ ] `atrous.comp` — single iteration with stride push-constant + 5×5
         B3-spline kernel + edge-stopping weights
   - [ ] Multi-pass à-trous dispatch with ping-pong + feedback to history
   - [ ] `remodulate.comp` — × albedo + Reinhard + gamma → display
   - [ ] Specular-path marker bit in albedo.a, denoiser bypass on those pixels
   - [ ] Runtime tunables on F-keys (`σ_n`, `σ_z`, `σ_l`, iteration count)
   - [ ] CPU-side history pointer swap (`history_curr ↔ history_prev`,
         `gbuffer ↔ gbuffer_prev`) at end of frame
   - [ ] Reset on resize: re-create all denoiser storage images at new extent
