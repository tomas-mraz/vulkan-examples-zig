const ash = @import("ash");
const std = @import("std");
const zgltf = @import("zgltf");
const vk = ash.vk;
const math = ash.math;

const raygen_spv align(@alignOf(u32)) = @embedFile("raygen_shader").*;
const miss_spv align(@alignOf(u32)) = @embedFile("miss_shader").*;
const shadow_miss_spv align(@alignOf(u32)) = @embedFile("shadow_miss_shader").*;
const closest_hit_spv align(@alignOf(u32)) = @embedFile("closest_hit_shader").*;
const any_hit_spv align(@alignOf(u32)) = @embedFile("any_hit_shader").*;

const Allocator = std.mem.Allocator;
const Device = vk.DeviceProxy;
const Mat4 = math.Mat4;

const floats_per_vertex: usize = 8; // pos(3) + normal(3) + uv(2) = 8, read as 2x vec4 in shader
const vertex_stride_bytes: vk.DeviceSize = floats_per_vertex * @sizeOf(f32);

const UniformData = extern struct {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    frame: u32,
};

const GeometryNode = extern struct {
    vertex_device_address: u64,
    index_device_address: u64,
    texture_index_base_color: i32,
    texture_index_occlusion: i32,
    is_glass: u32,
    _padding: u32 = 0,
};

// Per-surface glass opacity — kept in sync with GLASS_OPACITY in shaders/payload.glsl.
// Transparency is applied deterministically in any-hit (coverage accumulator) plus tint in raygen,
// so there is no stochastic noise at the glass surfaces regardless of sample count.
const glass_opacity: f32 = 0.5;

const Buffer = struct {
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    size: vk.DeviceSize = 0,
    device_address: vk.DeviceAddress = 0,

    fn deinit(self: *Buffer, device: Device) void {
        if (self.buffer != .null_handle) {
            device.destroyBuffer(self.buffer, null);
            self.buffer = .null_handle;
        }
        if (self.memory != .null_handle) {
            device.freeMemory(self.memory, null);
            self.memory = .null_handle;
        }
        self.size = 0;
        self.device_address = 0;
    }
};

const AccelStruct = struct {
    handle: vk.AccelerationStructureKHR = .null_handle,
    buffer: Buffer = .{},
    device_address: vk.DeviceAddress = 0,

    fn deinit(self: *AccelStruct, device: Device) void {
        if (self.handle != .null_handle) {
            device.destroyAccelerationStructureKHR(self.handle, null);
            self.handle = .null_handle;
        }
        self.buffer.deinit(device);
        self.device_address = 0;
    }
};

const StorageImage = struct {
    image: vk.Image = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    view: vk.ImageView = .null_handle,
    format: vk.Format = .undefined,
    extent: vk.Extent2D = .{ .width = 0, .height = 0 },

    fn deinit(self: *StorageImage, device: Device) void {
        if (self.view != .null_handle) {
            device.destroyImageView(self.view, null);
            self.view = .null_handle;
        }
        if (self.image != .null_handle) {
            device.destroyImage(self.image, null);
            self.image = .null_handle;
        }
        if (self.memory != .null_handle) {
            device.freeMemory(self.memory, null);
            self.memory = .null_handle;
        }
    }
};

const Primitive = struct {
    vertex_buf: Buffer = .{},
    index_buf: Buffer = .{},
    triangle_count: u32 = 0,
    max_vertex: u32 = 0,
    transform: [12]f32 = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 },
    base_color_tex: i32 = 0,
    occlusion_tex: i32 = -1,
    is_glass: bool = false,
};

pub const Ray2Renderer = struct {
    allocator: Allocator,

    manager: ?*ash.Manager = null,
    cmd_ctx: ?*ash.CommandContext = null,
    device: ?Device = null,

    rt_props: vk.PhysicalDeviceRayTracingPipelinePropertiesKHR = undefined,

    // once
    primitives: []Primitive = &.{},
    geometry_nodes_buf: Buffer = .{},
    blas: AccelStruct = .{},
    tlas: AccelStruct = .{},
    textures: []ash.ImageResource = &.{},
    frame_counter: u32 = 0,

    // sized
    storage_img: StorageImage = .{},
    uniforms: []Buffer = &.{},
    desc_pool: vk.DescriptorPool = .null_handle,
    desc_set_layout: vk.DescriptorSetLayout = .null_handle,
    desc_sets: []vk.DescriptorSet = &.{},
    pipeline_layout: vk.PipelineLayout = .null_handle,
    pipeline: vk.Pipeline = .null_handle,
    sbt: Buffer = .{},
    sbt_raygen_region: vk.StridedDeviceAddressRegionKHR = .{ .stride = 0, .size = 0 },
    sbt_miss_region: vk.StridedDeviceAddressRegionKHR = .{ .stride = 0, .size = 0 },
    sbt_hit_region: vk.StridedDeviceAddressRegionKHR = .{ .stride = 0, .size = 0 },
    sbt_callable_region: vk.StridedDeviceAddressRegionKHR = .{ .stride = 0, .size = 0 },

    view_matrix: Mat4 = math.identity(),
    start_time: f64 = 0,

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: Allocator) Ray2Renderer {
        return .{ .allocator = allocator };
    }

    pub fn createOnce(self: *Ray2Renderer, session: *ash.Session) !void {
        self.manager = &session.manager.?;
        self.cmd_ctx = &session.cmd_ctx.?;
        self.device = self.manager.?.device.?;

        try self.queryRtProperties();

        try self.loadModel("assets/FlightHelmet/FlightHelmet.gltf");
        std.log.info("loaded {} primitives, {} textures", .{ self.primitives.len, self.textures.len });

        try self.buildBlas();
        try self.createGeometryNodesBuffer();
        try self.buildTlas();

        self.view_matrix = math.translation(0.0, -0.1, -1.0);
        self.start_time = ash.glfw.getTime();
        self.frame_counter = 0;

        self.once_built = true;
    }

    pub fn destroyOnce(self: *Ray2Renderer) void {
        if (!self.once_built) return;
        const device = self.device.?;
        self.tlas.deinit(device);
        self.geometry_nodes_buf.deinit(device);
        self.blas.deinit(device);
        for (self.primitives) |*p| {
            p.index_buf.deinit(device);
            p.vertex_buf.deinit(device);
        }
        if (self.primitives.len != 0) self.allocator.free(self.primitives);
        self.primitives = &.{};
        for (self.textures) |*t| t.deinit(device);
        if (self.textures.len != 0) self.allocator.free(self.textures);
        self.textures = &.{};
        self.once_built = false;
    }

    pub fn createSized(self: *Ray2Renderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const swap_len = session.swapchain.?.imageCount();
        const display_format = session.swapchain.?.surface_format.format;

        try self.createStorageImage(extent, display_format);
        try self.createUniformBuffers(swap_len);
        try self.createDescriptorSet(swap_len);
        try self.createRtPipeline();
        try self.createShaderBindingTable();

        self.frame_counter = 0;
        self.sized_built = true;
    }

    pub fn destroySized(self: *Ray2Renderer) void {
        if (!self.sized_built) return;
        const device = self.device.?;
        self.sbt.deinit(device);
        if (self.pipeline != .null_handle) {
            device.destroyPipeline(self.pipeline, null);
            self.pipeline = .null_handle;
        }
        if (self.pipeline_layout != .null_handle) {
            device.destroyPipelineLayout(self.pipeline_layout, null);
            self.pipeline_layout = .null_handle;
        }
        if (self.desc_pool != .null_handle) {
            device.destroyDescriptorPool(self.desc_pool, null);
            self.desc_pool = .null_handle;
        }
        if (self.desc_set_layout != .null_handle) {
            device.destroyDescriptorSetLayout(self.desc_set_layout, null);
            self.desc_set_layout = .null_handle;
        }
        if (self.desc_sets.len != 0) self.allocator.free(self.desc_sets);
        self.desc_sets = &.{};
        for (self.uniforms) |*ub| ub.deinit(device);
        if (self.uniforms.len != 0) self.allocator.free(self.uniforms);
        self.uniforms = &.{};
        self.storage_img.deinit(device);
        self.sized_built = false;
    }

    pub fn draw(self: *Ray2Renderer, session: *ash.Session, frame: *const ash.Frame) !void {
        _ = session;
        const device = self.device.?;

        const aspect: f32 = @as(f32, @floatFromInt(frame.extent.width)) / @as(f32, @floatFromInt(frame.extent.height));
        const proj = perspectiveZO(math.degreesToRadians(60.0), aspect, 0.1, 512.0);

        const elapsed = @as(f32, @floatCast(ash.glfw.getTime() - self.start_time)) * 45.0;
        const rot = math.rotationAxis(0, 1, 0, math.degreesToRadians(elapsed));
        const rotated_view = math.multiply(&self.view_matrix, &rot);
        // Camera animates every frame — reset accumulation each frame.
        self.frame_counter = 0;

        var ubo = UniformData{
            .view_inverse = math.invert(rotated_view),
            .proj_inverse = math.invert(proj),
            .frame = self.frame_counter,
        };
        self.frame_counter +%= 1;

        const ub = &self.uniforms[frame.image_index];
        const mapped = try device.mapMemory(ub.memory, 0, vk.WHOLE_SIZE, .{});
        @memcpy(@as([*]u8, @ptrCast(@alignCast(mapped)))[0..@sizeOf(UniformData)], std.mem.asBytes(&ubo));
        device.unmapMemory(ub.memory);

        transitionImage(device, frame.cmd, self.storage_img.image, .undefined, .general, .{
            .src_stage = .{ .top_of_pipe_bit = true },
            .dst_stage = .{ .ray_tracing_shader_bit_khr = true },
            .src_access = .{},
            .dst_access = .{ .shader_write_bit = true },
            .aspect = .{ .color_bit = true },
        });

        device.cmdBindPipeline(frame.cmd, .ray_tracing_khr, self.pipeline);
        device.cmdBindDescriptorSets(
            frame.cmd,
            .ray_tracing_khr,
            self.pipeline_layout,
            0,
            self.desc_sets[frame.image_index..@intCast(frame.image_index + 1)],
            null,
        );
        device.cmdTraceRaysKHR(
            frame.cmd,
            &self.sbt_raygen_region,
            &self.sbt_miss_region,
            &self.sbt_hit_region,
            &self.sbt_callable_region,
            frame.extent.width,
            frame.extent.height,
            1,
        );

        transitionImage(device, frame.cmd, self.storage_img.image, .general, .transfer_src_optimal, .{
            .src_stage = .{ .ray_tracing_shader_bit_khr = true },
            .dst_stage = .{ .transfer_bit = true },
            .src_access = .{ .shader_write_bit = true },
            .dst_access = .{ .transfer_read_bit = true },
            .aspect = .{ .color_bit = true },
        });
        const swap_image = frame.swapchain.swap_images[frame.image_index].image;
        transitionImage(device, frame.cmd, swap_image, .undefined, .transfer_dst_optimal, .{
            .src_stage = .{ .top_of_pipe_bit = true },
            .dst_stage = .{ .transfer_bit = true },
            .src_access = .{},
            .dst_access = .{ .transfer_write_bit = true },
            .aspect = .{ .color_bit = true },
        });

        const copy_region = vk.ImageCopy{
            .src_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
            .src_offset = .{ .x = 0, .y = 0, .z = 0 },
            .dst_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
            .dst_offset = .{ .x = 0, .y = 0, .z = 0 },
            .extent = .{ .width = frame.extent.width, .height = frame.extent.height, .depth = 1 },
        };
        device.cmdCopyImage(
            frame.cmd,
            self.storage_img.image,
            .transfer_src_optimal,
            swap_image,
            .transfer_dst_optimal,
            @as([]const vk.ImageCopy, (&copy_region)[0..1]),
        );

        transitionImage(device, frame.cmd, swap_image, .transfer_dst_optimal, .present_src_khr, .{
            .src_stage = .{ .transfer_bit = true },
            .dst_stage = .{ .bottom_of_pipe_bit = true },
            .src_access = .{ .transfer_write_bit = true },
            .dst_access = .{},
            .aspect = .{ .color_bit = true },
        });
    }

    // --- model loading ---

    fn loadModel(self: *Ray2Renderer, path: []const u8) !void {
        var io_threaded: std.Io.Threaded = .init_single_threaded;
        const io = io_threaded.io();
        const gltf_json = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .unlimited);
        defer self.allocator.free(gltf_json);

        const aligned_json = try self.allocator.alignedAlloc(u8, .of(u32), gltf_json.len);
        defer self.allocator.free(aligned_json);
        @memcpy(aligned_json, gltf_json);

        var gltf = zgltf.Gltf.init(self.allocator);
        defer gltf.deinit();
        try gltf.parse(aligned_json);

        if (gltf.data.buffers.len == 0) return error.GltfMissingBuffer;
        const bin_uri = gltf.data.buffers[0].uri orelse return error.GltfMissingBufferUri;

        const base_dir = std.fs.path.dirname(path) orelse ".";
        const bin_path = try std.fs.path.join(self.allocator, &.{ base_dir, bin_uri });
        defer self.allocator.free(bin_path);
        const raw_bin = try std.Io.Dir.cwd().readFileAlloc(io, bin_path, self.allocator, .unlimited);
        defer self.allocator.free(raw_bin);
        const binary = try self.allocator.alignedAlloc(u8, .of(f32), raw_bin.len);
        defer self.allocator.free(binary);
        @memcpy(binary, raw_bin);

        // Load all textures (index 0 = 1x1 white fallback).
        try self.loadTextures(&gltf, base_dir);

        // Walk the scene and build primitives.
        if (gltf.data.scenes.len == 0) return error.GltfNoScenes;
        const scene = gltf.data.scenes[0];
        const roots = scene.nodes orelse return error.GltfSceneEmpty;

        var prims = std.ArrayList(Primitive).empty;
        errdefer {
            for (prims.items) |*p| {
                p.index_buf.deinit(self.device.?);
                p.vertex_buf.deinit(self.device.?);
            }
            prims.deinit(self.allocator);
        }

        for (roots) |root| {
            try self.visitNode(&gltf, binary, root, math.identity(), &prims);
        }

        if (prims.items.len == 0) return error.GltfNoPrimitives;
        self.primitives = try prims.toOwnedSlice(self.allocator);
    }

    fn visitNode(
        self: *Ray2Renderer,
        gltf: *const zgltf.Gltf,
        binary: []align(@alignOf(f32)) const u8,
        node_index: usize,
        parent_transform: Mat4,
        prims: *std.ArrayList(Primitive),
    ) !void {
        const node = gltf.data.nodes[node_index];
        const local = nodeTransform(node);
        const world = math.multiply(&parent_transform, &local);

        if (node.mesh) |mesh_index| {
            const mesh = gltf.data.meshes[mesh_index];
            for (mesh.primitives) |prim| {
                try self.buildPrimitive(gltf, binary, prim, world, prims);
            }
        }
        for (node.children) |child| {
            try self.visitNode(gltf, binary, child, world, prims);
        }
    }

    fn buildPrimitive(
        self: *Ray2Renderer,
        gltf: *const zgltf.Gltf,
        binary: []align(@alignOf(f32)) const u8,
        prim: zgltf.Gltf.Primitive,
        world: Mat4,
        prims: *std.ArrayList(Primitive),
    ) !void {
        var pos_idx: ?usize = null;
        var norm_idx: ?usize = null;
        var uv_idx: ?usize = null;
        for (prim.attributes) |attr| switch (attr) {
            .position => |i| pos_idx = i,
            .normal => |i| norm_idx = i,
            .texcoord => |i| {
                if (uv_idx == null) uv_idx = i;
            },
            else => {},
        };
        const pa = pos_idx orelse return error.GltfMissingPositions;
        const na = norm_idx orelse return error.GltfMissingNormals;
        const ua = uv_idx orelse return error.GltfMissingTexcoords;
        const ia = prim.indices orelse return error.GltfMissingIndices;

        const pos_acc = gltf.data.accessors[pa];
        const norm_acc = gltf.data.accessors[na];
        const uv_acc = gltf.data.accessors[ua];
        if (pos_acc.count != norm_acc.count or pos_acc.count != uv_acc.count) {
            return error.GltfAttributeMismatch;
        }
        const vertex_count: u32 = @intCast(pos_acc.count);

        const vertices = try self.allocator.alloc(f32, pos_acc.count * floats_per_vertex);
        defer self.allocator.free(vertices);

        var pi = pos_acc.iterator(f32, @constCast(gltf), binary);
        var ni = norm_acc.iterator(f32, @constCast(gltf), binary);
        var ui = uv_acc.iterator(f32, @constCast(gltf), binary);

        var k: usize = 0;
        while (k < pos_acc.count) : (k += 1) {
            const p = pi.next() orelse return error.GltfPositionShort;
            const n = ni.next() orelse return error.GltfNormalShort;
            const u = ui.next() orelse return error.GltfUvShort;
            const tn = transformNormal(world, n[0], n[1], n[2]);
            const base = k * floats_per_vertex;
            vertices[base + 0] = p[0];
            vertices[base + 1] = p[1];
            vertices[base + 2] = p[2];
            vertices[base + 3] = tn[0];
            vertices[base + 4] = tn[1];
            vertices[base + 5] = tn[2];
            vertices[base + 6] = u[0];
            vertices[base + 7] = u[1];
        }

        const idx_acc = gltf.data.accessors[ia];
        const indices = try self.allocator.alloc(u32, idx_acc.count);
        defer self.allocator.free(indices);
        switch (idx_acc.component_type) {
            .unsigned_short => {
                var it = idx_acc.iterator(u16, @constCast(gltf), binary);
                var j: usize = 0;
                while (it.next()) |s| : (j += 1) indices[j] = @as(u32, s[0]);
            },
            .unsigned_integer => {
                var it = idx_acc.iterator(u32, @constCast(gltf), binary);
                var j: usize = 0;
                while (it.next()) |s| : (j += 1) indices[j] = s[0];
            },
            .unsigned_byte => {
                var it = idx_acc.iterator(u8, @constCast(gltf), binary);
                var j: usize = 0;
                while (it.next()) |s| : (j += 1) indices[j] = @as(u32, s[0]);
            },
            else => return error.GltfUnsupportedIndexType,
        }

        const rt_usage = vk.BufferUsageFlags{
            .shader_device_address_bit = true,
            .acceleration_structure_build_input_read_only_bit_khr = true,
            .storage_buffer_bit = true,
        };
        var primitive = Primitive{
            .triangle_count = @intCast(indices.len / 3),
            .max_vertex = if (vertex_count == 0) 0 else vertex_count - 1,
            .transform = vkTransformFromMat4(world),
        };
        primitive.vertex_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(vertices), rt_usage, true);
        primitive.index_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(indices), rt_usage, true);

        // Material texture indices (glTF textures start at index 0, we offset by +1 for fallback at slot 0).
        if (prim.material) |mi| {
            if (mi < gltf.data.materials.len) {
                const mat = gltf.data.materials[mi];
                if (mat.metallic_roughness.base_color_texture) |tex_info| {
                    primitive.base_color_tex = @intCast(@as(u32, @intCast(tex_info.index)) + 1);
                }
                if (mat.occlusion_texture) |occ| {
                    primitive.occlusion_tex = @intCast(@as(u32, @intCast(occ.index)) + 1);
                }
                if (mat.name) |n| {
                    // FlightHelmet: "LensesMat" are the transparent visor lenses.
                    // "GlassPlasticMat" is the opaque plastic frame around them — do not match.
                    primitive.is_glass = std.ascii.indexOfIgnoreCase(n, "lens") != null;
                }
            }
        }

        try prims.append(self.allocator, primitive);
    }

    fn loadTextures(self: *Ray2Renderer, gltf: *const zgltf.Gltf, base_dir: []const u8) !void {
        const device = self.device.?;
        const white_pixel = [4]u8{ 255, 255, 255, 255 };
        const total = gltf.data.textures.len + 1;
        const textures = try self.allocator.alloc(ash.ImageResource, total);
        var created: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < created) : (i += 1) textures[i].deinit(device);
            self.allocator.free(textures);
        }

        textures[0] = try ash.createTextureFromRgba(self.manager.?, self.cmd_ctx.?, 1, 1, white_pixel[0..], .{});
        created = 1;

        var io_threaded: std.Io.Threaded = .init_single_threaded;
        const io = io_threaded.io();

        for (gltf.data.textures, 0..) |tex, idx| {
            const slot = idx + 1;
            var used_fallback = true;
            if (tex.source) |image_index| {
                const image = gltf.data.images[image_index];
                if (image.uri) |uri| {
                    const img_path = try std.fs.path.join(self.allocator, &.{ base_dir, uri });
                    defer self.allocator.free(img_path);
                    const encoded = try std.Io.Dir.cwd().readFileAlloc(io, img_path, self.allocator, .unlimited);
                    defer self.allocator.free(encoded);
                    textures[slot] = try decodeAndUpload(self.allocator, self.manager.?, self.cmd_ctx.?, encoded);
                    used_fallback = false;
                }
            }
            if (used_fallback) {
                textures[slot] = try ash.createTextureFromRgba(self.manager.?, self.cmd_ctx.?, 1, 1, white_pixel[0..], .{});
            }
            created = slot + 1;
        }

        self.textures = textures;
    }

    // --- RT setup helpers ---

    fn queryRtProperties(self: *Ray2Renderer) !void {
        var props2: vk.PhysicalDeviceProperties2 = .{ .properties = undefined };
        self.rt_props = .{
            .p_next = null,
            .shader_group_handle_size = 0,
            .max_ray_recursion_depth = 0,
            .max_shader_group_stride = 0,
            .shader_group_base_alignment = 0,
            .shader_group_handle_capture_replay_size = 0,
            .max_ray_dispatch_invocation_count = 0,
            .shader_group_handle_alignment = 0,
            .max_ray_hit_attribute_size = 0,
        };
        props2.p_next = @ptrCast(&self.rt_props);
        self.manager.?.instance.?.getPhysicalDeviceProperties2(self.manager.?.gpu, &props2);
    }

    fn createBufferHostVisible(
        self: *Ray2Renderer,
        data: []const u8,
        usage: vk.BufferUsageFlags,
        device_address: bool,
    ) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = data.len };
        errdefer buf.deinit(device);

        buf.buffer = try device.createBuffer(&.{
            .size = data.len,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);

        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        });

        var alloc_flags = vk.MemoryAllocateFlagsInfo{
            .device_mask = 0,
            .flags = .{ .device_address_bit = device_address },
        };
        const alloc_info = vk.MemoryAllocateInfo{
            .p_next = if (device_address) @as(?*const anyopaque, @ptrCast(&alloc_flags)) else null,
            .allocation_size = reqs.size,
            .memory_type_index = mem_index,
        };
        buf.memory = try device.allocateMemory(&alloc_info, null);
        try device.bindBufferMemory(buf.buffer, buf.memory, 0);

        const mapped = try device.mapMemory(buf.memory, 0, data.len, .{});
        @memcpy(@as([*]u8, @ptrCast(@alignCast(mapped)))[0..data.len], data);
        device.unmapMemory(buf.memory);

        if (device_address) {
            buf.device_address = device.getBufferDeviceAddress(&.{ .buffer = buf.buffer });
        }
        return buf;
    }

    fn createBufferDeviceLocal(
        self: *Ray2Renderer,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        device_address: bool,
    ) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = size };
        errdefer buf.deinit(device);

        buf.buffer = try device.createBuffer(&.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);

        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .device_local_bit = true });

        var alloc_flags = vk.MemoryAllocateFlagsInfo{
            .device_mask = 0,
            .flags = .{ .device_address_bit = device_address },
        };
        const alloc_info = vk.MemoryAllocateInfo{
            .p_next = if (device_address) @as(?*const anyopaque, @ptrCast(&alloc_flags)) else null,
            .allocation_size = reqs.size,
            .memory_type_index = mem_index,
        };
        buf.memory = try device.allocateMemory(&alloc_info, null);
        try device.bindBufferMemory(buf.buffer, buf.memory, 0);

        if (device_address) {
            buf.device_address = device.getBufferDeviceAddress(&.{ .buffer = buf.buffer });
        }
        return buf;
    }

    fn createBufferHostVisibleEmpty(
        self: *Ray2Renderer,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        device_address: bool,
    ) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = size };
        errdefer buf.deinit(device);

        buf.buffer = try device.createBuffer(&.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);
        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        });
        var alloc_flags = vk.MemoryAllocateFlagsInfo{
            .device_mask = 0,
            .flags = .{ .device_address_bit = device_address },
        };
        const alloc_info = vk.MemoryAllocateInfo{
            .p_next = if (device_address) @as(?*const anyopaque, @ptrCast(&alloc_flags)) else null,
            .allocation_size = reqs.size,
            .memory_type_index = mem_index,
        };
        buf.memory = try device.allocateMemory(&alloc_info, null);
        try device.bindBufferMemory(buf.buffer, buf.memory, 0);
        if (device_address) {
            buf.device_address = device.getBufferDeviceAddress(&.{ .buffer = buf.buffer });
        }
        return buf;
    }

    fn buildBlas(self: *Ray2Renderer) !void {
        const device = self.device.?;

        // Upload per-primitive transforms (VkTransformMatrixKHR = 3x4 row-major floats).
        const transforms = try self.allocator.alloc([12]f32, self.primitives.len);
        defer self.allocator.free(transforms);
        for (self.primitives, 0..) |p, i| transforms[i] = p.transform;

        var transform_buf = try self.createBufferHostVisible(
            std.mem.sliceAsBytes(transforms),
            .{ .shader_device_address_bit = true, .acceleration_structure_build_input_read_only_bit_khr = true },
            true,
        );
        defer transform_buf.deinit(device);
        const transform_stride: vk.DeviceAddress = @sizeOf([12]f32);

        const geometries = try self.allocator.alloc(vk.AccelerationStructureGeometryKHR, self.primitives.len);
        defer self.allocator.free(geometries);
        const primitive_counts = try self.allocator.alloc(u32, self.primitives.len);
        defer self.allocator.free(primitive_counts);
        const ranges = try self.allocator.alloc(vk.AccelerationStructureBuildRangeInfoKHR, self.primitives.len);
        defer self.allocator.free(ranges);

        for (self.primitives, 0..) |p, i| {
            const triangles = vk.AccelerationStructureGeometryTrianglesDataKHR{
                .vertex_format = .r32g32b32_sfloat,
                .vertex_data = .{ .device_address = p.vertex_buf.device_address },
                .vertex_stride = vertex_stride_bytes,
                .max_vertex = p.max_vertex,
                .index_type = .uint32,
                .index_data = .{ .device_address = p.index_buf.device_address },
                .transform_data = .{ .device_address = transform_buf.device_address + @as(vk.DeviceAddress, @intCast(i)) * transform_stride },
            };
            geometries[i] = .{
                .geometry_type = .triangles_khr,
                .geometry = .{ .triangles = triangles },
                // Glass primitives must NOT be opaque — any-hit shader handles stochastic transparency.
                .flags = .{ .opaque_bit_khr = !p.is_glass, .no_duplicate_any_hit_invocation_bit_khr = p.is_glass },
            };
            primitive_counts[i] = p.triangle_count;
            ranges[i] = .{
                .primitive_count = p.triangle_count,
                .primitive_offset = 0,
                .first_vertex = 0,
                .transform_offset = 0,
            };
        }

        const flags = vk.BuildAccelerationStructureFlagsKHR{ .prefer_fast_trace_bit_khr = true };
        var build_info_sz = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .bottom_level_khr,
            .flags = flags,
            .mode = .build_khr,
            .geometry_count = @intCast(geometries.len),
            .p_geometries = geometries.ptr,
            .scratch_data = .{ .device_address = 0 },
        };
        var sizes = vk.AccelerationStructureBuildSizesInfoKHR{
            .acceleration_structure_size = 0,
            .update_scratch_size = 0,
            .build_scratch_size = 0,
        };
        device.getAccelerationStructureBuildSizesKHR(
            .device_khr,
            &build_info_sz,
            primitive_counts.ptr,
            &sizes,
        );

        self.blas.buffer = try self.createBufferDeviceLocal(
            sizes.acceleration_structure_size,
            .{ .acceleration_structure_storage_bit_khr = true, .shader_device_address_bit = true },
            true,
        );
        self.blas.handle = try device.createAccelerationStructureKHR(&.{
            .buffer = self.blas.buffer.buffer,
            .offset = 0,
            .size = sizes.acceleration_structure_size,
            .type = .bottom_level_khr,
            .device_address = 0,
        }, null);

        var scratch = try self.createBufferDeviceLocal(
            sizes.build_scratch_size,
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            true,
        );
        defer scratch.deinit(device);

        var build_info = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .bottom_level_khr,
            .flags = flags,
            .mode = .build_khr,
            .dst_acceleration_structure = self.blas.handle,
            .geometry_count = @intCast(geometries.len),
            .p_geometries = geometries.ptr,
            .scratch_data = .{ .device_address = scratch.device_address },
        };
        const range_ptr: [*]const vk.AccelerationStructureBuildRangeInfoKHR = ranges.ptr;

        const cmd = try self.cmd_ctx.?.beginOneTime();
        device.cmdBuildAccelerationStructuresKHR(
            cmd,
            @as([*]const vk.AccelerationStructureBuildGeometryInfoKHR, @ptrCast(&build_info))[0..1],
            @as([*]const [*]const vk.AccelerationStructureBuildRangeInfoKHR, @ptrCast(&range_ptr))[0..1],
        );
        try self.cmd_ctx.?.endOneTime(self.manager.?.graphics_queue.handle, cmd);

        self.blas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{
            .acceleration_structure = self.blas.handle,
        });
    }

    fn createGeometryNodesBuffer(self: *Ray2Renderer) !void {
        const nodes = try self.allocator.alloc(GeometryNode, self.primitives.len);
        defer self.allocator.free(nodes);
        for (self.primitives, 0..) |p, i| {
            nodes[i] = .{
                .vertex_device_address = p.vertex_buf.device_address,
                .index_device_address = p.index_buf.device_address,
                .texture_index_base_color = p.base_color_tex,
                .texture_index_occlusion = p.occlusion_tex,
                .is_glass = if (p.is_glass) 1 else 0,
            };
        }
        self.geometry_nodes_buf = try self.createBufferHostVisible(
            std.mem.sliceAsBytes(nodes),
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            true,
        );
    }

    fn buildTlas(self: *Ray2Renderer) !void {
        const device = self.device.?;

        const instance = vk.AccelerationStructureInstanceKHR{
            // Mirror the Go example: flip Y to correct the model orientation.
            .transform = .{ .matrix = .{
                .{ 1, 0, 0, 0 },
                .{ 0, -1, 0, 0 },
                .{ 0, 0, 1, 0 },
            } },
            .instance_custom_index_and_mask = .{ .instance_custom_index = 0, .mask = 0xFF },
            .instance_shader_binding_table_record_offset_and_flags = .{
                .instance_shader_binding_table_record_offset = 0,
                .flags = @truncate((vk.GeometryInstanceFlagsKHR{ .triangle_facing_cull_disable_bit_khr = true }).toInt()),
            },
            .acceleration_structure_reference = self.blas.device_address,
        };
        var instance_buf = try self.createBufferHostVisible(
            std.mem.asBytes(&instance),
            .{ .shader_device_address_bit = true, .acceleration_structure_build_input_read_only_bit_khr = true },
            true,
        );
        defer instance_buf.deinit(device);

        const instances_data = vk.AccelerationStructureGeometryInstancesDataKHR{
            .array_of_pointers = .false,
            .data = .{ .device_address = instance_buf.device_address },
        };
        const geometry = vk.AccelerationStructureGeometryKHR{
            .geometry_type = .instances_khr,
            .geometry = .{ .instances = instances_data },
            .flags = .{ .opaque_bit_khr = true },
        };

        const flags = vk.BuildAccelerationStructureFlagsKHR{ .prefer_fast_trace_bit_khr = true };
        const primitive_count: u32 = 1;
        var build_info_sz = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .top_level_khr,
            .flags = flags,
            .mode = .build_khr,
            .geometry_count = 1,
            .p_geometries = @ptrCast(&geometry),
            .scratch_data = .{ .device_address = 0 },
        };
        var sizes = vk.AccelerationStructureBuildSizesInfoKHR{
            .acceleration_structure_size = 0,
            .update_scratch_size = 0,
            .build_scratch_size = 0,
        };
        device.getAccelerationStructureBuildSizesKHR(.device_khr, &build_info_sz, @ptrCast(&primitive_count), &sizes);

        self.tlas.buffer = try self.createBufferDeviceLocal(
            sizes.acceleration_structure_size,
            .{ .acceleration_structure_storage_bit_khr = true, .shader_device_address_bit = true },
            true,
        );
        self.tlas.handle = try device.createAccelerationStructureKHR(&.{
            .buffer = self.tlas.buffer.buffer,
            .offset = 0,
            .size = sizes.acceleration_structure_size,
            .type = .top_level_khr,
            .device_address = 0,
        }, null);

        var scratch = try self.createBufferDeviceLocal(
            sizes.build_scratch_size,
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            true,
        );
        defer scratch.deinit(device);

        var build_info = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .top_level_khr,
            .flags = flags,
            .mode = .build_khr,
            .dst_acceleration_structure = self.tlas.handle,
            .geometry_count = 1,
            .p_geometries = @ptrCast(&geometry),
            .scratch_data = .{ .device_address = scratch.device_address },
        };
        const range = vk.AccelerationStructureBuildRangeInfoKHR{
            .primitive_count = 1,
            .primitive_offset = 0,
            .first_vertex = 0,
            .transform_offset = 0,
        };
        const range_ptr: [*]const vk.AccelerationStructureBuildRangeInfoKHR = @ptrCast(&range);

        const cmd = try self.cmd_ctx.?.beginOneTime();
        device.cmdBuildAccelerationStructuresKHR(
            cmd,
            @as([*]const vk.AccelerationStructureBuildGeometryInfoKHR, @ptrCast(&build_info))[0..1],
            @as([*]const [*]const vk.AccelerationStructureBuildRangeInfoKHR, @ptrCast(&range_ptr))[0..1],
        );
        try self.cmd_ctx.?.endOneTime(self.manager.?.graphics_queue.handle, cmd);

        self.tlas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{
            .acceleration_structure = self.tlas.handle,
        });
    }

    fn createStorageImage(self: *Ray2Renderer, extent: vk.Extent2D, format: vk.Format) !void {
        const device = self.device.?;
        self.storage_img = .{ .format = format, .extent = extent };
        errdefer self.storage_img.deinit(device);

        self.storage_img.image = try device.createImage(&.{
            .image_type = .@"2d",
            .format = format,
            .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{ .storage_bit = true, .transfer_src_bit = true, .sampled_bit = true },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);

        const reqs = device.getImageMemoryRequirements(self.storage_img.image);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .device_local_bit = true });
        self.storage_img.memory = try device.allocateMemory(&.{
            .allocation_size = reqs.size,
            .memory_type_index = mem_index,
        }, null);
        try device.bindImageMemory(self.storage_img.image, self.storage_img.memory, 0);

        self.storage_img.view = try device.createImageView(&.{
            .image = self.storage_img.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
    }

    fn createUniformBuffers(self: *Ray2Renderer, count: usize) !void {
        self.uniforms = try self.allocator.alloc(Buffer, count);
        errdefer {
            for (self.uniforms) |*b| b.deinit(self.device.?);
            self.allocator.free(self.uniforms);
            self.uniforms = &.{};
        }
        var i: usize = 0;
        while (i < count) : (i += 1) {
            self.uniforms[i] = try self.createBufferHostVisibleEmpty(@sizeOf(UniformData), .{ .uniform_buffer_bit = true }, false);
        }
    }

    fn createDescriptorSet(self: *Ray2Renderer, count: usize) !void {
        const device = self.device.?;
        const texture_count: u32 = @intCast(self.textures.len);
        const stage_rchit_rahit = vk.ShaderStageFlags{ .closest_hit_bit_khr = true, .any_hit_bit_khr = true };
        const stage_rgen_rchit = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true };
        const stage_rgen = vk.ShaderStageFlags{ .raygen_bit_khr = true };
        const stage_rgen_rchit_rmiss = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true, .miss_bit_khr = true };

        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .acceleration_structure_khr, .descriptor_count = 1, .stage_flags = stage_rgen_rchit },
            .{ .binding = 1, .descriptor_type = .storage_image, .descriptor_count = 1, .stage_flags = stage_rgen },
            .{ .binding = 2, .descriptor_type = .uniform_buffer, .descriptor_count = 1, .stage_flags = stage_rgen_rchit_rmiss },
            .{ .binding = 3, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = stage_rchit_rahit },
            .{ .binding = 4, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = stage_rchit_rahit },
            .{ .binding = 5, .descriptor_type = .combined_image_sampler, .descriptor_count = texture_count, .stage_flags = stage_rchit_rahit },
        };
        const binding_flags = [_]vk.DescriptorBindingFlags{
            .{},                                        .{}, .{}, .{}, .{},
            .{ .variable_descriptor_count_bit = true },
        };
        const flags_info = vk.DescriptorSetLayoutBindingFlagsCreateInfo{
            .binding_count = binding_flags.len,
            .p_binding_flags = &binding_flags,
        };
        self.desc_set_layout = try device.createDescriptorSetLayout(&.{
            .p_next = @ptrCast(&flags_info),
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);

        const count_u32: u32 = @intCast(count);
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .acceleration_structure_khr, .descriptor_count = count_u32 },
            .{ .type = .storage_image, .descriptor_count = count_u32 },
            .{ .type = .uniform_buffer, .descriptor_count = count_u32 },
            .{ .type = .combined_image_sampler, .descriptor_count = count_u32 * (1 + texture_count) },
            .{ .type = .storage_buffer, .descriptor_count = count_u32 },
        };
        self.desc_pool = try device.createDescriptorPool(&.{
            .max_sets = count_u32,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null);

        const layouts = try self.allocator.alloc(vk.DescriptorSetLayout, count);
        defer self.allocator.free(layouts);
        for (layouts) |*l| l.* = self.desc_set_layout;

        const variable_counts = try self.allocator.alloc(u32, count);
        defer self.allocator.free(variable_counts);
        for (variable_counts) |*v| v.* = texture_count;
        const variable_info = vk.DescriptorSetVariableDescriptorCountAllocateInfo{
            .descriptor_set_count = count_u32,
            .p_descriptor_counts = variable_counts.ptr,
        };

        self.desc_sets = try self.allocator.alloc(vk.DescriptorSet, count);
        try device.allocateDescriptorSets(&.{
            .p_next = @ptrCast(&variable_info),
            .descriptor_pool = self.desc_pool,
            .descriptor_set_count = count_u32,
            .p_set_layouts = layouts.ptr,
        }, self.desc_sets.ptr);

        // Prepare image infos for the texture array (stable across frames).
        const tex_infos = try self.allocator.alloc(vk.DescriptorImageInfo, self.textures.len);
        defer self.allocator.free(tex_infos);
        for (self.textures, 0..) |t, i| {
            tex_infos[i] = .{
                .sampler = t.sampler,
                .image_view = t.view,
                .image_layout = .shader_read_only_optimal,
            };
        }
        const fallback_info = tex_infos[0];

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const as_info = vk.WriteDescriptorSetAccelerationStructureKHR{
                .acceleration_structure_count = 1,
                .p_acceleration_structures = @ptrCast(&self.tlas.handle),
            };
            const storage_image_info = vk.DescriptorImageInfo{
                .sampler = .null_handle,
                .image_view = self.storage_img.view,
                .image_layout = .general,
            };
            const ubo_info = vk.DescriptorBufferInfo{
                .buffer = self.uniforms[i].buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            };
            const geom_info = vk.DescriptorBufferInfo{
                .buffer = self.geometry_nodes_buf.buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            };
            const writes = [_]vk.WriteDescriptorSet{
                .{
                    .p_next = @ptrCast(&as_info),
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 0,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .acceleration_structure_khr,
                    .p_image_info = undefined,
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 1,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_image,
                    .p_image_info = @ptrCast(&storage_image_info),
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 2,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .uniform_buffer,
                    .p_image_info = undefined,
                    .p_buffer_info = @ptrCast(&ubo_info),
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 3,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .combined_image_sampler,
                    .p_image_info = @ptrCast(&fallback_info),
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 4,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_buffer,
                    .p_image_info = undefined,
                    .p_buffer_info = @ptrCast(&geom_info),
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 5,
                    .dst_array_element = 0,
                    .descriptor_count = texture_count,
                    .descriptor_type = .combined_image_sampler,
                    .p_image_info = tex_infos.ptr,
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
            };
            device.updateDescriptorSets(&writes, null);
        }
    }

    fn createRtPipeline(self: *Ray2Renderer) !void {
        const device = self.device.?;

        const raygen = try createShaderModule(device, &raygen_spv);
        defer device.destroyShaderModule(raygen, null);
        const miss = try createShaderModule(device, &miss_spv);
        defer device.destroyShaderModule(miss, null);
        const shadow_miss = try createShaderModule(device, &shadow_miss_spv);
        defer device.destroyShaderModule(shadow_miss, null);
        const chit = try createShaderModule(device, &closest_hit_spv);
        defer device.destroyShaderModule(chit, null);
        const ahit = try createShaderModule(device, &any_hit_spv);
        defer device.destroyShaderModule(ahit, null);

        const stages = [_]vk.PipelineShaderStageCreateInfo{
            .{ .stage = .{ .raygen_bit_khr = true }, .module = raygen, .p_name = "main" },
            .{ .stage = .{ .miss_bit_khr = true }, .module = miss, .p_name = "main" },
            .{ .stage = .{ .miss_bit_khr = true }, .module = shadow_miss, .p_name = "main" },
            .{ .stage = .{ .closest_hit_bit_khr = true }, .module = chit, .p_name = "main" },
            .{ .stage = .{ .any_hit_bit_khr = true }, .module = ahit, .p_name = "main" },
        };
        const unused: u32 = vk.SHADER_UNUSED_KHR;
        const groups = [_]vk.RayTracingShaderGroupCreateInfoKHR{
            .{ .type = .general_khr, .general_shader = 0, .closest_hit_shader = unused, .any_hit_shader = unused, .intersection_shader = unused },
            .{ .type = .general_khr, .general_shader = 1, .closest_hit_shader = unused, .any_hit_shader = unused, .intersection_shader = unused },
            .{ .type = .general_khr, .general_shader = 2, .closest_hit_shader = unused, .any_hit_shader = unused, .intersection_shader = unused },
            .{ .type = .triangles_hit_group_khr, .general_shader = unused, .closest_hit_shader = 3, .any_hit_shader = 4, .intersection_shader = unused },
        };

        self.pipeline_layout = try device.createPipelineLayout(&.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&self.desc_set_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const create_info = vk.RayTracingPipelineCreateInfoKHR{
            .stage_count = stages.len,
            .p_stages = &stages,
            .group_count = groups.len,
            .p_groups = &groups,
            .max_pipeline_ray_recursion_depth = 2,
            .layout = self.pipeline_layout,
            .base_pipeline_index = -1,
        };
        var pipeline: vk.Pipeline = .null_handle;
        _ = try device.createRayTracingPipelinesKHR(
            .null_handle,
            .null_handle,
            @as([]const vk.RayTracingPipelineCreateInfoKHR, (&create_info)[0..1]),
            null,
            (&pipeline)[0..1],
        );
        self.pipeline = pipeline;
    }

    fn createShaderBindingTable(self: *Ray2Renderer) !void {
        const device = self.device.?;
        const handle_size = self.rt_props.shader_group_handle_size;
        const handle_align = self.rt_props.shader_group_handle_alignment;
        const base_align = self.rt_props.shader_group_base_alignment;
        const handle_size_aligned = alignUp(handle_size, handle_align);

        const group_count: u32 = 4; // raygen + 2 miss + 1 hit
        const handles_size = handle_size * group_count;
        const handles = try self.allocator.alloc(u8, handles_size);
        defer self.allocator.free(handles);
        try device.getRayTracingShaderGroupHandlesKHR(self.pipeline, 0, group_count, handles_size, handles.ptr);

        const raygen_size = alignUp(handle_size_aligned, base_align);
        const miss_size = alignUp(handle_size_aligned * 2, base_align);
        const hit_size = alignUp(handle_size_aligned, base_align);

        const raygen_offset: u32 = 0;
        const miss_offset: u32 = raygen_offset + raygen_size;
        const hit_offset: u32 = miss_offset + miss_size;
        const sbt_size: vk.DeviceSize = hit_offset + hit_size;

        self.sbt = try self.createBufferHostVisibleEmpty(
            sbt_size,
            .{ .shader_binding_table_bit_khr = true, .shader_device_address_bit = true, .transfer_src_bit = true },
            true,
        );

        const mapped = try device.mapMemory(self.sbt.memory, 0, sbt_size, .{});
        const dst = @as([*]u8, @ptrCast(@alignCast(mapped)));
        @memset(dst[0..sbt_size], 0);
        // raygen -> group 0
        @memcpy(dst[raygen_offset .. raygen_offset + handle_size], handles[0..handle_size]);
        // miss -> groups 1, 2 (packed back-to-back, stride=handle_size_aligned)
        @memcpy(
            dst[miss_offset .. miss_offset + handle_size],
            handles[handle_size .. 2 * handle_size],
        );
        @memcpy(
            dst[miss_offset + handle_size_aligned .. miss_offset + handle_size_aligned + handle_size],
            handles[2 * handle_size .. 3 * handle_size],
        );
        // hit -> group 3
        @memcpy(dst[hit_offset .. hit_offset + handle_size], handles[3 * handle_size .. 4 * handle_size]);
        device.unmapMemory(self.sbt.memory);

        self.sbt_raygen_region = .{
            .device_address = self.sbt.device_address + raygen_offset,
            .stride = handle_size_aligned,
            .size = handle_size_aligned,
        };
        self.sbt_miss_region = .{
            .device_address = self.sbt.device_address + miss_offset,
            .stride = handle_size_aligned,
            .size = handle_size_aligned * 2,
        };
        self.sbt_hit_region = .{
            .device_address = self.sbt.device_address + hit_offset,
            .stride = handle_size_aligned,
            .size = handle_size_aligned,
        };
        self.sbt_callable_region = .{ .device_address = 0, .stride = 0, .size = 0 };
    }
};

fn createShaderModule(device: Device, spv: []align(@alignOf(u32)) const u8) !vk.ShaderModule {
    return try device.createShaderModule(&.{
        .code_size = spv.len,
        .p_code = @ptrCast(@alignCast(spv.ptr)),
    }, null);
}

fn alignUp(value: u32, alignment: u32) u32 {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

const TransitionOpts = struct {
    src_stage: vk.PipelineStageFlags,
    dst_stage: vk.PipelineStageFlags,
    src_access: vk.AccessFlags,
    dst_access: vk.AccessFlags,
    aspect: vk.ImageAspectFlags,
};

fn transitionImage(
    device: Device,
    cmd: vk.CommandBuffer,
    image: vk.Image,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    opts: TransitionOpts,
) void {
    const barrier = vk.ImageMemoryBarrier{
        .src_access_mask = opts.src_access,
        .dst_access_mask = opts.dst_access,
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = opts.aspect,
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    };
    device.cmdPipelineBarrier(
        cmd,
        opts.src_stage,
        opts.dst_stage,
        .{},
        &.{},
        &.{},
        &.{barrier},
    );
}

fn nodeTransform(node: zgltf.Gltf.Node) Mat4 {
    if (node.matrix) |m| {
        var out: Mat4 = undefined;
        for (0..4) |c| {
            for (0..4) |r| {
                out[c][r] = @floatCast(m[c * 4 + r]);
            }
        }
        return out;
    }
    const t = math.translation(
        @floatCast(node.translation[0]),
        @floatCast(node.translation[1]),
        @floatCast(node.translation[2]),
    );
    const r = math.fromQuaternion(
        @floatCast(node.rotation[0]),
        @floatCast(node.rotation[1]),
        @floatCast(node.rotation[2]),
        @floatCast(node.rotation[3]),
    );
    const s = math.scaling(
        @floatCast(node.scale[0]),
        @floatCast(node.scale[1]),
        @floatCast(node.scale[2]),
    );
    const rs = math.multiply(&r, &s);
    return math.multiply(&t, &rs);
}

fn transformNormal(m: Mat4, nx: f32, ny: f32, nz: f32) [3]f32 {
    // Use the 3x3 rotation/scale part; for pure rigid+uniform-scale transforms
    // this yields correct orientation after renormalization.
    const ox = m[0][0] * nx + m[1][0] * ny + m[2][0] * nz;
    const oy = m[0][1] * nx + m[1][1] * ny + m[2][1] * nz;
    const oz = m[0][2] * nx + m[1][2] * ny + m[2][2] * nz;
    const l = @sqrt(ox * ox + oy * oy + oz * oz);
    if (l > 0) return .{ ox / l, oy / l, oz / l };
    return .{ 0, 0, 0 };
}

fn vkTransformFromMat4(m: Mat4) [12]f32 {
    // VkTransformMatrixKHR is 3x4 row-major: [col][row] column-major → transpose.
    return .{
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
    };
}

fn decodeAndUpload(
    allocator: std.mem.Allocator,
    manager: *const ash.Manager,
    cmd_ctx: *ash.CommandContext,
    encoded: []const u8,
) !ash.ImageResource {
    return try ash.createTextureFromEncoded(allocator, manager, cmd_ctx, encoded, .{});
}

fn perspectiveZO(y_fov: f32, aspect: f32, near: f32, far: f32) Mat4 {
    // Vulkan 0..1 depth convention, matching Sascha Willems' setPerspectiveZO in the Go reference.
    const f = 1.0 / @tan(y_fov / 2.0);
    return .{
        .{ f / aspect, 0, 0, 0 },
        .{ 0, f, 0, 0 },
        .{ 0, 0, far / (near - far), -1 },
        .{ 0, 0, (near * far) / (near - far), 0 },
    };
}
