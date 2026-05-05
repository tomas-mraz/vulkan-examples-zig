const ash = @import("ash");
const std = @import("std");
const zgltf = @import("zgltf");
const vk = ash.vk;
const math = ash.math;

const raygen_spv align(@alignOf(u32)) = @embedFile("raygen_shader").*;
const miss_spv align(@alignOf(u32)) = @embedFile("miss_shader").*;
const closest_hit_spv align(@alignOf(u32)) = @embedFile("closest_hit_shader").*;

const Allocator = std.mem.Allocator;
const Device = vk.DeviceProxy;
const Mat4 = math.Mat4;

// Layout the closesthit.rchit unpack() expects: 3x vec4 per vertex.
//   d0 = (pos.x, pos.y, pos.z, normal.x)
//   d1 = (normal.y, normal.z, uv.x, uv.y)
//   d2 = (color.r, color.g, color.b, color.a)
// vertexSize is read from the UBO and divided by 16 to get the vec4 stride.
const floats_per_vertex: usize = 12;
const vertex_stride_bytes: vk.DeviceSize = floats_per_vertex * @sizeOf(f32);

const UniformData = extern struct {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    light_pos: [4]f32,
    vertex_size: i32,
};

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

pub const Ray4Renderer = struct {
    allocator: Allocator,

    manager: ?*ash.Manager = null,
    cmd_ctx: ?*ash.CommandContext = null,
    device: ?Device = null,

    rt_props: vk.PhysicalDeviceRayTracingPipelinePropertiesKHR = undefined,

    scene_vertex_buf: Buffer = .{},
    scene_index_buf: Buffer = .{},
    triangle_count: u32 = 0,
    vertex_count: u32 = 0,

    blas: AccelStruct = .{},
    tlas: AccelStruct = .{},

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

    start_time: f64 = 0,

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: Allocator) Ray4Renderer {
        return .{ .allocator = allocator };
    }

    pub fn createOnce(self: *Ray4Renderer, session: *ash.Session) !void {
        self.manager = &session.manager.?;
        self.cmd_ctx = &session.cmd_ctx.?;
        self.device = self.manager.?.device.?;

        try self.queryRtProperties();

        try self.loadModel("assets/reflection_scene.gltf");
        std.log.info("loaded scene: {} vertices, {} triangles", .{ self.vertex_count, self.triangle_count });

        try self.buildBlas();
        try self.buildTlas();

        self.start_time = ash.glfw.getTime();

        self.once_built = true;
    }

    pub fn destroyOnce(self: *Ray4Renderer) void {
        if (!self.once_built) return;
        const device = self.device.?;
        self.tlas.deinit(device);
        self.blas.deinit(device);
        self.scene_index_buf.deinit(device);
        self.scene_vertex_buf.deinit(device);
        self.once_built = false;
    }

    pub fn createSized(self: *Ray4Renderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const swap_len = session.swapchain.?.imageCount();
        const display_format = session.swapchain.?.surface_format.format;

        try self.createStorageImage(extent, display_format);
        try self.createUniformBuffers(swap_len);
        try self.createDescriptorSet(swap_len);
        try self.createRtPipeline();
        try self.createShaderBindingTable();

        self.sized_built = true;
    }

    pub fn destroySized(self: *Ray4Renderer) void {
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

    pub fn draw(self: *Ray4Renderer, session: *ash.Session, frame: *const ash.Frame) !void {
        _ = session;
        const device = self.device.?;

        const aspect: f32 = @as(f32, @floatFromInt(frame.extent.width)) / @as(f32, @floatFromInt(frame.extent.height));
        const proj = perspectiveZO(math.degreesToRadians(60.0), aspect, 0.1, 512.0);

        const elapsed_seconds = @as(f32, @floatCast(ash.glfw.getTime() - self.start_time));
        const cam_yaw = elapsed_seconds * 11.25;
        const rot = math.rotationAxis(0, 1, 0, math.degreesToRadians(cam_yaw));
        // Camera orbits the world Y axis at radius |z|. C++ sample places it at -2,
        // but spheres sit on a ring of XZ radius ~1.5..2.25, so an r=2 orbit threads
        // through them and looks like the camera lurches past each sphere. Move it
        // out to r=4 so the entire scene stays comfortably in front of the camera.
        const view = math.translation(0.0, 0.5, -4.0);
        const rotated_view = math.multiply(&view, &rot);

        // C++ Sascha Willems sample: lightPos is driven by timer*360°, where
        // timer wraps every (1 / (timerSpeed * 0.5)) = 1 / 0.125 = 8 seconds.
        // Match that 8-second light orbit period.
        const light_angle = elapsed_seconds * (std.math.tau / 8.0);

        const ubo = UniformData{
            .view_inverse = math.invert(rotated_view),
            .proj_inverse = math.invert(proj),
            .light_pos = .{
                @cos(light_angle) * 40.0,
                -20.0 + @sin(light_angle) * 20.0,
                25.0 + @sin(light_angle) * 5.0,
                0.0,
            },
            .vertex_size = @intCast(vertex_stride_bytes),
        };

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

    // --- model loading: merge every primitive into one big vertex+index buffer
    // (mirrors the vkglTF::Model path used by the C++ sample) ---

    fn loadModel(self: *Ray4Renderer, path: []const u8) !void {
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

        var binary: []align(@alignOf(f32)) const u8 = undefined;
        if (std.mem.startsWith(u8, bin_uri, "data:")) {
            binary = try self.decodeEmbeddedBuffer(bin_uri);
        } else {
            const base_dir = std.fs.path.dirname(path) orelse ".";
            const bin_path = try std.fs.path.join(self.allocator, &.{ base_dir, bin_uri });
            defer self.allocator.free(bin_path);
            const raw_bin = try std.Io.Dir.cwd().readFileAlloc(io, bin_path, self.allocator, .unlimited);
            defer self.allocator.free(raw_bin);
            const aligned = try self.allocator.alignedAlloc(u8, .of(f32), raw_bin.len);
            @memcpy(aligned, raw_bin);
            binary = aligned;
        }
        defer self.allocator.free(binary);

        if (gltf.data.scenes.len == 0) return error.GltfNoScenes;
        const scene = gltf.data.scenes[0];
        const roots = scene.nodes orelse return error.GltfSceneEmpty;

        var verts = std.ArrayList(f32).empty;
        defer verts.deinit(self.allocator);
        var indices = std.ArrayList(u32).empty;
        defer indices.deinit(self.allocator);

        for (roots) |root| {
            try self.visitNode(&gltf, binary, root, math.identity(), &verts, &indices);
        }

        if (indices.items.len == 0) return error.GltfNoPrimitives;

        const rt_usage = vk.BufferUsageFlags{
            .shader_device_address_bit = true,
            .acceleration_structure_build_input_read_only_bit_khr = true,
            .storage_buffer_bit = true,
        };
        self.scene_vertex_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(verts.items), rt_usage, true);
        self.scene_index_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(indices.items), rt_usage, true);
        self.vertex_count = @intCast(verts.items.len / floats_per_vertex);
        self.triangle_count = @intCast(indices.items.len / 3);
    }

    fn decodeEmbeddedBuffer(self: *Ray4Renderer, uri: []const u8) ![]align(@alignOf(f32)) const u8 {
        const base64_start = std.mem.indexOfScalar(u8, uri, ',') orelse return error.InvalidDataUri;
        const base64_data = uri[base64_start + 1 ..];
        const decoder = std.base64.standard.Decoder;
        const decoded_len = try decoder.calcSizeForSlice(base64_data);
        const decoded = try self.allocator.alignedAlloc(u8, .of(f32), decoded_len);
        errdefer self.allocator.free(decoded);
        try decoder.decode(decoded, base64_data);
        return decoded;
    }

    fn visitNode(
        self: *Ray4Renderer,
        gltf: *const zgltf.Gltf,
        binary: []align(@alignOf(f32)) const u8,
        node_index: usize,
        parent_transform: Mat4,
        verts: *std.ArrayList(f32),
        indices: *std.ArrayList(u32),
    ) !void {
        const node = gltf.data.nodes[node_index];
        const local = nodeTransform(node);
        const world = math.multiply(&parent_transform, &local);

        if (node.mesh) |mesh_index| {
            const mesh = gltf.data.meshes[mesh_index];
            for (mesh.primitives) |prim| {
                try self.appendPrimitive(gltf, binary, prim, world, verts, indices);
            }
        }
        for (node.children) |child| {
            try self.visitNode(gltf, binary, child, world, verts, indices);
        }
    }

    fn appendPrimitive(
        self: *Ray4Renderer,
        gltf: *const zgltf.Gltf,
        binary: []align(@alignOf(f32)) const u8,
        prim: zgltf.Gltf.Primitive,
        world: Mat4,
        verts: *std.ArrayList(f32),
        indices: *std.ArrayList(u32),
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
        const ia = prim.indices orelse return error.GltfMissingIndices;

        const pos_acc = gltf.data.accessors[pa];
        const norm_acc = gltf.data.accessors[na];
        if (pos_acc.count != norm_acc.count) return error.GltfAttributeMismatch;
        const has_uv = uv_idx != null;

        // Resolve material color (and the white = reflector convention).
        var color: [4]f32 = .{ 1, 1, 1, 1 };
        if (prim.material) |mi| {
            if (mi < gltf.data.materials.len) {
                const m = gltf.data.materials[mi];
                color = .{
                    @floatCast(m.metallic_roughness.base_color_factor[0]),
                    @floatCast(m.metallic_roughness.base_color_factor[1]),
                    @floatCast(m.metallic_roughness.base_color_factor[2]),
                    @floatCast(m.metallic_roughness.base_color_factor[3]),
                };
            }
        }

        const base_vertex: u32 = @intCast(verts.items.len / floats_per_vertex);

        var pi = pos_acc.iterator(f32, @constCast(gltf), binary);
        var ni = norm_acc.iterator(f32, @constCast(gltf), binary);
        var ui_opt: ?@TypeOf(pi) = null;
        if (has_uv) {
            const uv_acc = gltf.data.accessors[uv_idx.?];
            ui_opt = uv_acc.iterator(f32, @constCast(gltf), binary);
        }

        try verts.ensureUnusedCapacity(self.allocator, pos_acc.count * floats_per_vertex);
        var k: usize = 0;
        while (k < pos_acc.count) : (k += 1) {
            const p = pi.next() orelse return error.GltfPositionShort;
            const n = ni.next() orelse return error.GltfNormalShort;

            const tp = transformPoint(world, p[0], p[1], p[2]);
            const tn = transformNormal(world, n[0], n[1], n[2]);

            var u: [2]f32 = .{ 0, 0 };
            if (ui_opt) |*it| {
                if (it.next()) |uv| {
                    u = .{ uv[0], uv[1] };
                }
            }

            // Mirror vkglTF::FileLoadingFlags::FlipY used by the C++ sample:
            // negate Y on positions and normals so the scene matches the
            // camera setup (camera at world +Y looking down at the floor).
            verts.appendSliceAssumeCapacity(&.{
                tp[0], -tp[1], tp[2],
                tn[0], -tn[1], tn[2],
                u[0],  u[1],
                color[0], color[1], color[2], color[3],
            });
        }

        const idx_acc = gltf.data.accessors[ia];
        try indices.ensureUnusedCapacity(self.allocator, idx_acc.count);
        switch (idx_acc.component_type) {
            .unsigned_short => {
                var it = idx_acc.iterator(u16, @constCast(gltf), binary);
                while (it.next()) |s| indices.appendAssumeCapacity(@as(u32, s[0]) + base_vertex);
            },
            .unsigned_integer => {
                var it = idx_acc.iterator(u32, @constCast(gltf), binary);
                while (it.next()) |s| indices.appendAssumeCapacity(s[0] + base_vertex);
            },
            .unsigned_byte => {
                var it = idx_acc.iterator(u8, @constCast(gltf), binary);
                while (it.next()) |s| indices.appendAssumeCapacity(@as(u32, s[0]) + base_vertex);
            },
            else => return error.GltfUnsupportedIndexType,
        }
    }

    // --- RT setup helpers ---

    fn queryRtProperties(self: *Ray4Renderer) !void {
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
        self: *Ray4Renderer,
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
        self: *Ray4Renderer,
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
        self: *Ray4Renderer,
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

    fn buildBlas(self: *Ray4Renderer) !void {
        const device = self.device.?;

        const triangles = vk.AccelerationStructureGeometryTrianglesDataKHR{
            .vertex_format = .r32g32b32_sfloat,
            .vertex_data = .{ .device_address = self.scene_vertex_buf.device_address },
            .vertex_stride = vertex_stride_bytes,
            .max_vertex = if (self.vertex_count == 0) 0 else self.vertex_count - 1,
            .index_type = .uint32,
            .index_data = .{ .device_address = self.scene_index_buf.device_address },
            .transform_data = .{ .device_address = 0 },
        };
        const geometry = vk.AccelerationStructureGeometryKHR{
            .geometry_type = .triangles_khr,
            .geometry = .{ .triangles = triangles },
            .flags = .{ .opaque_bit_khr = true },
        };

        const flags = vk.BuildAccelerationStructureFlagsKHR{ .prefer_fast_trace_bit_khr = true };
        var build_info_sz = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .bottom_level_khr,
            .flags = flags,
            .mode = .build_khr,
            .geometry_count = 1,
            .p_geometries = @ptrCast(&geometry),
            .scratch_data = .{ .device_address = 0 },
        };
        const primitive_count: u32 = self.triangle_count;
        var sizes = vk.AccelerationStructureBuildSizesInfoKHR{
            .acceleration_structure_size = 0,
            .update_scratch_size = 0,
            .build_scratch_size = 0,
        };
        device.getAccelerationStructureBuildSizesKHR(.device_khr, &build_info_sz, @ptrCast(&primitive_count), &sizes);

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
            .geometry_count = 1,
            .p_geometries = @ptrCast(&geometry),
            .scratch_data = .{ .device_address = scratch.device_address },
        };
        const range = vk.AccelerationStructureBuildRangeInfoKHR{
            .primitive_count = primitive_count,
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

        self.blas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{
            .acceleration_structure = self.blas.handle,
        });
    }

    fn buildTlas(self: *Ray4Renderer) !void {
        const device = self.device.?;

        const instance = vk.AccelerationStructureInstanceKHR{
            .transform = .{ .matrix = .{
                .{ 1, 0, 0, 0 },
                .{ 0, 1, 0, 0 },
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

    fn createStorageImage(self: *Ray4Renderer, extent: vk.Extent2D, format: vk.Format) !void {
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

    fn createUniformBuffers(self: *Ray4Renderer, count: usize) !void {
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

    fn createDescriptorSet(self: *Ray4Renderer, count: usize) !void {
        const device = self.device.?;

        const stage_rchit = vk.ShaderStageFlags{ .closest_hit_bit_khr = true };
        const stage_rgen_rchit = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true };
        const stage_rgen = vk.ShaderStageFlags{ .raygen_bit_khr = true };
        const stage_rgen_rchit_rmiss = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true, .miss_bit_khr = true };

        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .acceleration_structure_khr, .descriptor_count = 1, .stage_flags = stage_rgen_rchit },
            .{ .binding = 1, .descriptor_type = .storage_image, .descriptor_count = 1, .stage_flags = stage_rgen },
            .{ .binding = 2, .descriptor_type = .uniform_buffer, .descriptor_count = 1, .stage_flags = stage_rgen_rchit_rmiss },
            .{ .binding = 3, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = stage_rchit },
            .{ .binding = 4, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = stage_rchit },
        };
        self.desc_set_layout = try device.createDescriptorSetLayout(&.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);

        const count_u32: u32 = @intCast(count);
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .acceleration_structure_khr, .descriptor_count = count_u32 },
            .{ .type = .storage_image, .descriptor_count = count_u32 },
            .{ .type = .uniform_buffer, .descriptor_count = count_u32 },
            .{ .type = .storage_buffer, .descriptor_count = count_u32 * 2 },
        };
        self.desc_pool = try device.createDescriptorPool(&.{
            .max_sets = count_u32,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null);

        const layouts = try self.allocator.alloc(vk.DescriptorSetLayout, count);
        defer self.allocator.free(layouts);
        for (layouts) |*l| l.* = self.desc_set_layout;

        self.desc_sets = try self.allocator.alloc(vk.DescriptorSet, count);
        try device.allocateDescriptorSets(&.{
            .descriptor_pool = self.desc_pool,
            .descriptor_set_count = count_u32,
            .p_set_layouts = layouts.ptr,
        }, self.desc_sets.ptr);

        const storage_image_info = vk.DescriptorImageInfo{
            .sampler = .null_handle,
            .image_view = self.storage_img.view,
            .image_layout = .general,
        };
        const vertex_info = vk.DescriptorBufferInfo{
            .buffer = self.scene_vertex_buf.buffer,
            .offset = 0,
            .range = vk.WHOLE_SIZE,
        };
        const index_info = vk.DescriptorBufferInfo{
            .buffer = self.scene_index_buf.buffer,
            .offset = 0,
            .range = vk.WHOLE_SIZE,
        };

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const as_info = vk.WriteDescriptorSetAccelerationStructureKHR{
                .acceleration_structure_count = 1,
                .p_acceleration_structures = @ptrCast(&self.tlas.handle),
            };
            const ubo_info = vk.DescriptorBufferInfo{
                .buffer = self.uniforms[i].buffer,
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
                    .descriptor_type = .storage_buffer,
                    .p_image_info = undefined,
                    .p_buffer_info = @ptrCast(&vertex_info),
                    .p_texel_buffer_view = undefined,
                },
                .{
                    .dst_set = self.desc_sets[i],
                    .dst_binding = 4,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_buffer,
                    .p_image_info = undefined,
                    .p_buffer_info = @ptrCast(&index_info),
                    .p_texel_buffer_view = undefined,
                },
            };
            device.updateDescriptorSets(&writes, null);
        }
    }

    fn createRtPipeline(self: *Ray4Renderer) !void {
        const device = self.device.?;

        const raygen = try createShaderModule(device, &raygen_spv);
        defer device.destroyShaderModule(raygen, null);
        const miss = try createShaderModule(device, &miss_spv);
        defer device.destroyShaderModule(miss, null);
        const chit = try createShaderModule(device, &closest_hit_spv);
        defer device.destroyShaderModule(chit, null);

        const stages = [_]vk.PipelineShaderStageCreateInfo{
            .{ .stage = .{ .raygen_bit_khr = true }, .module = raygen, .p_name = "main" },
            .{ .stage = .{ .miss_bit_khr = true }, .module = miss, .p_name = "main" },
            .{ .stage = .{ .closest_hit_bit_khr = true }, .module = chit, .p_name = "main" },
        };
        const unused: u32 = vk.SHADER_UNUSED_KHR;
        const groups = [_]vk.RayTracingShaderGroupCreateInfoKHR{
            .{ .type = .general_khr, .general_shader = 0, .closest_hit_shader = unused, .any_hit_shader = unused, .intersection_shader = unused },
            .{ .type = .general_khr, .general_shader = 1, .closest_hit_shader = unused, .any_hit_shader = unused, .intersection_shader = unused },
            .{ .type = .triangles_hit_group_khr, .general_shader = unused, .closest_hit_shader = 2, .any_hit_shader = unused, .intersection_shader = unused },
        };

        self.pipeline_layout = try device.createPipelineLayout(&.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&self.desc_set_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const max_recursion: u32 = @min(@as(u32, 4), self.rt_props.max_ray_recursion_depth);
        const create_info = vk.RayTracingPipelineCreateInfoKHR{
            .stage_count = stages.len,
            .p_stages = &stages,
            .group_count = groups.len,
            .p_groups = &groups,
            .max_pipeline_ray_recursion_depth = max_recursion,
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

    fn createShaderBindingTable(self: *Ray4Renderer) !void {
        const device = self.device.?;
        const handle_size = self.rt_props.shader_group_handle_size;
        const handle_align = self.rt_props.shader_group_handle_alignment;
        const base_align = self.rt_props.shader_group_base_alignment;
        const handle_size_aligned = alignUp(handle_size, handle_align);

        const group_count: u32 = 3;
        const handles_size = handle_size * group_count;
        const handles = try self.allocator.alloc(u8, handles_size);
        defer self.allocator.free(handles);
        try device.getRayTracingShaderGroupHandlesKHR(self.pipeline, 0, group_count, handles_size, handles.ptr);

        const raygen_size = alignUp(handle_size_aligned, base_align);
        const miss_size = alignUp(handle_size_aligned, base_align);
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
        @memcpy(dst[raygen_offset .. raygen_offset + handle_size], handles[0..handle_size]);
        @memcpy(
            dst[miss_offset .. miss_offset + handle_size],
            handles[handle_size .. 2 * handle_size],
        );
        @memcpy(dst[hit_offset .. hit_offset + handle_size], handles[2 * handle_size .. 3 * handle_size]);
        device.unmapMemory(self.sbt.memory);

        self.sbt_raygen_region = .{
            .device_address = self.sbt.device_address + raygen_offset,
            .stride = handle_size_aligned,
            .size = handle_size_aligned,
        };
        self.sbt_miss_region = .{
            .device_address = self.sbt.device_address + miss_offset,
            .stride = handle_size_aligned,
            .size = handle_size_aligned,
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

fn transformPoint(m: Mat4, x: f32, y: f32, z: f32) [3]f32 {
    const ox = m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0];
    const oy = m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1];
    const oz = m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2];
    return .{ ox, oy, oz };
}

fn transformNormal(m: Mat4, nx: f32, ny: f32, nz: f32) [3]f32 {
    const ox = m[0][0] * nx + m[1][0] * ny + m[2][0] * nz;
    const oy = m[0][1] * nx + m[1][1] * ny + m[2][1] * nz;
    const oz = m[0][2] * nx + m[1][2] * ny + m[2][2] * nz;
    const l = @sqrt(ox * ox + oy * oy + oz * oz);
    if (l > 0) return .{ ox / l, oy / l, oz / l };
    return .{ 0, 0, 0 };
}

fn perspectiveZO(y_fov: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const f = 1.0 / @tan(y_fov / 2.0);
    return .{
        .{ f / aspect, 0, 0, 0 },
        .{ 0, f, 0, 0 },
        .{ 0, 0, far / (near - far), -1 },
        .{ 0, 0, (near * far) / (near - far), 0 },
    };
}
