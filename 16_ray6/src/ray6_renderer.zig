const ash = @import("ash");
const std = @import("std");
const vk = ash.vk;
const math = ash.math;

const raygen_spv align(@alignOf(u32)) = @embedFile("raygen_shader").*;
const miss_spv align(@alignOf(u32)) = @embedFile("miss_shader").*;
const closest_hit_spv align(@alignOf(u32)) = @embedFile("closest_hit_shader").*;

const Allocator = std.mem.Allocator;
const Device = vk.DeviceProxy;
const Mat4 = math.Mat4;

const floats_per_vertex: usize = 12;
const vertex_stride_bytes: vk.DeviceSize = floats_per_vertex * @sizeOf(f32);

const UniformData = extern struct {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    frame_index: u32,
    accum_count: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
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

pub const Ray6Renderer = struct {
    allocator: Allocator,
    host: *ash.DesktopHost,

    manager: ?*ash.Manager = null,
    cmd_ctx: ?*ash.CommandContext = null,
    device: ?Device = null,

    rt_props: vk.PhysicalDeviceRayTracingPipelinePropertiesKHR = undefined,

    vertex_buf: Buffer = .{},
    index_buf: Buffer = .{},
    triangle_count: u32 = 0,
    vertex_count: u32 = 0,

    blas: AccelStruct = .{},
    tlas: AccelStruct = .{},

    accum_img: StorageImage = .{},
    display_img: StorageImage = .{},

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

    frame_index: u32 = 0,
    accum_count: u32 = 0,
    prev_view: Mat4 = undefined,
    prev_view_valid: bool = false,

    cam_pos: [3]f32 = .{ 0.0, 0.0, 3.0 },
    cam_yaw: f32 = 0.0,
    cam_pitch: f32 = 0.0,
    cursor_captured: bool = false,
    cursor_initialized: bool = false,
    prev_cursor_x: f64 = 0.0,
    prev_cursor_y: f64 = 0.0,
    last_time: f64 = 0.0,

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: Allocator, host: *ash.DesktopHost) Ray6Renderer {
        return .{ .allocator = allocator, .host = host };
    }

    pub fn createOnce(self: *Ray6Renderer, session: *ash.Session) !void {
        self.manager = &session.manager.?;
        self.cmd_ctx = &session.cmd_ctx.?;
        self.device = self.manager.?.device.?;

        try self.queryRtProperties();
        try self.createCornellBoxGeometry();
        try self.buildBlas();
        try self.buildTlas();

        self.once_built = true;
    }

    pub fn destroyOnce(self: *Ray6Renderer) void {
        if (!self.once_built) return;
        const device = self.device.?;
        self.tlas.deinit(device);
        self.blas.deinit(device);
        self.index_buf.deinit(device);
        self.vertex_buf.deinit(device);
        self.once_built = false;
    }

    pub fn createSized(self: *Ray6Renderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const swap_len = session.swapchain.?.imageCount();
        const display_format = session.swapchain.?.surface_format.format;

        try self.createAccumulationImage(extent);
        try self.createDisplayImage(extent, display_format);
        try self.initImageLayouts();

        try self.createUniformBuffers(swap_len);
        try self.createDescriptorSet(swap_len);
        try self.createRtPipeline();
        try self.createShaderBindingTable();

        self.frame_index = 0;
        self.accum_count = 0;
        self.prev_view_valid = false;

        self.sized_built = true;
    }

    pub fn destroySized(self: *Ray6Renderer) void {
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
        self.display_img.deinit(device);
        self.accum_img.deinit(device);
        self.sized_built = false;
    }

    pub fn draw(self: *Ray6Renderer, session: *ash.Session, frame: *const ash.Frame) !void {
        _ = session;
        const device = self.device.?;

        const aspect: f32 = @as(f32, @floatFromInt(frame.extent.width)) / @as(f32, @floatFromInt(frame.extent.height));
        const proj = perspectiveZO(math.degreesToRadians(40.0), aspect, 0.1, 100.0);

        self.updateCamera();
        const view = self.cameraView();

        if (!self.prev_view_valid or !matEqual(&self.prev_view, &view)) {
            self.accum_count = 0;
            self.prev_view = view;
            self.prev_view_valid = true;
        }

        const ubo = UniformData{
            .view_inverse = math.invert(view),
            .proj_inverse = math.invert(proj),
            .frame_index = self.frame_index,
            .accum_count = self.accum_count,
        };

        const ub = &self.uniforms[frame.image_index];
        const mapped = try device.mapMemory(ub.memory, 0, vk.WHOLE_SIZE, .{});
        @memcpy(@as([*]u8, @ptrCast(@alignCast(mapped)))[0..@sizeOf(UniformData)], std.mem.asBytes(&ubo));
        device.unmapMemory(ub.memory);

        // Sync the accumulation image: previous frame's writes must complete
        // before this frame's read-modify-write. Layout stays in .general.
        accumBarrier(device, frame.cmd, self.accum_img.image);

        // Display image: contents are fully overwritten; just transition to general.
        transitionImage(device, frame.cmd, self.display_img.image, .undefined, .general, .{
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

        transitionImage(device, frame.cmd, self.display_img.image, .general, .transfer_src_optimal, .{
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
            self.display_img.image,
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

        self.frame_index +%= 1;
        self.accum_count +%= 1;
    }

    // --- FPS-style free camera (WSAD + mouse-look) ---

    fn updateCamera(self: *Ray6Renderer) void {
        const window = self.host.window orelse return;

        const now = ash.glfw.getTime();
        if (self.last_time == 0.0) self.last_time = now;
        var dt: f32 = @floatCast(now - self.last_time);
        self.last_time = now;
        if (dt > 0.1) dt = 0.1; // clamp huge stalls (e.g. window drag)

        if (window.getKey(.escape) == .press) {
            window.setShouldClose(true);
        }

        // Capture cursor lazily so the user can still interact with the window
        // before the first draw lands.
        if (!self.cursor_captured) {
            window.setInputModeCursor(.disabled);
            self.cursor_captured = true;
            self.cursor_initialized = false;
        }

        const cursor = window.getCursorPos();
        if (!self.cursor_initialized) {
            self.prev_cursor_x = cursor.xpos;
            self.prev_cursor_y = cursor.ypos;
            self.cursor_initialized = true;
        }
        const dx: f32 = @floatCast(cursor.xpos - self.prev_cursor_x);
        const dy: f32 = @floatCast(cursor.ypos - self.prev_cursor_y);
        self.prev_cursor_x = cursor.xpos;
        self.prev_cursor_y = cursor.ypos;

        const mouse_sensitivity: f32 = 0.0025;
        self.cam_yaw -= dx * mouse_sensitivity;
        self.cam_pitch -= dy * mouse_sensitivity;
        const pitch_limit: f32 = math.degreesToRadians(89.0);
        if (self.cam_pitch > pitch_limit) self.cam_pitch = pitch_limit;
        if (self.cam_pitch < -pitch_limit) self.cam_pitch = -pitch_limit;

        // World-space movement basis from yaw only — keeps WSAD on the
        // ground plane regardless of where the camera is looking.
        const sy = @sin(self.cam_yaw);
        const cy = @cos(self.cam_yaw);
        const fwd = [3]f32{ -sy, 0.0, -cy };
        const right = [3]f32{ cy, 0.0, -sy };

        var move_speed: f32 = 1.5;
        if (window.getKey(.left_shift) == .press or window.getKey(.right_shift) == .press) {
            move_speed *= 3.0;
        }
        const step = move_speed * dt;

        if (window.getKey(.w) == .press) {
            self.cam_pos[0] += fwd[0] * step;
            self.cam_pos[1] += fwd[1] * step;
            self.cam_pos[2] += fwd[2] * step;
        }
        if (window.getKey(.s) == .press) {
            self.cam_pos[0] -= fwd[0] * step;
            self.cam_pos[1] -= fwd[1] * step;
            self.cam_pos[2] -= fwd[2] * step;
        }
        if (window.getKey(.d) == .press) {
            self.cam_pos[0] += right[0] * step;
            self.cam_pos[1] += right[1] * step;
            self.cam_pos[2] += right[2] * step;
        }
        if (window.getKey(.a) == .press) {
            self.cam_pos[0] -= right[0] * step;
            self.cam_pos[1] -= right[1] * step;
            self.cam_pos[2] -= right[2] * step;
        }
        if (window.getKey(.space) == .press) self.cam_pos[1] += step;
        if (window.getKey(.left_control) == .press) self.cam_pos[1] -= step;
    }

    fn cameraView(self: *const Ray6Renderer) Mat4 {
        const cy = @cos(self.cam_yaw);
        const sy = @sin(self.cam_yaw);
        const cp = @cos(self.cam_pitch);
        const sp = @sin(self.cam_pitch);
        // yaw=0, pitch=0 → looking down -Z, matching the original static view.
        const fwd = math.Vec3{ .x = -sy * cp, .y = sp, .z = -cy * cp };
        const eye = math.Vec3{ .x = self.cam_pos[0], .y = self.cam_pos[1], .z = self.cam_pos[2] };
        const center = math.Vec3{ .x = eye.x + fwd.x, .y = eye.y + fwd.y, .z = eye.z + fwd.z };
        return math.lookAt(eye, center, .{ .x = 0, .y = 1, .z = 0 });
    }

    // --- Cornell box geometry ---

    fn createCornellBoxGeometry(self: *Ray6Renderer) !void {
        var verts = std.ArrayList(f32).empty;
        defer verts.deinit(self.allocator);
        var indices = std.ArrayList(u32).empty;
        defer indices.deinit(self.allocator);

        const white = [3]f32{ 0.73, 0.73, 0.73 };
        const red = [3]f32{ 0.65, 0.05, 0.05 };
        const green = [3]f32{ 0.12, 0.45, 0.15 };
        const blue = [3]f32{ 0.10, 0.20, 0.65 };
        const black = [3]f32{ 0.0, 0.0, 0.0 };

        // Box from (-1,-1,-1) to (1,1,1), front (+Z) wall omitted so the
        // camera at z=+3 looks into the open box.
        // Floor (y = -1, normal up) — metallic glossy, F0 = white, roughness 0.6.
        try appendQuad(&verts, &indices, self.allocator, .{ -1, -1, -1 }, .{ 1, -1, -1 }, .{ 1, -1, 1 }, .{ -1, -1, 1 }, .{ 0, 1, 0 }, white, -1.65);
        // Ceiling (y = 1, normal down) — pure black Lambert (absorber).
        try appendQuad(&verts, &indices, self.allocator, .{ -1, 1, 1 }, .{ 1, 1, 1 }, .{ 1, 1, -1 }, .{ -1, 1, -1 }, .{ 0, -1, 0 }, black, 0.0);
        // Back wall (z = -1, normal +Z) — glossy roughness 0.7.
        try appendQuad(&verts, &indices, self.allocator, .{ -1, -1, -1 }, .{ -1, 1, -1 }, .{ 1, 1, -1 }, .{ 1, -1, -1 }, .{ 0, 0, 1 }, white, -1.65);
        // Left wall (x = -1, red, normal +X) — glossy roughness 0.7.
        try appendQuad(&verts, &indices, self.allocator, .{ -1, -1, 1 }, .{ -1, 1, 1 }, .{ -1, 1, -1 }, .{ -1, -1, -1 }, .{ 1, 0, 0 }, red, -1.65);
        // Right wall (x = 1, green, normal -X) — glossy roughness 0.7.
        try appendQuad(&verts, &indices, self.allocator, .{ 1, -1, -1 }, .{ 1, 1, -1 }, .{ 1, 1, 1 }, .{ 1, -1, 1 }, .{ -1, 0, 0 }, green, -1.65);
        // Light: small emissive quad just below the ceiling.
        try appendQuad(&verts, &indices, self.allocator, .{ -0.3, 0.999, 0.3 }, .{ 0.3, 0.999, 0.3 }, .{ 0.3, 0.999, -0.3 }, .{ -0.3, 0.999, -0.3 }, .{ 0, -1, 0 }, .{ 1, 1, 1 }, 8.0);
        // Tall box near the red wall — glossy roughness 0.7.
        try appendBox(&verts, &indices, self.allocator, .{ -1.0, -1.0, -0.35 }, .{ -0.5, 0.3, 0.15 }, white, true, -1.65);
        // Blue overlay on the +X face (the side facing the green wall). Offset by 0.001
        // outward to win the coplanar tie-break against the white face underneath.
        try appendQuad(
            &verts,
            &indices,
            self.allocator,
            .{ -0.499, -1.0,  0.15 },
            .{ -0.499,  0.3,  0.15 },
            .{ -0.499,  0.3, -0.35 },
            .{ -0.499, -1.0, -0.35 },
            .{ 1, 0, 0 },
            blue,
            -1.65,
        );
        // Short box near the green wall — glossy sides (r=0.7), separate glossy top below (r=0.15).
        const sb_min = [3]f32{ 0.15, -1.0, -0.05 };
        const sb_max = [3]f32{ 0.65, -0.35, 0.45 };
        try appendBox(&verts, &indices, self.allocator, sb_min, sb_max, white, false, -1.65);
        // Glossy top quad: emission = -(1 + roughness). roughness 0 ⇒ -1.0 = perfect mirror,
        // 0.15 here ⇒ tight specular lobe with visible blur. F0 = albedo (metallic-tinted).
        try appendQuad(
            &verts,
            &indices,
            self.allocator,
            .{ sb_min[0], sb_max[1], sb_min[2] },
            .{ sb_max[0], sb_max[1], sb_min[2] },
            .{ sb_max[0], sb_max[1], sb_max[2] },
            .{ sb_min[0], sb_max[1], sb_max[2] },
            .{ 0, 1, 0 },
            .{ 1, 1, 1 },
            -1.15,
        );
        // Glass sphere — sentinel emission = -10 * IOR. IOR = 1.5 ⇒ emission = -15.
        // Sits on the floor in the front-center area, between the two boxes.
        try appendSphere(&verts, &indices, self.allocator, .{ -0.175, -0.65, 0.65 }, 0.35, 24, 48, .{ 1, 1, 1 }, -15.0);
        // Orange glass slab hanging above the short box. Width matches short box X-extent,
        // height matches the box wall (0.65), thin in Z. Albedo tints transmitted light:
        // R=100%, G=50%, B=10% per traversal ⇒ orange with ~50% translucency.
        const orange_glass = [3]f32{ 1.0, 0.5, 0.1 };
        try appendBox(
            &verts,
            &indices,
            self.allocator,
            .{ sb_min[0], 0.0,  0.175 },
            .{ sb_max[0], 0.65, 0.225 },
            orange_glass,
            true,
            -15.0,
        );

        const rt_usage = vk.BufferUsageFlags{
            .shader_device_address_bit = true,
            .acceleration_structure_build_input_read_only_bit_khr = true,
            .storage_buffer_bit = true,
        };
        self.vertex_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(verts.items), rt_usage, true);
        self.index_buf = try self.createBufferHostVisible(std.mem.sliceAsBytes(indices.items), rt_usage, true);
        self.vertex_count = @intCast(verts.items.len / floats_per_vertex);
        self.triangle_count = @intCast(indices.items.len / 3);
    }

    // --- RT setup helpers (shared shape with 14_ray4 / 15_ray5) ---

    fn queryRtProperties(self: *Ray6Renderer) !void {
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
        self: *Ray6Renderer,
        data: []const u8,
        usage: vk.BufferUsageFlags,
        device_address: bool,
    ) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = data.len };
        errdefer buf.deinit(device);

        buf.buffer = try device.createBuffer(&.{ .size = data.len, .usage = usage, .sharing_mode = .exclusive }, null);

        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .host_visible_bit = true, .host_coherent_bit = true });
        var alloc_flags = vk.MemoryAllocateFlagsInfo{ .device_mask = 0, .flags = .{ .device_address_bit = device_address } };
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

    fn createBufferDeviceLocal(self: *Ray6Renderer, size: vk.DeviceSize, usage: vk.BufferUsageFlags, device_address: bool) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = size };
        errdefer buf.deinit(device);
        buf.buffer = try device.createBuffer(&.{ .size = size, .usage = usage, .sharing_mode = .exclusive }, null);
        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .device_local_bit = true });
        var alloc_flags = vk.MemoryAllocateFlagsInfo{ .device_mask = 0, .flags = .{ .device_address_bit = device_address } };
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

    fn createBufferHostVisibleEmpty(self: *Ray6Renderer, size: vk.DeviceSize, usage: vk.BufferUsageFlags, device_address: bool) !Buffer {
        const device = self.device.?;
        var buf = Buffer{ .size = size };
        errdefer buf.deinit(device);
        buf.buffer = try device.createBuffer(&.{ .size = size, .usage = usage, .sharing_mode = .exclusive }, null);
        const reqs = device.getBufferMemoryRequirements(buf.buffer);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .host_visible_bit = true, .host_coherent_bit = true });
        var alloc_flags = vk.MemoryAllocateFlagsInfo{ .device_mask = 0, .flags = .{ .device_address_bit = device_address } };
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

    fn buildBlas(self: *Ray6Renderer) !void {
        const device = self.device.?;

        const triangles = vk.AccelerationStructureGeometryTrianglesDataKHR{
            .vertex_format = .r32g32b32_sfloat,
            .vertex_data = .{ .device_address = self.vertex_buf.device_address },
            .vertex_stride = vertex_stride_bytes,
            .max_vertex = if (self.vertex_count == 0) 0 else self.vertex_count - 1,
            .index_type = .uint32,
            .index_data = .{ .device_address = self.index_buf.device_address },
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
        var sizes = vk.AccelerationStructureBuildSizesInfoKHR{ .acceleration_structure_size = 0, .update_scratch_size = 0, .build_scratch_size = 0 };
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

        var scratch = try self.createBufferDeviceLocal(sizes.build_scratch_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, true);
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
        const range = vk.AccelerationStructureBuildRangeInfoKHR{ .primitive_count = primitive_count, .primitive_offset = 0, .first_vertex = 0, .transform_offset = 0 };
        const range_ptr: [*]const vk.AccelerationStructureBuildRangeInfoKHR = @ptrCast(&range);

        const cmd = try self.cmd_ctx.?.beginOneTime();
        device.cmdBuildAccelerationStructuresKHR(
            cmd,
            @as([*]const vk.AccelerationStructureBuildGeometryInfoKHR, @ptrCast(&build_info))[0..1],
            @as([*]const [*]const vk.AccelerationStructureBuildRangeInfoKHR, @ptrCast(&range_ptr))[0..1],
        );
        try self.cmd_ctx.?.endOneTime(self.manager.?.graphics_queue.handle, cmd);

        self.blas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{ .acceleration_structure = self.blas.handle });
    }

    fn buildTlas(self: *Ray6Renderer) !void {
        const device = self.device.?;
        const instance = vk.AccelerationStructureInstanceKHR{
            .transform = .{ .matrix = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } } },
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
        var sizes = vk.AccelerationStructureBuildSizesInfoKHR{ .acceleration_structure_size = 0, .update_scratch_size = 0, .build_scratch_size = 0 };
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

        var scratch = try self.createBufferDeviceLocal(sizes.build_scratch_size, .{ .storage_buffer_bit = true, .shader_device_address_bit = true }, true);
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
        const range = vk.AccelerationStructureBuildRangeInfoKHR{ .primitive_count = 1, .primitive_offset = 0, .first_vertex = 0, .transform_offset = 0 };
        const range_ptr: [*]const vk.AccelerationStructureBuildRangeInfoKHR = @ptrCast(&range);
        const cmd = try self.cmd_ctx.?.beginOneTime();
        device.cmdBuildAccelerationStructuresKHR(
            cmd,
            @as([*]const vk.AccelerationStructureBuildGeometryInfoKHR, @ptrCast(&build_info))[0..1],
            @as([*]const [*]const vk.AccelerationStructureBuildRangeInfoKHR, @ptrCast(&range_ptr))[0..1],
        );
        try self.cmd_ctx.?.endOneTime(self.manager.?.graphics_queue.handle, cmd);

        self.tlas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{ .acceleration_structure = self.tlas.handle });
    }

    fn createAccumulationImage(self: *Ray6Renderer, extent: vk.Extent2D) !void {
        try self.createGenericStorageImage(&self.accum_img, extent, .r32g32b32a32_sfloat);
    }

    fn createDisplayImage(self: *Ray6Renderer, extent: vk.Extent2D, format: vk.Format) !void {
        try self.createGenericStorageImage(&self.display_img, extent, format);
    }

    fn createGenericStorageImage(self: *Ray6Renderer, img: *StorageImage, extent: vk.Extent2D, format: vk.Format) !void {
        const device = self.device.?;
        img.* = .{ .format = format, .extent = extent };
        errdefer img.deinit(device);

        img.image = try device.createImage(&.{
            .image_type = .@"2d",
            .format = format,
            .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{ .storage_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true, .sampled_bit = true },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);

        const reqs = device.getImageMemoryRequirements(img.image);
        const mem_index = try self.manager.?.findMemoryTypeIndex(reqs.memory_type_bits, .{ .device_local_bit = true });
        img.memory = try device.allocateMemory(&.{ .allocation_size = reqs.size, .memory_type_index = mem_index }, null);
        try device.bindImageMemory(img.image, img.memory, 0);
        img.view = try device.createImageView(&.{
            .image = img.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        }, null);
    }

    fn initImageLayouts(self: *Ray6Renderer) !void {
        const device = self.device.?;
        const cmd = try self.cmd_ctx.?.beginOneTime();

        // Accumulation: undefined → transfer_dst, clear to zero, → general.
        transitionImage(device, cmd, self.accum_img.image, .undefined, .transfer_dst_optimal, .{
            .src_stage = .{ .top_of_pipe_bit = true },
            .dst_stage = .{ .transfer_bit = true },
            .src_access = .{},
            .dst_access = .{ .transfer_write_bit = true },
            .aspect = .{ .color_bit = true },
        });
        const clear_color = vk.ClearColorValue{ .float_32 = .{ 0, 0, 0, 0 } };
        const range = vk.ImageSubresourceRange{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 };
        device.cmdClearColorImage(cmd, self.accum_img.image, .transfer_dst_optimal, &clear_color, @as([]const vk.ImageSubresourceRange, (&range)[0..1]));
        transitionImage(device, cmd, self.accum_img.image, .transfer_dst_optimal, .general, .{
            .src_stage = .{ .transfer_bit = true },
            .dst_stage = .{ .ray_tracing_shader_bit_khr = true },
            .src_access = .{ .transfer_write_bit = true },
            .dst_access = .{ .shader_read_bit = true, .shader_write_bit = true },
            .aspect = .{ .color_bit = true },
        });

        try self.cmd_ctx.?.endOneTime(self.manager.?.graphics_queue.handle, cmd);
    }

    fn createUniformBuffers(self: *Ray6Renderer, count: usize) !void {
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

    fn createDescriptorSet(self: *Ray6Renderer, count: usize) !void {
        const device = self.device.?;

        const stage_rgen = vk.ShaderStageFlags{ .raygen_bit_khr = true };
        const stage_rgen_rchit = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true };
        const stage_rchit = vk.ShaderStageFlags{ .closest_hit_bit_khr = true };
        const stage_rgen_rchit_rmiss = vk.ShaderStageFlags{ .raygen_bit_khr = true, .closest_hit_bit_khr = true, .miss_bit_khr = true };

        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .acceleration_structure_khr, .descriptor_count = 1, .stage_flags = stage_rgen_rchit },
            .{ .binding = 1, .descriptor_type = .storage_image, .descriptor_count = 1, .stage_flags = stage_rgen },
            .{ .binding = 2, .descriptor_type = .storage_image, .descriptor_count = 1, .stage_flags = stage_rgen },
            .{ .binding = 3, .descriptor_type = .uniform_buffer, .descriptor_count = 1, .stage_flags = stage_rgen_rchit_rmiss },
            .{ .binding = 4, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = stage_rchit },
            .{ .binding = 5, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = stage_rchit },
        };
        self.desc_set_layout = try device.createDescriptorSetLayout(&.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);

        const count_u32: u32 = @intCast(count);
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .acceleration_structure_khr, .descriptor_count = count_u32 },
            .{ .type = .storage_image, .descriptor_count = count_u32 * 2 },
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

        const accum_info = vk.DescriptorImageInfo{ .sampler = .null_handle, .image_view = self.accum_img.view, .image_layout = .general };
        const display_info = vk.DescriptorImageInfo{ .sampler = .null_handle, .image_view = self.display_img.view, .image_layout = .general };
        const vertex_info = vk.DescriptorBufferInfo{ .buffer = self.vertex_buf.buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        const index_info = vk.DescriptorBufferInfo{ .buffer = self.index_buf.buffer, .offset = 0, .range = vk.WHOLE_SIZE };

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const as_info = vk.WriteDescriptorSetAccelerationStructureKHR{
                .acceleration_structure_count = 1,
                .p_acceleration_structures = @ptrCast(&self.tlas.handle),
            };
            const ubo_info = vk.DescriptorBufferInfo{ .buffer = self.uniforms[i].buffer, .offset = 0, .range = vk.WHOLE_SIZE };
            const writes = [_]vk.WriteDescriptorSet{
                .{ .p_next = @ptrCast(&as_info), .dst_set = self.desc_sets[i], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .acceleration_structure_khr, .p_image_info = undefined, .p_buffer_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = self.desc_sets[i], .dst_binding = 1, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_image, .p_image_info = @ptrCast(&accum_info), .p_buffer_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = self.desc_sets[i], .dst_binding = 2, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_image, .p_image_info = @ptrCast(&display_info), .p_buffer_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = self.desc_sets[i], .dst_binding = 3, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .uniform_buffer, .p_image_info = undefined, .p_buffer_info = @ptrCast(&ubo_info), .p_texel_buffer_view = undefined },
                .{ .dst_set = self.desc_sets[i], .dst_binding = 4, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_image_info = undefined, .p_buffer_info = @ptrCast(&vertex_info), .p_texel_buffer_view = undefined },
                .{ .dst_set = self.desc_sets[i], .dst_binding = 5, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_image_info = undefined, .p_buffer_info = @ptrCast(&index_info), .p_texel_buffer_view = undefined },
            };
            device.updateDescriptorSets(&writes, null);
        }
    }

    fn createRtPipeline(self: *Ray6Renderer) !void {
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

        // Path tracer uses an iterative loop in raygen; only one traceRayEXT
        // is in flight at a time, so ray recursion depth = 1 is enough.
        const max_recursion: u32 = @min(@as(u32, 1), self.rt_props.max_ray_recursion_depth);
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

    fn createShaderBindingTable(self: *Ray6Renderer) !void {
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

        const region_size = alignUp(handle_size_aligned, base_align);
        const raygen_offset: u32 = 0;
        const miss_offset: u32 = raygen_offset + region_size;
        const hit_offset: u32 = miss_offset + region_size;
        const sbt_size: vk.DeviceSize = hit_offset + region_size;

        self.sbt = try self.createBufferHostVisibleEmpty(
            sbt_size,
            .{ .shader_binding_table_bit_khr = true, .shader_device_address_bit = true, .transfer_src_bit = true },
            true,
        );
        const mapped = try device.mapMemory(self.sbt.memory, 0, sbt_size, .{});
        const dst = @as([*]u8, @ptrCast(@alignCast(mapped)));
        @memset(dst[0..sbt_size], 0);
        @memcpy(dst[raygen_offset .. raygen_offset + handle_size], handles[0..handle_size]);
        @memcpy(dst[miss_offset .. miss_offset + handle_size], handles[handle_size .. 2 * handle_size]);
        @memcpy(dst[hit_offset .. hit_offset + handle_size], handles[2 * handle_size .. 3 * handle_size]);
        device.unmapMemory(self.sbt.memory);

        self.sbt_raygen_region = .{ .device_address = self.sbt.device_address + raygen_offset, .stride = handle_size_aligned, .size = handle_size_aligned };
        self.sbt_miss_region = .{ .device_address = self.sbt.device_address + miss_offset, .stride = handle_size_aligned, .size = handle_size_aligned };
        self.sbt_hit_region = .{ .device_address = self.sbt.device_address + hit_offset, .stride = handle_size_aligned, .size = handle_size_aligned };
        self.sbt_callable_region = .{ .device_address = 0, .stride = 0, .size = 0 };
    }
};

fn createShaderModule(device: Device, spv: []align(@alignOf(u32)) const u8) !vk.ShaderModule {
    return try device.createShaderModule(&.{ .code_size = spv.len, .p_code = @ptrCast(@alignCast(spv.ptr)) }, null);
}

fn matEqual(a: *const Mat4, b: *const Mat4) bool {
    for (0..4) |c| {
        for (0..4) |r| if (a[c][r] != b[c][r]) return false;
    }
    return true;
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

fn transitionImage(device: Device, cmd: vk.CommandBuffer, image: vk.Image, old_layout: vk.ImageLayout, new_layout: vk.ImageLayout, opts: TransitionOpts) void {
    const barrier = vk.ImageMemoryBarrier{
        .src_access_mask = opts.src_access,
        .dst_access_mask = opts.dst_access,
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{ .aspect_mask = opts.aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
    };
    device.cmdPipelineBarrier(cmd, opts.src_stage, opts.dst_stage, .{}, &.{}, &.{}, &.{barrier});
}

// general → general image barrier guarding ray-tracing read-modify-write of the
// accumulation image across frames (no layout change, only sync).
fn accumBarrier(device: Device, cmd: vk.CommandBuffer, image: vk.Image) void {
    transitionImage(device, cmd, image, .general, .general, .{
        .src_stage = .{ .ray_tracing_shader_bit_khr = true },
        .dst_stage = .{ .ray_tracing_shader_bit_khr = true },
        .src_access = .{ .shader_write_bit = true },
        .dst_access = .{ .shader_read_bit = true, .shader_write_bit = true },
        .aspect = .{ .color_bit = true },
    });
}

fn appendBox(
    verts: *std.ArrayList(f32),
    indices: *std.ArrayList(u32),
    allocator: Allocator,
    min: [3]f32,
    max: [3]f32,
    albedo: [3]f32,
    include_top: bool,
    emission: f32,
) !void {
    const x0 = min[0]; const x1 = max[0];
    const y0 = min[1]; const y1 = max[1];
    const z0 = min[2]; const z1 = max[2];
    // Bottom (y = y0, normal -Y).
    try appendQuad(verts, indices, allocator, .{ x0, y0, z0 }, .{ x0, y0, z1 }, .{ x1, y0, z1 }, .{ x1, y0, z0 }, .{ 0, -1, 0 }, albedo, emission);
    // Top (y = y1, normal +Y).
    if (include_top) {
        try appendQuad(verts, indices, allocator, .{ x0, y1, z0 }, .{ x1, y1, z0 }, .{ x1, y1, z1 }, .{ x0, y1, z1 }, .{ 0, 1, 0 }, albedo, emission);
    }
    // -X face (normal -X).
    try appendQuad(verts, indices, allocator, .{ x0, y0, z0 }, .{ x0, y1, z0 }, .{ x0, y1, z1 }, .{ x0, y0, z1 }, .{ -1, 0, 0 }, albedo, emission);
    // +X face (normal +X).
    try appendQuad(verts, indices, allocator, .{ x1, y0, z1 }, .{ x1, y1, z1 }, .{ x1, y1, z0 }, .{ x1, y0, z0 }, .{ 1, 0, 0 }, albedo, emission);
    // -Z face (normal -Z).
    try appendQuad(verts, indices, allocator, .{ x1, y0, z0 }, .{ x1, y1, z0 }, .{ x0, y1, z0 }, .{ x0, y0, z0 }, .{ 0, 0, -1 }, albedo, emission);
    // +Z face (normal +Z, the side closest to the camera).
    try appendQuad(verts, indices, allocator, .{ x0, y0, z1 }, .{ x0, y1, z1 }, .{ x1, y1, z1 }, .{ x1, y0, z1 }, .{ 0, 0, 1 }, albedo, emission);
}

fn appendQuad(
    verts: *std.ArrayList(f32),
    indices: *std.ArrayList(u32),
    allocator: Allocator,
    p0: [3]f32,
    p1: [3]f32,
    p2: [3]f32,
    p3: [3]f32,
    normal: [3]f32,
    albedo: [3]f32,
    emission: f32,
) !void {
    const base: u32 = @intCast(verts.items.len / floats_per_vertex);
    const corners = [_][3]f32{ p0, p1, p2, p3 };
    for (corners) |p| {
        try verts.appendSlice(allocator, &.{
            p[0],      p[1],      p[2],
            normal[0], normal[1], normal[2],
            0.0,       0.0,
            albedo[0], albedo[1], albedo[2], emission,
        });
    }
    try indices.appendSlice(allocator, &.{
        base,     base + 1, base + 2,
        base,     base + 2, base + 3,
    });
}

fn appendSphere(
    verts: *std.ArrayList(f32),
    indices: *std.ArrayList(u32),
    allocator: Allocator,
    center: [3]f32,
    radius: f32,
    stacks: u32,
    sectors: u32,
    albedo: [3]f32,
    emission: f32,
) !void {
    const base: u32 = @intCast(verts.items.len / floats_per_vertex);
    const pi: f32 = std.math.pi;

    var i: u32 = 0;
    while (i <= stacks) : (i += 1) {
        const phi = pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(stacks));
        const cy = @cos(phi);
        const sy = @sin(phi);
        var j: u32 = 0;
        while (j <= sectors) : (j += 1) {
            const theta = 2.0 * pi * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(sectors));
            const nx = sy * @cos(theta);
            const nz = sy * @sin(theta);
            const ny = cy;
            const px = center[0] + radius * nx;
            const py = center[1] + radius * ny;
            const pz = center[2] + radius * nz;
            try verts.appendSlice(allocator, &.{
                px,        py,        pz,
                nx,        ny,        nz,
                0.0,       0.0,
                albedo[0], albedo[1], albedo[2], emission,
            });
        }
    }

    const row: u32 = sectors + 1;
    var s: u32 = 0;
    while (s < stacks) : (s += 1) {
        var t: u32 = 0;
        while (t < sectors) : (t += 1) {
            const k1 = base + s * row + t;
            const k2 = k1 + row;
            try indices.appendSlice(allocator, &.{ k1, k2, k1 + 1, k1 + 1, k2, k2 + 1 });
        }
    }
}

fn perspectiveZO(y_fov: f32, aspect: f32, near: f32, far: f32) Mat4 {
    // Vulkan NDC has +Y pointing down; negate the Y scale so world-space +Y
    // ends up at the top of the rendered image (standard Vulkan idiom).
    const f = 1.0 / @tan(y_fov / 2.0);
    return .{
        .{ f / aspect, 0, 0, 0 },
        .{ 0, -f, 0, 0 },
        .{ 0, 0, far / (near - far), -1 },
        .{ 0, 0, (near * far) / (near - far), 0 },
    };
}
