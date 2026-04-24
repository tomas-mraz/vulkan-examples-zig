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

const triangle_vertices = [_]f32{
    1.0, 1.0, 0.0,
    -1.0, 1.0, 0.0,
    0.0, -1.0, 0.0,
};
const triangle_indices = [_]u32{ 0, 1, 2 };

const UniformData = extern struct {
    view_inverse: Mat4,
    proj_inverse: Mat4,
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

pub const Ray1Renderer = struct {
    allocator: Allocator,

    manager: ?*ash.Manager = null,
    cmd_ctx: ?*ash.CommandContext = null,
    device: ?Device = null,

    rt_props: vk.PhysicalDeviceRayTracingPipelinePropertiesKHR = .{
        .shader_group_handle_size = 0,
        .max_ray_recursion_depth = 0,
        .max_shader_group_stride = 0,
        .shader_group_base_alignment = 0,
        .shader_group_handle_capture_replay_size = 0,
        .max_ray_dispatch_invocation_count = 0,
        .shader_group_handle_alignment = 0,
        .max_ray_hit_attribute_size = 0,
    },

    // once
    vertex_buf: Buffer = .{},
    index_buf: Buffer = .{},
    blas: AccelStruct = .{},
    tlas: AccelStruct = .{},

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

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: Allocator) Ray1Renderer {
        return .{ .allocator = allocator };
    }

    pub fn createOnce(self: *Ray1Renderer, session: *ash.Session) !void {
        self.manager = &session.manager.?;
        self.cmd_ctx = &session.cmd_ctx.?;
        self.device = self.manager.?.device.?;

        try self.queryRtProperties();

        const rt_usage = vk.BufferUsageFlags{
            .shader_device_address_bit = true,
            .acceleration_structure_build_input_read_only_bit_khr = true,
            .storage_buffer_bit = true,
        };
        self.vertex_buf = try self.createBufferHostVisible(
            std.mem.sliceAsBytes(&triangle_vertices),
            rt_usage,
            true,
        );
        self.index_buf = try self.createBufferHostVisible(
            std.mem.sliceAsBytes(&triangle_indices),
            rt_usage,
            true,
        );

        try self.buildBlas();
        try self.buildTlas();

        self.view_matrix = math.lookAt(
            .{ .x = 0, .y = 0, .z = -2.5 },
            .{ .x = 0, .y = 0, .z = 0 },
            .{ .x = 0, .y = 1, .z = 0 },
        );

        self.once_built = true;
    }

    pub fn destroyOnce(self: *Ray1Renderer) void {
        if (!self.once_built) return;
        const device = self.device.?;
        self.tlas.deinit(device);
        self.blas.deinit(device);
        self.index_buf.deinit(device);
        self.vertex_buf.deinit(device);
        self.once_built = false;
    }

    pub fn createSized(self: *Ray1Renderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const swap_len = session.swapchain.?.imageCount();
        const display_format = session.swapchain.?.surface_format.format;

        try self.createStorageImage(extent, display_format);
        try self.createUniformBuffers(swap_len);
        try self.createDescriptorSet(swap_len);
        try self.createRtPipeline();
        try self.createShaderBindingTable();

        self.sized_built = true;
    }

    pub fn destroySized(self: *Ray1Renderer) void {
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
        if (self.desc_sets.len != 0) {
            self.allocator.free(self.desc_sets);
            self.desc_sets = &.{};
        }
        for (self.uniforms) |*ub| ub.deinit(device);
        if (self.uniforms.len != 0) {
            self.allocator.free(self.uniforms);
            self.uniforms = &.{};
        }
        self.storage_img.deinit(device);
        self.sized_built = false;
    }

    pub fn draw(self: *Ray1Renderer, session: *ash.Session, frame: *const ash.Frame) !void {
        const device = self.device.?;
        const aspect: f32 = @as(f32, @floatFromInt(frame.extent.width)) / @as(f32, @floatFromInt(frame.extent.height));
        var proj = math.perspective(math.degreesToRadians(60.0), aspect, 0.1, 512.0);
        proj[1][1] *= -1.0;

        var ubo = UniformData{
            .view_inverse = math.invert(self.view_matrix),
            .proj_inverse = math.invert(proj),
        };

        const ub = &self.uniforms[frame.image_index];
        const mapped = try device.mapMemory(ub.memory, 0, vk.WHOLE_SIZE, .{});
        @memcpy(@as([*]u8, @ptrCast(@alignCast(mapped)))[0..@sizeOf(UniformData)], std.mem.asBytes(&ubo));
        device.unmapMemory(ub.memory);

        _ = session;

        // Transition storage image to GENERAL (for writes from raygen) and
        // swapchain image to TRANSFER_DST for the copy afterwards.
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

        // Copy storage image to swapchain image.
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
            .src_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .src_offset = .{ .x = 0, .y = 0, .z = 0 },
            .dst_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
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

    // --- helpers ---

    fn queryRtProperties(self: *Ray1Renderer) !void {
        const manager = self.manager.?;
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
        manager.instance.?.getPhysicalDeviceProperties2(manager.gpu, &props2);
    }

    fn createBufferHostVisible(
        self: *Ray1Renderer,
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
        self: *Ray1Renderer,
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
            .device_local_bit = true,
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

    fn buildBlas(self: *Ray1Renderer) !void {
        const device = self.device.?;

        const triangles_data = vk.AccelerationStructureGeometryTrianglesDataKHR{
            .vertex_format = .r32g32b32_sfloat,
            .vertex_data = .{ .device_address = self.vertex_buf.device_address },
            .vertex_stride = 12,
            .max_vertex = 3,
            .index_type = .uint32,
            .index_data = .{ .device_address = self.index_buf.device_address },
            .transform_data = .{ .device_address = 0 },
        };
        const geometry = vk.AccelerationStructureGeometryKHR{
            .geometry_type = .triangles_khr,
            .geometry = .{ .triangles = triangles_data },
            .flags = .{ .opaque_bit_khr = true },
        };

        const flags = vk.BuildAccelerationStructureFlagsKHR{ .prefer_fast_trace_bit_khr = true };
        const primitive_count: u32 = 1;
        var build_info_sz = vk.AccelerationStructureBuildGeometryInfoKHR{
            .type = .bottom_level_khr,
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
        device.getAccelerationStructureBuildSizesKHR(
            .device_khr,
            &build_info_sz,
            @ptrCast(&primitive_count),
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

        self.blas.device_address = device.getAccelerationStructureDeviceAddressKHR(&.{
            .acceleration_structure = self.blas.handle,
        });
    }

    fn buildTlas(self: *Ray1Renderer) !void {
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
        const instance_bytes = std.mem.asBytes(&instance);

        var instance_buf = try self.createBufferHostVisible(
            instance_bytes,
            .{
                .shader_device_address_bit = true,
                .acceleration_structure_build_input_read_only_bit_khr = true,
            },
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
        device.getAccelerationStructureBuildSizesKHR(
            .device_khr,
            &build_info_sz,
            @ptrCast(&primitive_count),
            &sizes,
        );

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

    fn createStorageImage(self: *Ray1Renderer, extent: vk.Extent2D, format: vk.Format) !void {
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
            .usage = .{ .storage_bit = true, .transfer_src_bit = true },
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

    fn createUniformBuffers(self: *Ray1Renderer, count: usize) !void {
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

    fn createBufferHostVisibleEmpty(
        self: *Ray1Renderer,
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

    fn createDescriptorSet(self: *Ray1Renderer, count: usize) !void {
        const device = self.device.?;
        const rt_stage = vk.ShaderStageFlags{ .raygen_bit_khr = true };

        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .acceleration_structure_khr, .descriptor_count = 1, .stage_flags = rt_stage },
            .{ .binding = 1, .descriptor_type = .storage_image, .descriptor_count = 1, .stage_flags = rt_stage },
            .{ .binding = 2, .descriptor_type = .uniform_buffer, .descriptor_count = 1, .stage_flags = rt_stage },
        };
        self.desc_set_layout = try device.createDescriptorSetLayout(&.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);

        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .acceleration_structure_khr, .descriptor_count = @intCast(count) },
            .{ .type = .storage_image, .descriptor_count = @intCast(count) },
            .{ .type = .uniform_buffer, .descriptor_count = @intCast(count) },
        };
        self.desc_pool = try device.createDescriptorPool(&.{
            .max_sets = @intCast(count),
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null);

        const layouts = try self.allocator.alloc(vk.DescriptorSetLayout, count);
        defer self.allocator.free(layouts);
        for (layouts) |*l| l.* = self.desc_set_layout;

        self.desc_sets = try self.allocator.alloc(vk.DescriptorSet, count);
        try device.allocateDescriptorSets(&.{
            .descriptor_pool = self.desc_pool,
            .descriptor_set_count = @intCast(count),
            .p_set_layouts = layouts.ptr,
        }, self.desc_sets.ptr);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const as_info = vk.WriteDescriptorSetAccelerationStructureKHR{
                .acceleration_structure_count = 1,
                .p_acceleration_structures = @ptrCast(&self.tlas.handle),
            };
            const image_info = vk.DescriptorImageInfo{
                .sampler = .null_handle,
                .image_view = self.storage_img.view,
                .image_layout = .general,
            };
            const buffer_info = vk.DescriptorBufferInfo{
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
                    .p_image_info = @ptrCast(&image_info),
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
                    .p_buffer_info = @ptrCast(&buffer_info),
                    .p_texel_buffer_view = undefined,
                },
            };
            device.updateDescriptorSets(&writes, null);
        }
    }

    fn createRtPipeline(self: *Ray1Renderer) !void {
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

        const create_info = vk.RayTracingPipelineCreateInfoKHR{
            .stage_count = stages.len,
            .p_stages = &stages,
            .group_count = groups.len,
            .p_groups = &groups,
            .max_pipeline_ray_recursion_depth = 1,
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

    fn createShaderBindingTable(self: *Ray1Renderer) !void {
        const device = self.device.?;
        const handle_size = self.rt_props.shader_group_handle_size;
        const handle_align = self.rt_props.shader_group_handle_alignment;
        const base_align = self.rt_props.shader_group_base_alignment;
        const handle_size_aligned = alignUp(handle_size, handle_align);

        const group_count: u32 = 3; // raygen, miss, hit
        const handles_size = handle_size * group_count;
        const handles = try self.allocator.alloc(u8, handles_size);
        defer self.allocator.free(handles);
        try device.getRayTracingShaderGroupHandlesKHR(self.pipeline, 0, group_count, handles_size, handles.ptr);

        // Each region starts at base_align; one handle per group here.
        const raygen_offset: u32 = 0;
        const raygen_size = alignUp(handle_size_aligned, base_align);
        const miss_offset: u32 = raygen_offset + raygen_size;
        const miss_size = alignUp(handle_size_aligned, base_align);
        const hit_offset: u32 = miss_offset + miss_size;
        const hit_size = alignUp(handle_size_aligned, base_align);
        const sbt_size: vk.DeviceSize = hit_offset + hit_size;

        self.sbt = try self.createBufferHostVisibleEmpty(
            sbt_size,
            .{
                .shader_binding_table_bit_khr = true,
                .shader_device_address_bit = true,
                .transfer_src_bit = true,
            },
            true,
        );

        const mapped = try device.mapMemory(self.sbt.memory, 0, sbt_size, .{});
        const dst = @as([*]u8, @ptrCast(@alignCast(mapped)));
        @memset(dst[0..sbt_size], 0);
        @memcpy(dst[raygen_offset .. raygen_offset + handle_size], handles[0..handle_size]);
        @memcpy(dst[miss_offset .. miss_offset + handle_size], handles[handle_size .. 2 * handle_size]);
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
