const ash = @import("ash");
const std = @import("std");
const zgltf = @import("zgltf");
const vk = ash.vk;
const math = ash.math;

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;
const model_gltf_bytes align(4) = @embedFile("model_gltf").*;

const depth_format: vk.Format = .d32_sfloat;
const floats_per_vertex: usize = 6;
const vertex_stride_bytes: u32 = floats_per_vertex * @sizeOf(f32);

const UniformData = extern struct {
    mvp: math.Mat4,
    model: math.Mat4,
};

const UniformBuffers = struct {
    allocator: std.mem.Allocator,
    buffers: []vk.Buffer = &.{},
    memories: []vk.DeviceMemory = &.{},
    mapped: []?[*]u8 = &.{},

    fn init(
        allocator: std.mem.Allocator,
        manager: *const ash.Manager,
        count: usize,
    ) !UniformBuffers {
        var self = UniformBuffers{
            .allocator = allocator,
            .buffers = try allocator.alloc(vk.Buffer, count),
            .memories = try allocator.alloc(vk.DeviceMemory, count),
            .mapped = try allocator.alloc(?[*]u8, count),
        };
        errdefer self.deinit(manager.device.?);

        const device = manager.device orelse return error.DeviceNotInitialized;
        for (self.buffers, self.memories, self.mapped) |*buffer, *memory, *mapped| {
            buffer.* = try device.createBuffer(&.{
                .size = @sizeOf(UniformData),
                .usage = .{ .uniform_buffer_bit = true },
                .sharing_mode = .exclusive,
            }, null);
            errdefer device.destroyBuffer(buffer.*, null);

            const requirements = device.getBufferMemoryRequirements(buffer.*);
            memory.* = try manager.allocate(requirements, .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            });
            errdefer device.freeMemory(memory.*, null);

            try device.bindBufferMemory(buffer.*, memory.*, 0);
            mapped.* = @ptrCast(@alignCast(try device.mapMemory(memory.*, 0, vk.WHOLE_SIZE, .{})));
        }

        return self;
    }

    fn update(self: *UniformBuffers, index: u32, data: *const UniformData) void {
        const bytes = std.mem.asBytes(data);
        @memcpy(self.mapped[index].?[0..bytes.len], bytes);
    }

    fn deinit(self: *UniformBuffers, device: vk.DeviceProxy) void {
        for (self.memories) |memory| {
            if (memory != .null_handle) {
                device.unmapMemory(memory);
                device.freeMemory(memory, null);
            }
        }
        for (self.buffers) |buffer| {
            if (buffer != .null_handle) {
                device.destroyBuffer(buffer, null);
            }
        }
        self.allocator.free(self.mapped);
        self.allocator.free(self.memories);
        self.allocator.free(self.buffers);
        self.* = .{ .allocator = self.allocator };
    }
};

const DescriptorInfo = struct {
    allocator: std.mem.Allocator,
    layout: vk.DescriptorSetLayout = .null_handle,
    pool: vk.DescriptorPool = .null_handle,
    sets: []vk.DescriptorSet = &.{},

    fn deinit(self: *DescriptorInfo, device: vk.DeviceProxy) void {
        if (self.pool != .null_handle) {
            device.destroyDescriptorPool(self.pool, null);
            self.pool = .null_handle;
        }
        if (self.layout != .null_handle) {
            device.destroyDescriptorSetLayout(self.layout, null);
            self.layout = .null_handle;
        }
        self.allocator.free(self.sets);
        self.* = .{ .allocator = self.allocator };
    }
};

const BufferResource = struct {
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,

    fn deinit(self: *BufferResource, device: vk.DeviceProxy) void {
        if (self.buffer != .null_handle) {
            device.destroyBuffer(self.buffer, null);
            self.buffer = .null_handle;
        }
        if (self.memory != .null_handle) {
            device.freeMemory(self.memory, null);
            self.memory = .null_handle;
        }
    }
};

const ModelData = struct {
    allocator: std.mem.Allocator,
    vertices: []f32,
    indices: []u32,

    fn deinit(self: *ModelData) void {
        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
    }
};

pub const ModelRenderer = struct {
    allocator: std.mem.Allocator,
    device: ?vk.DeviceProxy = null,

    model: ?ModelData = null,
    vertex_buffer: BufferResource = .{},
    index_buffer: BufferResource = .{},
    uniforms: ?UniformBuffers = null,
    descriptor: ?DescriptorInfo = null,

    depth: ash.ImageResource = .{},
    render_pass: vk.RenderPass = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    pipeline: vk.Pipeline = .null_handle,
    framebuffers: []vk.Framebuffer = &.{},

    start_time: f64 = 0,
    proj_matrix: math.Mat4 = math.identity(),
    view_matrix: math.Mat4 = math.identity(),

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: std.mem.Allocator) ModelRenderer {
        return .{ .allocator = allocator };
    }

    pub fn createOnce(self: *ModelRenderer, session: *ash.Session) !void {
        const manager = &session.manager.?;
        const device = manager.device orelse return error.DeviceNotInitialized;
        self.device = device;

        self.model = try loadTeapotModel(self.allocator);
        std.log.info("loaded model: {} vertices, {} indices", .{
            self.model.?.vertices.len / floats_per_vertex,
            self.model.?.indices.len,
        });

        self.vertex_buffer = try createHostVisibleBuffer(
            manager,
            std.mem.sliceAsBytes(self.model.?.vertices),
            .{ .vertex_buffer_bit = true },
        );
        self.index_buffer = try createHostVisibleBuffer(
            manager,
            std.mem.sliceAsBytes(self.model.?.indices),
            .{ .index_buffer_bit = true },
        );

        const swap_len = session.swapchain.?.imageCount();
        self.uniforms = try UniformBuffers.init(self.allocator, manager, swap_len);
        self.descriptor = try createDescriptors(self.allocator, device, self.uniforms.?, swap_len);

        self.view_matrix = math.lookAt(
            .{ .x = 0, .y = 3, .z = 8 },
            .{ .x = 0, .y = 0, .z = 0 },
            .{ .x = 0, .y = 1, .z = 0 },
        );
        self.start_time = ash.glfw.getTime();
        self.once_built = true;
    }

    pub fn destroyOnce(self: *ModelRenderer) void {
        if (!self.once_built) {
            return;
        }
        const device = self.device orelse return;
        if (self.descriptor) |*descriptor| {
            descriptor.deinit(device);
            self.descriptor = null;
        }
        if (self.uniforms) |*uniforms| {
            uniforms.deinit(device);
            self.uniforms = null;
        }
        self.index_buffer.deinit(device);
        self.vertex_buffer.deinit(device);
        if (self.model) |*model| {
            model.deinit();
            self.model = null;
        }
        self.device = null;
        self.once_built = false;
    }

    pub fn createSized(self: *ModelRenderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const manager = &session.manager.?;
        const device = manager.device orelse return error.DeviceNotInitialized;

        self.depth = try createDepthImage(manager, extent.width, extent.height);
        errdefer self.depth.deinit(device);

        self.render_pass = try createRenderPass(device, session.swapchain.?.surface_format.format);
        errdefer device.destroyRenderPass(self.render_pass, null);

        self.pipeline_layout = try createPipelineLayout(device, self.descriptor.?.layout);
        errdefer device.destroyPipelineLayout(self.pipeline_layout, null);

        self.pipeline = try createPipeline(device, self.pipeline_layout, self.render_pass);
        errdefer device.destroyPipeline(self.pipeline, null);

        self.framebuffers = try createFramebuffers(
            self.allocator,
            device,
            session.swapchain.?,
            self.render_pass,
            self.depth.view,
        );
        errdefer destroyFramebuffers(self.allocator, device, self.framebuffers);

        self.proj_matrix = math.perspective(
            math.degreesToRadians(45.0),
            @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height)),
            0.1,
            100.0,
        );
        self.proj_matrix[1][1] *= -1;

        self.sized_built = true;
    }

    pub fn destroySized(self: *ModelRenderer) void {
        if (!self.sized_built) {
            return;
        }
        const device = self.device orelse return;
        destroyFramebuffers(self.allocator, device, self.framebuffers);
        self.framebuffers = &.{};
        if (self.pipeline != .null_handle) {
            device.destroyPipeline(self.pipeline, null);
            self.pipeline = .null_handle;
        }
        if (self.pipeline_layout != .null_handle) {
            device.destroyPipelineLayout(self.pipeline_layout, null);
            self.pipeline_layout = .null_handle;
        }
        if (self.render_pass != .null_handle) {
            device.destroyRenderPass(self.render_pass, null);
            self.render_pass = .null_handle;
        }
        self.depth.deinit(device);
        self.sized_built = false;
    }

    pub fn draw(self: *ModelRenderer, session: *ash.Session, frame: *const ash.Frame) !void {
        const device = session.manager.?.device orelse return error.DeviceNotInitialized;

        const elapsed = @as(f32, @floatCast(ash.glfw.getTime() - self.start_time)) * 45.0;
        const model_matrix = math.rotateY(&math.identity(), math.degreesToRadians(elapsed));
        const vp = math.multiply(&self.proj_matrix, &self.view_matrix);
        const mvp = math.multiply(&vp, &model_matrix);
        const ubo = UniformData{ .mvp = mvp, .model = model_matrix };
        self.uniforms.?.update(frame.image_index, &ubo);

        const clear_values = [_]vk.ClearValue{
            .{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.12, 1.0 } } },
            .{ .depth_stencil = .{ .depth = 1.0, .stencil = 0 } },
        };

        device.cmdBeginRenderPass(frame.cmd, &.{
            .render_pass = self.render_pass,
            .framebuffer = self.framebuffers[@intCast(frame.image_index)],
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = frame.extent,
            },
            .clear_value_count = clear_values.len,
            .p_clear_values = &clear_values,
        }, .@"inline");

        device.cmdBindPipeline(frame.cmd, .graphics, self.pipeline);
        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(frame.extent.width),
            .height = @floatFromInt(frame.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };
        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = frame.extent,
        };
        device.cmdSetViewport(frame.cmd, 0, &.{viewport});
        device.cmdSetScissor(frame.cmd, 0, &.{scissor});
        device.cmdBindDescriptorSets(
            frame.cmd,
            .graphics,
            self.pipeline_layout,
            0,
            (&self.descriptor.?.sets[@intCast(frame.image_index)])[0..1],
            &.{},
        );
        const vertex_buffers = [_]vk.Buffer{self.vertex_buffer.buffer};
        const offsets = [_]vk.DeviceSize{0};
        device.cmdBindVertexBuffers(frame.cmd, 0, &vertex_buffers, &offsets);
        device.cmdBindIndexBuffer(frame.cmd, self.index_buffer.buffer, 0, .uint32);
        device.cmdDrawIndexed(frame.cmd, @intCast(self.model.?.indices.len), 1, 0, 0, 0);
        device.cmdEndRenderPass(frame.cmd);
    }
};

fn loadTeapotModel(allocator: std.mem.Allocator) !ModelData {
    var gltf = zgltf.Gltf.init(allocator);
    defer gltf.deinit();

    try gltf.parse(&model_gltf_bytes);

    if (gltf.data.buffers.len == 0) {
        return error.GltfMissingBuffer;
    }
    const uri = gltf.data.buffers[0].uri orelse return error.GltfMissingBufferUri;
    const binary = try decodeDataUri(allocator, uri);
    defer allocator.free(binary);

    if (gltf.data.meshes.len == 0 or gltf.data.meshes[0].primitives.len == 0) {
        return error.GltfMissingMesh;
    }

    var vertices = std.ArrayList(f32).empty;
    errdefer vertices.deinit(allocator);
    var indices = std.ArrayList(u32).empty;
    errdefer indices.deinit(allocator);

    for (gltf.data.meshes) |mesh| {
        for (mesh.primitives) |primitive| {
            const vertex_offset: u32 = @intCast(vertices.items.len / floats_per_vertex);

            var position_index: ?usize = null;
            var normal_index: ?usize = null;
            for (primitive.attributes) |attribute| {
                switch (attribute) {
                    .position => |idx| position_index = idx,
                    .normal => |idx| normal_index = idx,
                    else => {},
                }
            }
            const pos_idx = position_index orelse return error.GltfMissingPositions;
            const norm_idx = normal_index orelse return error.GltfMissingNormals;

            const pos_accessor = gltf.data.accessors[pos_idx];
            const norm_accessor = gltf.data.accessors[norm_idx];
            if (pos_accessor.count != norm_accessor.count) {
                return error.GltfVertexAttributeMismatch;
            }

            const vertex_count = pos_accessor.count;
            try vertices.ensureUnusedCapacity(allocator, vertex_count * floats_per_vertex);

            var pos_it = pos_accessor.iterator(f32, &gltf, binary);
            var norm_it = norm_accessor.iterator(f32, &gltf, binary);
            var i: usize = 0;
            while (i < vertex_count) : (i += 1) {
                const p = pos_it.next() orelse return error.GltfPositionShort;
                const n = norm_it.next() orelse return error.GltfNormalShort;
                vertices.appendSliceAssumeCapacity(&.{
                    p[0], p[1], p[2],
                    n[0], n[1], n[2],
                });
            }

            const indices_accessor_index = primitive.indices orelse return error.GltfMissingIndices;
            const indices_accessor = gltf.data.accessors[indices_accessor_index];
            try indices.ensureUnusedCapacity(allocator, indices_accessor.count);
            switch (indices_accessor.component_type) {
                .unsigned_short => {
                    var it = indices_accessor.iterator(u16, &gltf, binary);
                    while (it.next()) |slice| {
                        indices.appendAssumeCapacity(@as(u32, slice[0]) + vertex_offset);
                    }
                },
                .unsigned_integer => {
                    var it = indices_accessor.iterator(u32, &gltf, binary);
                    while (it.next()) |slice| {
                        indices.appendAssumeCapacity(slice[0] + vertex_offset);
                    }
                },
                .unsigned_byte => {
                    var it = indices_accessor.iterator(u8, &gltf, binary);
                    while (it.next()) |slice| {
                        indices.appendAssumeCapacity(@as(u32, slice[0]) + vertex_offset);
                    }
                },
                else => return error.GltfUnsupportedIndexType,
            }
        }
    }

    return .{
        .allocator = allocator,
        .vertices = try vertices.toOwnedSlice(allocator),
        .indices = try indices.toOwnedSlice(allocator),
    };
}

fn decodeDataUri(allocator: std.mem.Allocator, uri: []const u8) ![]align(4) const u8 {
    const prefix = "data:application/octet-stream;base64,";
    if (!std.mem.startsWith(u8, uri, prefix)) {
        return error.UnsupportedBufferUri;
    }
    const encoded = uri[prefix.len..];
    const decoder = std.base64.standard.Decoder;
    const decoded_len = try decoder.calcSizeForSlice(encoded);
    const out = try allocator.alignedAlloc(u8, .of(f32), decoded_len);
    errdefer allocator.free(out);
    try decoder.decode(out, encoded);
    return out;
}

fn createHostVisibleBuffer(
    manager: *const ash.Manager,
    data: []const u8,
    usage: vk.BufferUsageFlags,
) !BufferResource {
    const device = manager.device orelse return error.DeviceNotInitialized;
    var result = BufferResource{};
    errdefer result.deinit(device);

    result.buffer = try device.createBuffer(&.{
        .size = data.len,
        .usage = usage,
        .sharing_mode = .exclusive,
    }, null);
    const requirements = device.getBufferMemoryRequirements(result.buffer);
    result.memory = try manager.allocate(requirements, .{
        .host_visible_bit = true,
        .host_coherent_bit = true,
    });
    try device.bindBufferMemory(result.buffer, result.memory, 0);

    const mapped = try device.mapMemory(result.memory, 0, vk.WHOLE_SIZE, .{});
    defer device.unmapMemory(result.memory);
    @memcpy((@as([*]u8, @ptrCast(@alignCast(mapped))))[0..data.len], data);

    return result;
}

fn createDepthImage(manager: *const ash.Manager, width: u32, height: u32) !ash.ImageResource {
    const device = manager.device orelse return error.DeviceNotInitialized;

    var result: ash.ImageResource = .{
        .format = depth_format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .layout = .undefined,
    };
    errdefer result.deinit(device);

    result.image = try device.createImage(&.{
        .image_type = .@"2d",
        .format = depth_format,
        .extent = result.extent,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{ .depth_stencil_attachment_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);

    const requirements = device.getImageMemoryRequirements(result.image);
    result.memory = try manager.allocate(requirements, .{ .device_local_bit = true });
    try device.bindImageMemory(result.image, result.memory, 0);

    result.view = try device.createImageView(&.{
        .image = result.image,
        .view_type = .@"2d",
        .format = depth_format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{
            .aspect_mask = .{ .depth_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    }, null);

    return result;
}

fn createDescriptors(
    allocator: std.mem.Allocator,
    device: vk.DeviceProxy,
    uniforms: UniformBuffers,
    set_count: usize,
) !DescriptorInfo {
    var descriptor = DescriptorInfo{
        .allocator = allocator,
        .sets = try allocator.alloc(vk.DescriptorSet, set_count),
    };
    errdefer descriptor.deinit(device);

    const bindings = [_]vk.DescriptorSetLayoutBinding{
        .{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true },
        },
    };
    descriptor.layout = try device.createDescriptorSetLayout(&.{
        .binding_count = bindings.len,
        .p_bindings = &bindings,
    }, null);

    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .uniform_buffer, .descriptor_count = @intCast(set_count) },
    };
    descriptor.pool = try device.createDescriptorPool(&.{
        .max_sets = @intCast(set_count),
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
    }, null);

    const layouts = try allocator.alloc(vk.DescriptorSetLayout, set_count);
    defer allocator.free(layouts);
    for (layouts) |*layout| {
        layout.* = descriptor.layout;
    }
    try device.allocateDescriptorSets(&.{
        .descriptor_pool = descriptor.pool,
        .descriptor_set_count = @intCast(set_count),
        .p_set_layouts = layouts.ptr,
    }, descriptor.sets.ptr);

    for (descriptor.sets, 0..) |set, index| {
        const buffer_info = vk.DescriptorBufferInfo{
            .buffer = uniforms.buffers[index],
            .offset = 0,
            .range = @sizeOf(UniformData),
        };
        const writes = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_buffer_info = @ptrCast(&buffer_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        };
        device.updateDescriptorSets(&writes, &.{});
    }

    return descriptor;
}

fn createRenderPass(device: vk.DeviceProxy, display_format: vk.Format) !vk.RenderPass {
    const attachments = [_]vk.AttachmentDescription{
        .{
            .format = display_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        },
        .{
            .format = depth_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .depth_stencil_attachment_optimal,
        },
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };
    const depth_attachment_ref = vk.AttachmentReference{
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };
    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_depth_stencil_attachment = &depth_attachment_ref,
    };

    return try device.createRenderPass(&.{
        .attachment_count = attachments.len,
        .p_attachments = &attachments,
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
    }, null);
}

fn createPipelineLayout(device: vk.DeviceProxy, descriptor_set_layout: vk.DescriptorSetLayout) !vk.PipelineLayout {
    return try device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&descriptor_set_layout),
    }, null);
}

fn createPipeline(
    device: vk.DeviceProxy,
    pipeline_layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try device.createShaderModule(&.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(&vert_spv),
    }, null);
    defer device.destroyShaderModule(vert, null);

    const frag = try device.createShaderModule(&.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(&frag_spv),
    }, null);
    defer device.destroyShaderModule(frag, null);

    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex_bit = true }, .module = vert, .p_name = "main" },
        .{ .stage = .{ .fragment_bit = true }, .module = frag, .p_name = "main" },
    };

    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = vertex_stride_bytes,
        .input_rate = .vertex,
    };
    const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{ .binding = 0, .location = 0, .format = .r32g32b32_sfloat, .offset = 0 },
        .{ .binding = 0, .location = 1, .format = .r32g32b32_sfloat, .offset = 3 * @sizeOf(f32) },
    };
    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&binding_description),
        .vertex_attribute_description_count = attribute_descriptions.len,
        .p_vertex_attribute_descriptions = &attribute_descriptions,
    };
    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };
    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = null,
        .scissor_count = 1,
        .p_scissors = null,
    };
    const rasterization = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = .{},
        .front_face = .counter_clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };
    const multisample = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 0,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };
    const depth_stencil = vk.PipelineDepthStencilStateCreateInfo{
        .depth_test_enable = .true,
        .depth_write_enable = .true,
        .depth_compare_op = .less_or_equal,
        .depth_bounds_test_enable = .false,
        .stencil_test_enable = .false,
        .front = .{
            .fail_op = .keep,
            .pass_op = .keep,
            .depth_fail_op = .keep,
            .compare_op = .always,
            .compare_mask = 0,
            .write_mask = 0,
            .reference = 0,
        },
        .back = .{
            .fail_op = .keep,
            .pass_op = .keep,
            .depth_fail_op = .keep,
            .compare_op = .always,
            .compare_mask = 0,
            .write_mask = 0,
            .reference = 0,
        },
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
    const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };
    const color_blend = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_blend_attachment),
        .blend_constants = .{ 0, 0, 0, 0 },
    };
    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const pipeline_info = vk.GraphicsPipelineCreateInfo{
        .stage_count = shader_stages.len,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &input_assembly,
        .p_tessellation_state = null,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        .p_depth_stencil_state = &depth_stencil,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic_state,
        .layout = pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try device.createGraphicsPipelines(.null_handle, &.{pipeline_info}, null, (&pipeline)[0..1]);
    return pipeline;
}

fn createFramebuffers(
    allocator: std.mem.Allocator,
    device: vk.DeviceProxy,
    swapchain: ash.Swapchain,
    render_pass: vk.RenderPass,
    depth_view: vk.ImageView,
) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var created: usize = 0;
    errdefer {
        for (framebuffers[0..created]) |framebuffer| {
            device.destroyFramebuffer(framebuffer, null);
        }
    }

    for (swapchain.swap_images, 0..) |swap_image, index| {
        const attachments = [_]vk.ImageView{ swap_image.view, depth_view };
        framebuffers[index] = try device.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = attachments.len,
            .p_attachments = &attachments,
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        created += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(allocator: std.mem.Allocator, device: vk.DeviceProxy, framebuffers: []vk.Framebuffer) void {
    for (framebuffers) |framebuffer| {
        device.destroyFramebuffer(framebuffer, null);
    }
    allocator.free(framebuffers);
}
