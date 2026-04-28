const ash = @import("ash");
const std = @import("std");
const vk = ash.vk;

const math = ash.math;

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;

const depth_format: vk.Format = .d16_unorm;
const texture_path = "textures/gopher.png";

const UniformData = extern struct {
    mvp: math.Mat4,
    position: [36][4]f32,
    attr: [36][4]f32,
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
        for (self.buffers, self.memories, self.mapped, 0..) |*buffer, *memory, *mapped, i| {
            _ = i;
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

pub const CubeRenderer = struct {
    allocator: std.mem.Allocator,
    device: ?vk.DeviceProxy = null,
    texture: ash.ImageResource = .{},
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
    model_matrix: math.Mat4 = math.identity(),

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: std.mem.Allocator) CubeRenderer {
        return .{
            .allocator = allocator,
        };
    }

    pub fn createOnce(self: *CubeRenderer, session: *ash.Session) !void {
        const manager = &session.manager.?;
        const device = manager.device orelse return error.DeviceNotInitialized;
        const cmd_ctx = &session.cmd_ctx.?;
        self.device = device;

        self.texture = try ash.createTextureFromFile(
            self.allocator,
            manager,
            cmd_ctx,
            texture_path,
            .{},
        );
        self.uniforms = try UniformBuffers.init(self.allocator, manager, session.swapchain.?.imageCount());
        self.descriptor = try createDescriptors(self.allocator, device, self.uniforms.?, self.texture, session.swapchain.?.imageCount());

        self.model_matrix = math.identity();
        self.start_time = ash.glfw.getTime();
        self.once_built = true;
    }

    pub fn destroyOnce(self: *CubeRenderer) void {
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
        self.texture.deinit(device);
        self.device = null;
        self.once_built = false;
    }

    pub fn createSized(self: *CubeRenderer, session: *ash.Session, extent: vk.Extent2D) !void {
        const manager = &session.manager.?;
        const device = manager.device orelse return error.DeviceNotInitialized;

        self.depth = try createDepthImage(manager, extent.width, extent.height);
        errdefer self.depth.deinit(device);

        self.render_pass = try createRenderPass(device, session.swapchain.?.surface_format.format);
        errdefer device.destroyRenderPass(self.render_pass, null);

        self.pipeline_layout = try createPipelineLayout(device, self.descriptor.?.layout);
        errdefer device.destroyPipelineLayout(self.pipeline_layout, null);

        self.pipeline = try createPipeline(device, extent, self.pipeline_layout, self.render_pass);
        errdefer device.destroyPipeline(self.pipeline, null);

        self.framebuffers = try createFramebuffers(
            self.allocator,
            device,
            session.swapchain.?,
            self.render_pass,
            self.depth.view,
        );
        errdefer destroyFramebuffers(self.allocator, device, self.framebuffers);

        // Camera + projection — kept in sync with the Go reference
        // (vulkan-examples-go/05_cube). Two extent-driven adjustments:
        //   1. portrait dolly: at aspect < 1 the horizontal FOV gets too narrow
        //      for a 2x2x2 cube, so we pull the camera back by 1/aspect.
        //   2. preTransform: the surface rotation is folded into the projection
        //      so the Android compositor doesn't rotate the framebuffer.
        const aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
        const scale: f32 = if (aspect < 1.0) 1.0 / aspect else 1.0;
        self.view_matrix = math.lookAt(
            .{ .x = 0, .y = 3 * scale, .z = 5 * scale },
            .{ .x = 0, .y = 0, .z = 0 },
            .{ .x = 0, .y = 1, .z = 0 },
        );

        var proj = math.perspective(math.degreesToRadians(45.0), aspect, 0.1, 100.0);
        proj[1][1] *= -1;
        const pre_rot = session.swapchain.?.preRotationMatrix();
        self.proj_matrix = math.multiply(&pre_rot, &proj);

        self.sized_built = true;
    }

    pub fn destroySized(self: *CubeRenderer) void {
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

    pub fn draw(self: *CubeRenderer, session: *ash.Session, frame: *const ash.Frame) !void {
        const device = session.manager.?.device orelse return error.DeviceNotInitialized;

        const angle = math.degreesToRadians(@floatCast((ash.glfw.getTime() - self.start_time) * 45.0));
        self.model_matrix = math.rotateY(&math.identity(), angle);
        self.writeUniforms(frame.image_index);

        const clear_values = [_]vk.ClearValue{
            .{ .color = .{ .float_32 = .{ 0.2, 0.2, 0.2, 1.0 } } },
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
        device.cmdDraw(frame.cmd, 36, 1, 0, 0);
        device.cmdEndRenderPass(frame.cmd);
    }

    fn writeUniforms(self: *CubeRenderer, index: u32) void {
        const vp = math.multiply(&self.proj_matrix, &self.view_matrix);
        const mvp = math.multiply(&vp, &self.model_matrix);

        var data = UniformData{
            .mvp = mvp,
            .position = undefined,
            .attr = undefined,
        };
        for (0..36) |i| {
            data.position[i] = .{
                vertex_buffer_data[i * 3],
                vertex_buffer_data[i * 3 + 1],
                vertex_buffer_data[i * 3 + 2],
                1.0,
            };
            data.attr[i] = .{
                uv_buffer_data[i * 2],
                uv_buffer_data[i * 2 + 1],
                0.0,
                0.0,
            };
        }
        self.uniforms.?.update(index, &data);
    }
};

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
    texture: ash.ImageResource,
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
        .{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
            .descriptor_count = 1,
            .stage_flags = .{ .fragment_bit = true },
        },
    };
    descriptor.layout = try device.createDescriptorSetLayout(&.{
        .binding_count = bindings.len,
        .p_bindings = &bindings,
    }, null);

    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .uniform_buffer, .descriptor_count = @intCast(set_count) },
        .{ .type = .combined_image_sampler, .descriptor_count = @intCast(set_count) },
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
        const image_info = vk.DescriptorImageInfo{
            .sampler = texture.sampler,
            .image_view = texture.view,
            .image_layout = .shader_read_only_optimal,
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
            .{
                .dst_set = set,
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .combined_image_sampler,
                .p_buffer_info = undefined,
                .p_image_info = @ptrCast(&image_info),
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
    extent: vk.Extent2D,
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

    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 0,
        .p_vertex_binding_descriptions = null,
        .vertex_attribute_description_count = 0,
        .p_vertex_attribute_descriptions = null,
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
        .front_face = .clockwise,
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

    _ = extent;
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

const vertex_buffer_data = [_]f32{
    -1, -1, -1, -1, -1, 1,  -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1,
    -1, -1, -1, 1,  1,  -1, 1,  -1, -1, -1, -1, -1, -1, 1,  -1, 1,  1,  -1,
    -1, -1, -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, 1,  -1, -1, 1,
    -1, 1,  -1, -1, 1,  1,  1,  1,  1,  -1, 1,  -1, 1,  1,  1,  1,  1,  -1,
    1,  1,  -1, 1,  1,  1,  1,  -1, 1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1,
    -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,
};

const uv_buffer_data = [_]f32{
    0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
    1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
    1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0,
    0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
};
