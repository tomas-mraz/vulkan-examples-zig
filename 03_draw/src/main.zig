const ash = @import("ash");
const std = @import("std");
const glfw = ash.glfw;
const vk = ash.vk;

const app_name = "VulkanDraw";
const window_width = 640;
const window_height = 480;

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;

const Vertex = extern struct {
    pos: [3]f32,

    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
    };
};

const triangle_vertices = [_]Vertex{
    .{ .pos = .{ -1.0, -1.0, 0.0 } },
    .{ .pos = .{ 1.0, -1.0, 0.0 } },
    .{ .pos = .{ 0.0, 1.0, 0.0 } },
};

const BufferResource = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,

    fn deinit(self: BufferResource, manager: *const ash.Manager) void {
        const device = manager.device orelse return;
        device.destroyBuffer(self.buffer, null);
        device.freeMemory(self.memory, null);
    }
};

const SizedResources = struct {
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    framebuffers: []vk.Framebuffer,
    command_buffers: []vk.CommandBuffer,

    fn deinit(self: SizedResources, manager: *const ash.Manager, command_pool: vk.CommandPool, allocator: std.mem.Allocator) void {
        const device = manager.device orelse return;
        device.freeCommandBuffers(command_pool, self.command_buffers);
        allocator.free(self.command_buffers);

        for (self.framebuffers) |framebuffer| {
            device.destroyFramebuffer(framebuffer, null);
        }
        allocator.free(self.framebuffers);

        device.destroyPipeline(self.pipeline, null);
        device.destroyRenderPass(self.render_pass, null);
    }
};

fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}", .{ error_code, description });
}

pub fn main() !void {
    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{
        .platform = .wayland,
    })) {
        std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
        return error.GlfwInitFailed;
    }
    defer glfw.terminate();

    const window = glfw.Window.create(window_width, window_height, app_name, null, null, .{
        .client_api = .no_api,
    }) orelse {
        std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
        return error.WindowCreationFailed;
    };
    defer window.destroy();

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = try ash.Manager.initGlfw(allocator, app_name, window, .{});
    defer manager.destroy();

    std.debug.print("Using device: {s}\n", .{manager.deviceName()});

    var framebuffer_size = window.getFramebufferSize();
    var extent = vk.Extent2D{
        .width = framebuffer_size.width,
        .height = framebuffer_size.height,
    };

    var swapchain = try ash.Swapchain.init(&manager, allocator, extent);
    defer swapchain.deinit();
    extent = swapchain.extent;

    const pipeline_layout = try manager.device.?.createPipelineLayout(&.{
        .set_layout_count = 0,
        .p_set_layouts = undefined,
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer manager.device.?.destroyPipelineLayout(pipeline_layout, null);

    const command_pool = try manager.device.?.createCommandPool(&.{
        .queue_family_index = manager.graphics_queue.family,
    }, null);
    defer manager.device.?.destroyCommandPool(command_pool, null);

    const vertex_buffer = try createVertexBuffer(&manager);
    defer vertex_buffer.deinit(&manager);

    var sized = try createSizedResources(&manager, allocator, command_pool, pipeline_layout, swapchain, vertex_buffer.buffer);
    defer sized.deinit(&manager, command_pool, allocator);

    var present_state: ash.Swapchain.PresentState = .optimal;
    while (!window.shouldClose()) {
        framebuffer_size = window.getFramebufferSize();
        if (framebuffer_size.width == 0 or framebuffer_size.height == 0) {
            glfw.pollEvents();
            continue;
        }

        const desired_extent = vk.Extent2D{
            .width = framebuffer_size.width,
            .height = framebuffer_size.height,
        };

        if (present_state == .suboptimal or desired_extent.width != extent.width or desired_extent.height != extent.height) {
            try manager.device.?.deviceWaitIdle();

            sized.deinit(&manager, command_pool, allocator);
            try swapchain.recreate(desired_extent);
            sized = try createSizedResources(&manager, allocator, command_pool, pipeline_layout, swapchain, vertex_buffer.buffer);
            extent = swapchain.extent;
            present_state = .optimal;
        }

        present_state = swapchain.present(sized.command_buffers[swapchain.image_index]) catch |err| switch (err) {
            error.OutOfDateKHR => .suboptimal,
            else => return err,
        };

        glfw.pollEvents();
    }

    try swapchain.waitForAllFences();
    try manager.device.?.deviceWaitIdle();
}

fn createVertexBuffer(manager: *const ash.Manager) !BufferResource {
    const device = manager.device orelse return error.DeviceNotInitialized;

    const buffer = try device.createBuffer(&.{
        .size = @sizeOf(@TypeOf(triangle_vertices)),
        .usage = .{ .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    errdefer device.destroyBuffer(buffer, null);

    const requirements = device.getBufferMemoryRequirements(buffer);
    const memory = try manager.allocate(requirements, .{
        .host_visible_bit = true,
        .host_coherent_bit = true,
    });
    errdefer device.freeMemory(memory, null);

    try device.bindBufferMemory(buffer, memory, 0);

    const mapped = try device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{});
    defer device.unmapMemory(memory);

    const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(mapped));
    @memcpy(gpu_vertices, triangle_vertices[0..]);

    return .{
        .buffer = buffer,
        .memory = memory,
    };
}

fn createSizedResources(
    manager: *const ash.Manager,
    allocator: std.mem.Allocator,
    command_pool: vk.CommandPool,
    pipeline_layout: vk.PipelineLayout,
    swapchain: ash.Swapchain,
    vertex_buffer: vk.Buffer,
) !SizedResources {
    const render_pass = try createRenderPass(manager, swapchain);
    errdefer manager.device.?.destroyRenderPass(render_pass, null);

    const pipeline = try createPipeline(manager, pipeline_layout, render_pass);
    errdefer manager.device.?.destroyPipeline(pipeline, null);

    const framebuffers = try createFramebuffers(manager, allocator, render_pass, swapchain);
    errdefer {
        for (framebuffers) |framebuffer| manager.device.?.destroyFramebuffer(framebuffer, null);
        allocator.free(framebuffers);
    }

    const command_buffers = try createCommandBuffers(manager, allocator, command_pool, vertex_buffer, swapchain.extent, render_pass, pipeline, framebuffers);
    errdefer {
        manager.device.?.freeCommandBuffers(command_pool, command_buffers);
        allocator.free(command_buffers);
    }

    return .{
        .render_pass = render_pass,
        .pipeline = pipeline,
        .framebuffers = framebuffers,
        .command_buffers = command_buffers,
    };
}

fn createRenderPass(manager: *const ash.Manager, swapchain: ash.Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    return try manager.device.?.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
    }, null);
}

fn createPipeline(manager: *const ash.Manager, layout: vk.PipelineLayout, render_pass: vk.RenderPass) !vk.Pipeline {
    const device = manager.device orelse return error.DeviceNotInitialized;

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
        .{
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
        },
    };

    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_descriptions.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_descriptions,
    };

    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };

    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = undefined,
        .scissor_count = 1,
        .p_scissors = undefined,
    };

    const rasterization = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
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
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
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
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const create_info = vk.GraphicsPipelineCreateInfo{
        .stage_count = shader_stages.len,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &input_assembly,
        .p_tessellation_state = null,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic_state,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try device.createGraphicsPipelines(.null_handle, &.{create_info}, null, (&pipeline)[0..1]);
    return pipeline;
}

fn createFramebuffers(
    manager: *const ash.Manager,
    allocator: std.mem.Allocator,
    render_pass: vk.RenderPass,
    swapchain: ash.Swapchain,
) ![]vk.Framebuffer {
    const device = manager.device orelse return error.DeviceNotInitialized;
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var index: usize = 0;
    errdefer for (framebuffers[0..index]) |framebuffer| device.destroyFramebuffer(framebuffer, null);

    for (framebuffers) |*framebuffer| {
        framebuffer.* = try device.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[index].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        index += 1;
    }

    return framebuffers;
}

fn createCommandBuffers(
    manager: *const ash.Manager,
    allocator: std.mem.Allocator,
    command_pool: vk.CommandPool,
    vertex_buffer: vk.Buffer,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    framebuffers: []vk.Framebuffer,
) ![]vk.CommandBuffer {
    const device = manager.device orelse return error.DeviceNotInitialized;
    const command_buffers = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(command_buffers);

    try device.allocateCommandBuffers(&.{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = @intCast(command_buffers.len),
    }, command_buffers.ptr);
    errdefer device.freeCommandBuffers(command_pool, command_buffers);

    const clear_value = vk.ClearValue{
        .color = .{ .float_32 = .{ 0.098, 0.71, 0.996, 1.0 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(extent.width),
        .height = @floatFromInt(extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    for (command_buffers, framebuffers) |command_buffer, framebuffer| {
        try device.beginCommandBuffer(command_buffer, &.{});

        device.cmdSetViewport(command_buffer, 0, &.{viewport});
        device.cmdSetScissor(command_buffer, 0, &.{scissor});

        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        };

        device.cmdBeginRenderPass(command_buffer, &.{
            .render_pass = render_pass,
            .framebuffer = framebuffer,
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear_value),
        }, .@"inline");

        device.cmdBindPipeline(command_buffer, .graphics, pipeline);
        const offsets = [_]vk.DeviceSize{0};
        device.cmdBindVertexBuffers(command_buffer, 0, &.{vertex_buffer}, &offsets);
        device.cmdDraw(command_buffer, triangle_vertices.len, 1, 0, 0);
        device.cmdEndRenderPass(command_buffer);

        try device.endCommandBuffer(command_buffer);
    }

    return command_buffers;
}
